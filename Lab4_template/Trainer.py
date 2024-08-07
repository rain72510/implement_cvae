import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder

from dataloader import Dataset_Dance
from torchvision.utils import save_image
import random
import torch.optim as optim
from torch import stack

from tqdm import tqdm
import imageio

import matplotlib.pyplot as plt
from math import log10

def Generate_PSNR(imgs1, imgs2, data_range=1.):
    """PSNR for torch tensor"""
    mse = nn.functional.mse_loss(imgs1, imgs2) # wrong computation for batch size > 1
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr


def kl_criterion(mu, logvar, batch_size):
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  KLD /= batch_size  
  return KLD


class kl_annealing():
    def __init__(self, args, current_epoch=0):
        # TODO
        assert args.kl_anneal_type in ['Cyclical', 'Monotonic', 'none']
        self.kl_anneal_type = args.kl_anneal_type
        self.kl_anneal_cycle = args.kl_anneal_cycle # period of cycles
        self.kl_anneal_ratio = args.kl_anneal_ratio 
        self.current_epoch = current_epoch
        self.beta = 0.2

        # raise NotImplementedError
        
    def update(self):
        self.current_epoch += 1

        if self.kl_anneal_type == 'Cyclical':
            self.beta = self.frange_cycle_linear(self.current_epoch, n_cycle=self.kl_anneal_cycle, ratio=self.kl_anneal_ratio)
        elif self.kl_anneal_type == 'Monotonic':
            if self.current_epoch < self.kl_anneal_cycle:
                self.beta = self.frange_cycle_linear(self.current_epoch, n_cycle=self.kl_anneal_cycle, ratio=self.kl_anneal_ratio)
            else:
                self.beta = 1.0
        else:
            self.beta = 1.0

        # raise NotImplementedError
    
    def get_beta(self):
        return self.beta

    def frange_cycle_linear(self, n_iter, start=0.2, stop=1.0, n_cycle=1, ratio=1):
        # 0 to 1 in n_cycle * ratio, then 1 in n_cycle * (1-ratio)
        return max(start, min(stop, start + (stop - start) * min((n_iter % (n_cycle * ratio))/(n_cycle * ratio), 1.0)))
        raise NotImplementedError
        

class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args
        
        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)
        
        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor   = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion       = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)
        
        # Generative model
        self.Generator            = Generator(input_nc=args.D_out_dim, output_nc=3)
        
        self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
        self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 5], gamma=0.1)
        self.kl_annealing = kl_annealing(args, current_epoch=0)
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0
        
        # Teacher forcing arguments
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde
        
        self.train_vi_len = args.train_vi_len
        self.val_vi_len   = args.val_vi_len
        self.batch_size = args.batch_size

        self.log = {
            'train': {
                'loss': [],
                'mse': [],
                'kld': []
            },
            'val': {
                'loss': [],
                'mse': [],
                'kld': [],
                'psnr_per_frame': []
            }
        }
        
        
    def forward(self, img, label):
        pass

    def draw_logs(self):
        plt.figure()
        plt.plot(self.log['train']['loss'], label='train_loss')
        plt.plot(self.log['val']['loss'], label='val_loss')
        plt.legend()
        plt.title('Loss')
        plt.savefig(os.path.join(self.args.save_root, 'loss.png'))
        plt.clf()

        plt.figure()
        plt.plot(self.log['train']['mse'], label='train_mse')
        plt.plot(self.log['val']['mse'], label='val_mse')
        plt.legend()
        plt.title('MSE')
        plt.savefig(os.path.join(self.args.save_root, 'mse.png'))
        plt.clf()
        
        plt.figure()
        plt.plot(self.log['train']['kld'], label='train_kld')
        plt.plot(self.log['val']['kld'], label='val_kld')
        plt.legend()
        plt.title('KLD')
        plt.savefig(os.path.join(self.args.save_root, 'kld.png'))
        plt.clf()
        
        plt.figure()
        plt.plot(self.log['val']['psnr_per_frame'][-1], label='val_psnr')
        plt.legend()
        plt.title('PSNR')
        plt.savefig(os.path.join(self.args.save_root, 'psnr.png'))
        plt.clf()
    
    def training_stage(self):
        self.frame_transformation.train()
        self.label_transformation.train()
        self.Gaussian_Predictor.train()
        self.Decoder_Fusion.train()
        self.Generator.train()

        logfile = open(os.path.join(self.args.save_root, 'log.txt'), 'a')
        for i in range(self.args.num_epoch):
            train_loader = self.train_dataloader()
            adapt_TeacherForcing = True if random.random() < self.tfr else False
            
            cnt_o = 0
            for (img, label) in (pbar := tqdm(train_loader, ncols=120)):
                img = img.to(self.args.device)
                label = label.to(self.args.device)
                loss, mse, kld = self.training_one_step(img, label, adapt_TeacherForcing, cnt_o)
                cnt_o += 1
                self.log['train']['loss'].append(loss.detach().cpu())
                self.log['train']['mse'].append(mse.detach().cpu())
                self.log['train']['kld'].append(kld.detach().cpu())
                print("cnt_o: ", cnt_o, "total_mse: ", self.log['train']['loss'][-1], flush=True)
                print("cnt_o: ", cnt_o, "total_kld: ", self.log['train']['mse'][-1], flush=True)
                print("cnt_o: ", cnt_o, "total_loss: ", self.log['train']['kld'][-1], flush=True)
                beta = self.kl_annealing.get_beta()
                if adapt_TeacherForcing:
                    self.tqdm_bar('train [TeacherForcing: ON, {:.1f}], beta: {}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
                else:
                    self.tqdm_bar('train [TeacherForcing: OFF, {:.1f}], beta: {}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
                
            
            if self.current_epoch % self.args.per_save == 0:
                self.save(os.path.join(self.args.save_root, f"epoch={self.current_epoch}.ckpt"))
            # append log
            print(f"epoch: {i} [Train] loss: {self.log['train']['loss'][-1]}, mse: {self.log['train']['mse'][-1]}, kld: {self.log['train']['kld'][-1]}", file=logfile, flush=True)

            self.eval()
            self.current_epoch += 1
            self.scheduler.step()
            self.teacher_forcing_ratio_update()
            self.kl_annealing.update()

            print(f"epoch: {i} [Valid] loss: {self.log['val']['loss'][-1]}, mse: {self.log['val']['mse'][-1]}, kld: {self.log['val']['kld'][-1]}", file=logfile, flush=True)
            
        self.draw_logs()
            
            
    @torch.no_grad()
    def eval(self):
        val_loader = self.val_dataloader()
        self.frame_transformation.eval()
        self.label_transformation.eval()
        self.Gaussian_Predictor.eval()
        self.Decoder_Fusion.eval()
        self.Generator.eval()
        for (img, label) in (pbar := tqdm(val_loader, ncols=120)):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            loss, mse, kld, psnrs, preds = self.val_one_step(img, label)
            self.tqdm_bar('val', pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
            self.log['val']['loss'].append(loss.detach().cpu())
            self.log['val']['mse'].append(mse.detach().cpu())
            self.log['val']['kld'].append(kld.detach().cpu())
            self.log['val']['psnr_per_frame'].append(psnrs)
            print("img_shape: ", img.shape, flush=True)
        imgs = []
        for img in preds:
            img = img[0].detach().cpu()
            imgs.append(img)
        self.make_gif(imgs, os.path.join(self.args.save_root, f"epoch={self.current_epoch}.gif"))
        
    
    def training_one_step(self, img, label, adapt_TeacherForcing, cnt_o):
        img, label = img.transpose(0, 1), label.transpose(0, 1)
        code_img = []
        code_label = []
        for i in range(self.train_vi_len):
            code_img.append(self.frame_transformation(img[i]))
            code_label.append(self.label_transformation(label[i]))
        code_img = stack(code_img)
        code_label = stack(code_label)
        total_loss = 0.0
        total_mse = 0.0
        total_kld = 0.0
        pred = None
        beta = self.kl_annealing.get_beta()


        output_rem = None
        pred_rem = None
        mu_rem = None
        logvar_rem = None
        for i in range(self.train_vi_len - 1):
            z, mu, logvar = self.Gaussian_Predictor.forward(code_img[i+1], code_label[i+1])
            if adapt_TeacherForcing or not pred:
                output = self.Decoder_Fusion.forward(code_img[i], code_label[i+1], z)
            else:
                output = self.Decoder_Fusion.forward(pred, code_label[i+1], z)
            pred = self.Generator.forward(output)
            # if pred > 1 or pred < 0, clip it:
            pred = torch.clamp(pred, 0, 1)

            if i == 0:
                output_rem = output
                pred_rem = pred
                mu_rem = mu
                logvar_rem = logvar
            mse = self.mse_criterion(pred, img[i+1])
            kld = kl_criterion(mu, logvar, self.batch_size) / (self.args.frame_H * self.args.frame_W * self.args.N_dim)
            # loss = mse + kld * beta
            total_mse += mse
            total_kld += kld

            # mse nan check
            if mse > 10 or mse != mse:
                print("epoch:", self.current_epoch, " i: ", i, "mse: ", mse, flush=True)
                print("mu: ", mu, flush=True)
                print("logvar: ", logvar, flush=True)
                print("mu_rem: ", mu_rem, flush=True)
                print("logvar_rem: ", logvar_rem, flush=True)
                print("output_shape: ", output.shape, flush=True)
                print("output: ", output, flush=True)
                print("output_rem: ", output_rem, flush=True)
                print("pred_shape: ", pred.shape, flush=True)
                print("pred: ", pred, flush=True)
                print("pred_rem: ", pred_rem, flush=True)
                print("img_shape: ", img[i+1].shape, flush=True)
                print("img: ", img[i+1], flush=True)

            pred = self.frame_transformation(pred)
            # loss.backward()
        total_mse /= (self.train_vi_len - 1)
        total_kld /= (self.train_vi_len - 1)
        total_loss = total_mse + total_kld * beta

        self.optim.zero_grad()
        total_loss.backward()

        if total_mse > 10:
            print("explode epoch:", self.current_epoch, " i: ", i, "mse: ", mse, flush=True)
            # print maximum gradient
            max_grad = 0
            for name, param in self.Decoder_Fusion.named_parameters():
                if param.grad is not None:
                    max_grad = max(max_grad, param.grad.abs().max())
            print("Decoder_Fusion max_grad: ", max_grad, flush=True)
            max_grad = 0
            for name, param in self.Generator.named_parameters():
                if param.grad is not None:
                    max_grad = max(max_grad, param.grad.abs().max())
            print("Generator max_grad: ", max_grad, flush=True)


        nn.utils.clip_grad_norm_(self.parameters(), 1.)
        if total_mse > 10:
            print("explode epoch:", self.current_epoch, " i: ", i, "mse: ", mse, flush=True)
            # print maximum gradient
            max_grad = 0
            for name, param in self.Decoder_Fusion.named_parameters():
                if param.grad is not None:
                    max_grad = max(max_grad, param.grad.abs().max())
            print("Decoder_Fusion max_grad: ", max_grad, flush=True)
            max_grad = 0
            for name, param in self.Generator.named_parameters():
                if param.grad is not None:
                    max_grad = max(max_grad, param.grad.abs().max())
            print("Generator max_grad: ", max_grad, flush=True)
        self.optim.step()
        # self.optimizer_step()
        return total_loss, total_mse, total_kld
        
    
    def val_one_step(self, img, label):
        img, label = img.transpose(0, 1), label.transpose(0, 1)
        code_img = []
        code_label = []
        for i in range(self.val_vi_len):
            code_img.append(self.frame_transformation(img[i]))
            code_label.append(self.label_transformation(label[i]))
        code_img = stack(code_img)
        code_label = stack(code_label)
        total_loss = 0.0
        total_mse = 0.0
        total_kld = 0.0
        pred = code_img[0]
        psnrs = []
        preds = [img[0]]
        for i in range(self.val_vi_len - 1):
            z, mu, logvar = self.Gaussian_Predictor.forward(code_img[i+1], code_label[i+1])
            output = self.Decoder_Fusion.forward(pred, code_label[i+1], z)
            pred = self.Generator.forward(output)
            pred = torch.clamp(pred, 0, 1)
            preds.append(pred)
            psnr = Generate_PSNR(pred, img[i+1])
            psnrs.append(psnr)
            loss_mse = self.mse_criterion(pred, img[i+1])
            KLD = kl_criterion(mu, logvar, 1) / (self.args.frame_H * self.args.frame_W * self.args.N_dim)
            total_mse += loss_mse
            total_kld += KLD
            pred = self.frame_transformation(pred)
        
        beta = self.kl_annealing.get_beta()
        total_mse /= (self.val_vi_len - 1)
        total_kld /= (self.val_vi_len - 1)
        total_loss = total_mse + total_kld * beta
        return total_loss, total_mse, total_kld, psnrs, preds
                
    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))
            
        new_list[0].save(img_name, format="GIF", append_images=new_list,
                    save_all=True, duration=40, loop=0)
    
    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])

        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='train', video_len=self.train_vi_len, \
                                                partial=args.fast_partial if self.args.fast_train else args.partial)
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False
            
        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return train_loader
    
    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])
        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='val', video_len=self.val_vi_len, partial=1.0)  
        val_loader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return val_loader
    
    def teacher_forcing_ratio_update(self):
        if self.current_epoch > self.tfr_sde:
            self.tfr = max(0.0, self.tfr - self.tfr_d_step)
            
    def tqdm_bar(self, mode, pbar, loss, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr}" , refresh=False)
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()
        
    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),
            "optimizer": self.state_dict(),  
            "lr"        : self.scheduler.get_last_lr()[0],
            "tfr"       :   self.tfr,
            "last_epoch": self.current_epoch
        }, path)
        print(f"save ckpt to {path}")

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint['state_dict'], strict=True) 
            self.args.lr = checkpoint['lr']
            self.tfr = checkpoint['tfr']
            
            self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
            self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 4], gamma=0.1)
            self.kl_annealing = kl_annealing(self.args, current_epoch=checkpoint['last_epoch'])
            self.current_epoch = checkpoint['last_epoch']

    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.)
        for name, param in self.Decoder_Fusion.named_parameters():
            print(name, param.grad, flush=True)
        for name, param in self.Generator.named_parameters():
            print(name, param.grad, flush=True)

        self.optim.step()



def main(args):
    os.makedirs(args.save_root, exist_ok=True)
    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()
    if args.test:
        model.eval()
    else:
        model.training_stage()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=2)
    parser.add_argument('--lr',            type=float,  default=0.001,     help="initial learning rate")
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--gpu',           type=int, default=1)
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--store_visualization',      action='store_true', help="If you want to see the result while training")
    parser.add_argument('--DR',            type=str, required=True,  help="Your Dataset Path")
    parser.add_argument('--save_root',     type=str, required=True,  help="The path to save your data")
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--num_epoch',     type=int, default=70,     help="number of total epoch")
    parser.add_argument('--per_save',      type=int, default=3,      help="Save checkpoint every seted epoch")
    parser.add_argument('--partial',       type=float, default=1.0,  help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len',  type=int, default=16,     help="Training video length")
    parser.add_argument('--val_vi_len',    type=int, default=630,    help="valdation video length")
    parser.add_argument('--frame_H',       type=int, default=32,     help="Height input image to be resize")
    parser.add_argument('--frame_W',       type=int, default=64,     help="Width input image to be resize")
    
    
    # Module parameters setting
    parser.add_argument('--F_dim',         type=int, default=128,    help="Dimension of feature human frame")
    parser.add_argument('--L_dim',         type=int, default=32,     help="Dimension of feature label frame")
    parser.add_argument('--N_dim',         type=int, default=12,     help="Dimension of the Noise")
    parser.add_argument('--D_out_dim',     type=int, default=192,    help="Dimension of the output in Decoder_Fusion")
    
    # Teacher Forcing strategy
    parser.add_argument('--tfr',           type=float, default=1.0,  help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde',       type=int,   default=10,   help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step',    type=float, default=0.1,  help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--ckpt_path',     type=str,    default=None,help="The path of your checkpoints")   
    
    # Training Strategy
    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float, default=0.4,    help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch',   type=int, default=5,        help="Number of epoch to use fast train mode")
    
    # Kl annealing stratedy arguments
    parser.add_argument('--kl_anneal_type',     type=str, default='Cyclical',       help="")
    parser.add_argument('--kl_anneal_cycle',    type=int, default=10,               help="")
    parser.add_argument('--kl_anneal_ratio',    type=float, default=1,              help="")
    

    

    args = parser.parse_args()
    
    main(args)
