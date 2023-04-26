import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision import transforms 
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from model import MNISTDiffusion
from utils import ExponentialMovingAverage
import os
import math
import argparse
from data import *
from pytorch_lightning.loggers import WandbLogger



def parse_args():
    parser = argparse.ArgumentParser(description="Training MNISTDiffusion")
    parser.add_argument('--lr',type = float ,default=0.001) # default 0.001
    parser.add_argument('--batch_size',type = int ,default=128)    
    parser.add_argument('--epochs',type = int,default=100)
    parser.add_argument('--ckpt',type = str,help = 'define checkpoint path',default='')
    parser.add_argument('--n_samples',type = int,help = 'define sampling amounts after every epoch trained',default=36)
    parser.add_argument('--model_base_dim',type = int,help = 'base dim of Unet',default=64)
    parser.add_argument('--timesteps',type = int,help = 'sampling steps of DDPM',default=1000)
    parser.add_argument('--model_ema_steps',type = int,help = 'ema model evaluation interval',default=10)
    parser.add_argument('--model_ema_decay',type = float,help = 'ema model decay',default=0.995)
    parser.add_argument('--log_freq',type = int,help = 'training log message printing frequence',default=10)
    parser.add_argument('--no_clip',action='store_true',help = 'set to normal sampling method without clip x_0 which could yield unstable samples')
    parser.add_argument('--cpu',action='store_true',help = 'cpu training')
    parser.add_argument('--num_classes',type = int,help = 'num classes',default=None)
    parser.add_argument('--cond',type = int,help = 'num classes',default=None)

    args = parser.parse_args()

    return args

def main(args):
    device="cpu" if args.cpu else "cuda:0"
    train_dataloader,test_dataloader=create_mnist_dataloaders(batch_size=args.batch_size,image_size=28)
    y_test = torch.full((args.n_samples,),7).to(device) if args.num_classes>0 else None
    model=MNISTDiffusion(timesteps=args.timesteps,
                image_size=28,
                num_classes = args.num_classes,
                in_channels=1,
                out_channels = 1,
                base_dim=args.model_base_dim,
                dim_mults=[2,4]).to(device)

    if args.ckpt:
        print("loading checkpoint...")
        ckpt=torch.load(args.ckpt)
        print("loaded!")
        model.load_state_dict(ckpt["model"])
    
    for j,(image,target) in enumerate(test_dataloader):
            noise=torch.randn_like(image).to(device)
            image=image.to(device)

            dir="results/mnist_cond"
            ckpt_name = os.path.splitext(os.path.split(args.ckpt)[-1])[0]
            img_path = os.path.join(dir, "sample_{}_{}_class7.png".format(j,ckpt_name))
            os.makedirs(dir,exist_ok=True)

            model.eval()
            samples=model.sampling(args.n_samples,clipped_reverse_diffusion=not args.no_clip,device=device, y=y_test)
            save_image(samples,img_path,nrow=int(math.sqrt(args.n_samples)))

if __name__=="__main__":
    args=parse_args()
    main(args)