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
from backbones.unet_openai import UNetModel



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
    parser.add_argument('--cond',type = bool,help = 'conditional training',default=False)
    parser.add_argument('--num_classes',type = int,help = 'conditional training',default=0)

    args = parser.parse_args()

    return args


def main(args):
    device="cpu" if args.cpu else "cuda:0"
    train_dataloader,test_dataloader=create_mnist_dataloaders(batch_size=args.batch_size,image_size=28)
    num_classes = args.num_classes if args.num_classes > 0 else None
    image_size = 28
    in_channels,cond_channels,out_channels=1,0,1
    base_dim, dim_mults, attention_resolutions,num_res_blocks, num_heads=32,[2,4],[],1,1
    y_test = torch.full((args.n_samples,),2).to(device) if args.num_classes>0 else None # mnist cond result obtained without specifying the exact num_classes, only putting target=0...9
    unet = UNetModel(image_size, in_channels=in_channels+cond_channels, model_channels=base_dim, out_channels=out_channels, channel_mult=dim_mults, 
                     attention_resolutions=attention_resolutions,num_res_blocks=num_res_blocks, num_heads=num_heads, num_classes=num_classes)
    model=MNISTDiffusion(unet,
                timesteps=args.timesteps,
                image_size=image_size,
                in_channels=in_channels
                ).to(device)

    #torchvision ema setting
    #https://github.com/pytorch/vision/blob/main/references/classification/train.py#L317
    adjust = 1* args.batch_size * args.model_ema_steps / args.epochs
    alpha = 1.0 - args.model_ema_decay
    alpha = min(1.0, alpha * adjust)
    model_ema = ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)

    optimizer=AdamW(model.parameters(),lr=args.lr)
    scheduler=OneCycleLR(optimizer,args.lr,total_steps=args.epochs*len(train_dataloader),pct_start=0.25,anneal_strategy='cos')
    loss_fn=nn.MSELoss(reduction='mean')
    #logger = WandbLogger(project="EO-minimal-diffusion", name=f"mnistdiffusion", log_model=True)
    #logger.experiment.define_metric("global_step")
    #logger.experiment.define_metric("train/*", step_metric="global_step")

    #load checkpoint
    if args.ckpt:
        ckpt=torch.load(args.ckpt)
        model_ema.load_state_dict(ckpt["model_ema"])
        model.load_state_dict(ckpt["model"])

    global_steps=0
    for i in range(args.epochs):
        
        model.train()
        for j,(image,target) in enumerate(train_dataloader):
            breakpoint()
            noise=torch.randn_like(image).to(device)
            image, target=image.to(device), target.to(device)
            pred=model(image,noise, y=target)
            breakpoint()
            loss=loss_fn(pred,noise)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            if global_steps%args.model_ema_steps==0:
                model_ema.update_parameters(model)
            global_steps+=1
            if j%args.log_freq==0:
                #logger.experiment.log({"train/loss": loss, "train/learning_rate": scheduler.get_last_lr()[-1]})
                print("Epoch[{}/{}],Step[{}/{}],loss:{:.5f},lr:{:.5f}".format(i+1,args.epochs,j,len(train_dataloader),
                                                                    loss.detach().cpu().item(),scheduler.get_last_lr()[0]))
        ckpt={"model":model.state_dict(),
                "model_ema":model_ema.state_dict()}

        dir="results/prova"
        print_idx = i if i<10 else i%(args.epochs//25)
        ckpt_path = os.path.join(dir, "steps_{:0>8}.pt".format(global_steps))
        img_path = os.path.join(dir, "steps_{:0>8}.png".format(global_steps))
        os.makedirs(dir,exist_ok=True)

        model_ema.eval()
        samples=model_ema.module.sampling(args.n_samples,clipped_reverse_diffusion=not args.no_clip,device=device, y=y_test)
        if print_idx == i or print_idx==0 or print_idx==args.epochs-1:
          save_image(samples,img_path,nrow=int(math.sqrt(args.n_samples)))
          torch.save(ckpt, ckpt_path)

if __name__=="__main__":
    args=parse_args()
    main(args)