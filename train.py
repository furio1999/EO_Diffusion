import torch
import torch.nn as nn
from torchvision.utils import save_image
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import *
from diffusion.model import EODiffusion
from backbones.unet_openai import UNetModel
from utils import ExponentialMovingAverage
import os
import math
import argparse
from data import *
from pytorch_lightning.loggers import WandbLogger
import wandb, pdb
from tqdm import tqdm
from train_utils import KeyframeLR



def parse_args():
    parser = argparse.ArgumentParser(description="Training MNISTDiffusion")
    parser.add_argument('--lr',type = float ,default=0.001) # default 0.001
    parser.add_argument('--batch_size',type = int ,default=128)    
    parser.add_argument('--epochs',type = int,default=100)
    parser.add_argument('--dir',type = str,help = 'directory',default='results/prova')
    parser.add_argument('--ckpt',type = str,help = 'define checkpoint path',default='')
    parser.add_argument('--n_samples',type = int,help = 'define sampling amounts after every epoch trained',default=16)
    parser.add_argument('--model_base_dim',type = int,help = 'base dim of Unet',default=64)
    parser.add_argument('--timesteps',type = int,help = 'sampling steps of DDPM',default=1000)
    parser.add_argument('--model_ema_steps',type = int,help = 'ema model evaluation interval',default=10)
    parser.add_argument('--model_ema_decay',type = float,help = 'ema model decay',default=0.995)
    parser.add_argument('--log_freq',type = int,help = 'training log message printing frequence',default=10)
    parser.add_argument('--no_clip',action='store_true',help = 'set to normal sampling method without clip x_0 which could yield unstable samples')
    parser.add_argument('--cpu',action='store_true',help = 'cpu training')
    parser.add_argument('--wandb',action='store_true',help = 'wandb logger usage')
    parser.add_argument('--num_classes',type = int,help = 'conditional training',default=0)

    args = parser.parse_args()

    return args


def main(args):
    device="cpu" if args.cpu else "cuda:0"
    image_size = 64 # 256 - bs 8
    num_classes = args.num_classes if args.num_classes > 0 else None
    in_channels,cond_channels,out_channels=3,0,3
    base_dim, dim_mults, attention_resolutions,num_res_blocks, num_heads=128,[1,2,3,4],[],1,1
    train_dataloader,test_dataloader=create_Eurosat_dataloaders(batch_size=args.batch_size, num_workers=4
                    )
    l,bs = len(train_dataloader), min(train_dataloader.batch_size,len(train_dataloader))

    unet = UNetModel(image_size, in_channels=in_channels+cond_channels, model_channels=base_dim, out_channels=out_channels, channel_mult=dim_mults, 
                     attention_resolutions=attention_resolutions,num_res_blocks=num_res_blocks, num_heads=num_heads, num_classes=num_classes)
    model=EODiffusion(unet,
                timesteps=args.timesteps,
                image_size=image_size,
                in_channels=in_channels
                ).to(device)
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"Diffusion with {trainable_params/1e06} M params")
    #trainable_params_unet = sum(param.numel() for param in model.model.parameters() if param.requires_grad)
    #params_unet = len([param for param in model.model.diffusion_model.parameters()])
    #layers_unet = [layer for layer in model.model.diffusion_model.children()]

    #torchvision ema setting
    #https://github.com/pytorch/vision/blob/main/references/classification/train.py#L317
    adjust = 1* args.batch_size * args.model_ema_steps / args.epochs
    alpha = 1.0 - args.model_ema_decay
    alpha = min(1.0, alpha * adjust)
    model_ema = ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)

    optimizer=AdamW(model.parameters(),lr=args.lr)
    max_steps, posmax, end_lr = len(train_dataloader)*args.epochs, 10*len(train_dataloader), 1e-06
    scheduler = KeyframeLR(optimizer=optimizer, units="steps",
    frames=[
        {"position": 0, "lr": args.lr/100},
        {"transition": "cos"},
        {"position": posmax, "lr": args.lr},
        {"transition": lambda last_lr, sf, ef, pos, *_: args.lr * math.exp(-3*(pos-posmax)/(max_steps-posmax))},
    ],
    end=max_steps,
)
    loss_fn=nn.MSELoss(reduction='mean')
    #logger = WandbLogger(project="EO-minimal-diffusion", name=f"mnistdiffusion", log_model=True)
    #logger.experiment.define_metric("global_step")
    #logger.experiment.define_metric("train/*", step_metric="global_step")
    if args.wandb: wandb.init(project="EO-minimal-diffusion")


    #load checkpoint
    if args.ckpt:
        print("Loading checkpoint...")
        ckpt=torch.load(args.ckpt)
        model_ema.load_state_dict(ckpt["model_ema"])
        model.load_state_dict(ckpt["model"])

    global_steps,best_loss=0,0.9
    cond, y_test = None, torch.full((args.n_samples,),1).to(device) if args.num_classes>0 else None
    dir = args.dir
    dir_ckpt = "logs/" + os.path.split(dir)[1]
    os.makedirs(dir,exist_ok=True), os.makedirs(dir_ckpt, exist_ok=True)
    ckpt_best = os.path.join(dir_ckpt, "best.pt") # do it into
    l = len(train_dataloader)
    for i in range(args.epochs):
        model.train()
        for j,(data) in (enumerate(train_dataloader)):
            print(f"step {j}/{int(l)}")
            image = data["image"]
            target = target.to(device) if y_test is not None else None
            cond = cond.to(device) if cond is not None else None
            noise=torch.randn_like(image).to(device)
            image=image.to(device)
            pred=model(image,noise, cond=cond, y=target)
            loss=loss_fn(pred,noise)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            if global_steps%args.model_ema_steps==0:
                model_ema.update_parameters(model)
            global_steps+=1 # = (i+1)*(j+1)
            if j%args.log_freq==0:
                #logger.experiment.log({"train/loss": loss, "train/learning_rate": scheduler.get_last_lr()[-1]})
                print("Epoch[{}/{}],Step[{}/{}],loss:{:.5f},lr:{:.5f}".format(i+1,args.epochs,j,len(train_dataloader),
                                                                    loss.detach().cpu().item(),scheduler.get_last_lr()[0]))
            if args.wandb:
                wandb.log({"loss":loss.detach().cpu().item()})
                wandb.log({"lr":scheduler.get_last_lr()[0]})

            if loss < best_loss: 
                torch.save(ckpt, ckpt_best)
                best_loss = loss.detach().cpu().item()
                
            ckpt={"model":model.state_dict(),
                    "model_ema":model_ema.state_dict()}
            
            print_idx = global_steps%200 if global_steps<1000 else (global_steps)%(1000)
            ckpt_path = os.path.join(dir_ckpt, "steps_{:0>8}.pt".format(global_steps))
            img_path = os.path.join(dir, "steps_{:0>8}.png".format(global_steps))
            cond_path = os.path.join(dir, "steps_{:0>8}_cond.png".format(global_steps))

            model_ema.eval()
            cond = cond[:args.n_samples] if cond is not None else None
            # put cond[randn_idx]*n_samples
            if (print_idx==0):
                samples=model_ema.module.sampling(args.n_samples,clipped_reverse_diffusion=not args.no_clip,device=device, cond=cond, y=y_test)
                samples = samples.clip(0,1) if image.min()>=0 else (samples+1.)/2 #samples = (samples+1.)/2. only if train data is in (-1,1)
                samples = F.adjust_brightness(samples, 3) if samples.mean()<0.2 else samples
                print(f"saving in {img_path}, idx {print_idx} epoch {i}")
                save_image(samples,img_path,nrow=int(math.sqrt(args.n_samples)))
                save_image(cond,cond_path,nrow=int(math.sqrt(args.n_samples))) if cond is not None else print()
                torch.save(ckpt,ckpt_path)

    if args.wandb: wandb.finish()

if __name__=="__main__":
    args=parse_args()
    main(args)