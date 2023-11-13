import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision import transforms 
from torchvision.utils import save_image
import torchvision.transforms.functional as F
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.functional import peak_signal_noise_ratio
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from diffusion.model import EODiffusion
from script_utils.utils import *
import os
import math
import argparse
from data_utils.data import *
from pytorch_lightning.loggers import WandbLogger
from PIL import Image
from backbones.unet_openai import UNetModel
from diffusion.ddim import DDIMSampler

def parse_args(def_arg=False):
    parser = argparse.ArgumentParser(description="Training MNISTDiffusion")
    parser.add_argument('--lr',type = float ,default=0.001) # default 0.001
    parser.add_argument('--batch_size',type = int ,default=4)    
    parser.add_argument('--sampler_steps',type = int ,default=250) 
    parser.add_argument('--outdir',type = str,help = 'directory',default='results/prova')
    parser.add_argument('--ckpt',type = str,help = 'define checkpoint path',default='')
    parser.add_argument('--model_base_dim',type = int,help = 'base dim of Unet',default=64)
    parser.add_argument('--timesteps',type = int,help = 'sampling steps of DDPM',default=1000)
    parser.add_argument('--model_ema_steps',type = int,help = 'ema model evaluation interval',default=10)
    parser.add_argument('--model_ema_decay',type = float,help = 'ema model decay',default=0.995)
    parser.add_argument('--log_freq',type = int,help = 'training log message printing frequence',default=10)
    parser.add_argument('--no_clip',action='store_true',help = 'set to normal sampling method without clip x_0 which could yield unstable samples')
    parser.add_argument('--cpu',action='store_true',help = 'cpu training')
    parser.add_argument('--metrics',action='store_true',help = 'cpu training')
    parser.add_argument('--save',action='store_true',help = 'cpu training') # no need it, if save_dir is None then do not save
    parser.add_argument('--random_label',action='store_true',help = 'random label')
    parser.add_argument('--wandb',action='store_true',help = 'wandb logger usage')
    parser.add_argument('--num_classes',type = int,help = 'conditional training',default=0)
    parser.add_argument('--cond_type',type = str,help = 'cond type',default=None)
    parser.add_argument('--sampler',type = str,help = 'sampler',default="ddpm")
    parser.add_argument('--samples_fid',action='store_true',help = 'cpu training')
    parser.add_argument('--n_iter',type = int,help = 'sampler',default=None)

    args = parser.parse_args() if not def_arg else parser.parse_args("")
    #ckpt_path = os.path.splitext(os.path.split(args.ckpt)[-1])[0]
    # n_samples substitute with batch_size, or delete n_iter

    return args

def main(args):
    to_pil = transforms.ToPILImage()
    device="cpu" if args.cpu else "cuda:1"
    torch.cuda.device(device)
    ngpu = torch.cuda.device_count()
    image_size = 64
    in_ch,cond_channels,out_ch=3,0,3
    base_dim, dim_mults, attention_resolutions,num_res_blocks, num_heads=128,[1,2,4,8],[],1,1
    train_dataloader,test_dataloader=create_inria_dataloaders(batch_size=args.batch_size, num_workers=4,test=True,
                    )
    num_classes = args.num_classes if args.num_classes > 0 else None
    unet = UNetModel(image_size, in_channels=in_ch+cond_channels, model_channels=base_dim, out_channels=out_ch, channel_mult=dim_mults, 
                     attention_resolutions=attention_resolutions,num_res_blocks=num_res_blocks, num_heads=num_heads, num_classes=num_classes)
    model=EODiffusion(unet,
                timesteps=args.timesteps,
                image_size=image_size,
                in_channels=in_ch,
                cond_type = args.cond_type,
                device = device,
                ).to(device)
    sampler = DDIMSampler(model)
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"Diffusion with {trainable_params/1e06} M params")
    #plot_params(sampler)


    if args.ckpt:
        print("loading checkpoint...")
        ckpt=torch.load(args.ckpt)
        for key in list(ckpt.keys()):
            if 'model.' in key:
                ckpt[key.replace('model.', '')] = ckpt[key]
                del ckpt[key]
        model.load_state_dict(ckpt["model"])
        print("loaded!")
    cond, y_test  = None, torch.full((args.batch_size,),0).to(device) if args.num_classes>0 else None
    classes = list({'austin':0,  'chicago':1, 'kitsap':2, 'tyrol':3, 'vienna':4}.keys()) # to be returned from loader 
    

    dir=args.outdir
    dir_samples_fid, dir_samples  = os.path.join(dir,"samples_fid"), os.path.join(dir,"samples")
    os.makedirs(dir,exist_ok=True), os.makedirs(dir_samples_fid,exist_ok=True), os.makedirs(dir_samples,exist_ok=True)
    offset = len(os.listdir(dir_samples)) if cond is None else len(os.listdir(dir_samples))//3
    print("start inference")
    ssim, psnr, n  = 0, 0, 0
    for j,(data) in enumerate(test_dataloader):
        print(f"data {j}")
        image, mask = data["image"], data["segmentation"] if args.cond_type is not None else None  # data[input_key], data[cond_key]
        if args.cond_type == "sum": mask = 1-mask # do it inside dataloader?
        if args.random_label and args.cond_type=="sum": 
            mask = make_label((image_size, image_size), 10, 10, 40, 40)
        if mask is not None:
            if mask.mean()>=0.9 or mask.mean()<=0.1: pass # original bounds 0.8 and 0.2. Do it at data level, no continue statement here
        if args.cond_type is not None: cond = mask.to(device)
        
        image = image.to(device) # cond as a vocabulary with mask too? class_label, mask, image, text
        if args.cond_type == "sum": cond = torch.cat((image,cond),dim=1)
        y_test  = torch.full((args.batch_size,),min(j%(num_classes-1),num_classes-1)).to(device) if args.num_classes>0 else None
        catg = classes[y_test[0]] if y_test is not None else "sample"


        idx = j + offset
        gt_path, cond_path, img_path = os.path.join(dir_samples, f"sample_{idx}_gt.png"), \
        os.path.join(dir_samples, f"sample_{idx}_cond.png"), \
        os.path.join(dir_samples, f"sample_{idx}.png") # check samples in [-1,1]

        model.eval()
        # put mask inside sampler, from outside only a cond input
        if args.sampler=="ddpm": 
            samples=model.sampling(args.batch_size,clipped_reverse_diffusion=not args.no_clip,device=device, 
        cond=cond, y=y_test, idx=0, save=True) 
        else: 
            samples,_ = sampler.sample(S=args.sampler_steps, batch_size=args.batch_size, mask=mask,
                    shape=(out_ch, image_size, image_size), conditioning=None, verbose=False)
        
        samples = samples.clip(0,1) if image.min()>=0 else (samples+1.)/2. #samples = (samples+1.)/2. only if train data is in (-1,1)
        print(image.min())
        breakpoint()
        #if not args.save: continue # goal: mini script for noise visualization
        
        if cond is not None:
            if mask is not None: cond = image*((mask+0.7).clip(0,1))
            (gt,cond) = (image, cond) if image.min()>=0 else ( (image+1.)/2., (cond+1.)/2. )
            if args.metrics:
                s, p = structural_similarity_index_measure(samples, gt, data_range=1.0), peak_signal_noise_ratio(samples, gt, data_range=1.0)
                ssim, psnr = ssim + s, psnr + p

            gt = F.adjust_brightness(gt, 3) if gt.mean()<0.2 else gt
            cond = F.adjust_brightness(cond, 3) if cond.mean()<0.2 and args.cond_type!="sum" else cond
            if args.save: save_image(gt, gt_path, nrow=int(math.sqrt(args.batch_size)))
            if args.save: save_image(cond, cond_path, nrow=int(math.sqrt(args.batch_size)))
        if args.samples_fid:
            for i in range(samples.shape[0]):
                samples_path = os.path.join(dir_samples_fid, f"{catg}_{idx}-{i}.png") # check samples in [-1,1]
                save_image(samples[i],samples_path)

        samples = F.adjust_brightness(samples, 3) if samples.mean()<0.2 and samples.shape[0]==1 else samples
        save_image(samples,img_path,nrow=int(math.sqrt(args.batch_size))) if args.save else print()
        n = n+1
        
        ssim_avg, psnr_avg = ssim/n, psnr/n # correct if len is referred to num batches
        print("metrics: ",ssim_avg, psnr_avg)
        if args.metrics:
            with open(os.path.join(dir, "metrics.txt"), "w") as f:
                f.write(f"ssim: {ssim_avg}")
                f.write(f"psnr: {psnr_avg}")
                f.write(f"length: {n}")

        if args.n_iter is not None:
            if j > args.n_iter: break

def test(idx=0):
    to_pil = torchvision.transforms.ToPILImage()
    dl, dl2 = create_oscd_dataloaders(batch_size=1, return_dataset=True, test=True)
    print(len(dl), len(dl2))
    img, img2, lbl = dl[idx]["image"],dl2[idx]["image"], 1-dl[idx]["segmentation"]
    image = img*((lbl+0.7).clip(0,1))
    image = to_pil(image)
    image.show()
    breakpoint()



def test2(image_size):
    dl = create_inria_dataloaders(batch_size = 1, size=64, patch_overlap=0 ,return_dataset=True, test=True)
    for i, data in enumerate(dl):
        patch_idx = i
        #tile = self.to_tensor(Image.open(os.path.join(self.root, self.names[n])))[0][None] # be careful to grayscale opening
        orig_size = (dl.h, dl.w)
        np_I, np_J = int((orig_size[0]-image_size)/step+1), int((orig_size[1]-image_size)/step+1)
        np_I, np_J = max(np_I,1), max(np_J,1)
        idx_i, idx_j=int(patch_idx/np_J)*step, int(patch_idx%np_J)*step
        idx_i, idx_j = max(min(idx_i, orig_size[0]-image_size-1),0), max(min(idx_j, orig_size[1]-image_size-1),0)
        print(idx_i, idx_j)
        img = tile[:,idx_i:idx_i+self.size, idx_j:idx_j+self.size]

            

if __name__=="__main__":
    args=parse_args()
    main(args)