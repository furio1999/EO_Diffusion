import torch.nn as nn
import torch
import math
#from backbones.unet_mnist import Unet
from backbones.unet_openai import *
from tqdm import tqdm
from torchvision.utils import save_image
import torchvision.transforms.functional as F
import pytorch_lightning as pl
from utils import ExponentialMovingAverage
from train_utils import KeyframeLR
#def __init__(self,image_size,in_channels,out_channels,cond_channels=0, time_embedding_dim=256,timesteps=1000,
#    cond_type = None, base_dim=32, dim_mults= [2,4], attention_resolutions = tuple([]), num_classes=None):

class EO_Diffusion(pl.LightningModule):
    def __init__(self, model, image_size,in_channels, time_embedding_dim=256,timesteps=1000,
    cond_type = None, cond_stage_key = None, device = "cpu", lr=1e-03, model_ema=True, epochs = 100, steps_epochs = 100):
        super().__init__()
        print("loading model...")
        self.timesteps=timesteps
        self.in_channels=in_channels
        self.image_size=image_size
        self.cond_type, self.cond_key = cond_type, cond_stage_key
        self.device = device
        self.lr = lr
        self.model_ema = model_ema

        self.epochs, self.steps_epochs = epochs, steps_epochs
        self.loss_fn=nn.MSELoss(reduction='mean')

        betas=self._cosine_variance_schedule(timesteps)

        alphas=1.-betas
        alphas_cumprod=torch.cumprod(alphas,dim=-1)

        self.register_buffer("betas",betas)
        self.register_buffer("alphas",alphas)
        self.register_buffer("alphas_cumprod",alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod",torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod",torch.sqrt(1.-alphas_cumprod))

        #self.model=Unet(timesteps,time_embedding_dim,in_channels,in_channels,base_dim,dim_mults) # why out_channels=2??
        self.model = DiffusionWrapper(model, conditioning_key=cond_type)
        print("Loaded!!")

    def training_step(self, batch, batch_idx):
        x, x_c = batch[self.first_stage_key], batch[self.cond_stage_key]
        noise=torch.randn_like(x).to(self.device)
        x=x.to(self.device)
        pred=self.model(x,noise, cond=x_c)
        loss=self.loss_fn(pred,noise)
        return loss

    def configure_optimizers(self):
        optimizer=torch.optim.AdamW(self.parameters(), lr=self.lr)
        max_steps, posmax, end_lr = self.steps_epochs*self.epochs, 10*self.steps_epochs, 1e-06
        scheduler = KeyframeLR(optimizer=optimizer, units="steps",
        frames=[
            {"position": 0, "lr": self.lr/100},
            {"transition": "cos"},
            {"position": posmax, "lr": self.lr},
            {"transition": lambda last_lr, sf, ef, pos, *_: self.lr * math.exp(-3*(pos-posmax)/(max_steps-posmax))},
        ],
        end=max_steps,
    )
        return ({"optimizer":optimizer, "lr_scheduler":scheduler})
    
    def ema_module(self, epochs, batch_size, model_ema_steps, model_ema_decay, device):
        adjust = 1* batch_size * model_ema_steps / epochs
        alpha = 1.0 - model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = ExponentialMovingAverage(self, device=device, decay=1.0 - alpha)
        return model_ema

    def forward(self,x,noise, cond=None, y=None, context=None):
        # x:NCHW
        t=torch.randint(0,self.timesteps,(x.shape[0],)).to(x.device)
        x_t=self._forward_diffusion(x,t,noise)
        pred_noise=self.model(x_t,t, cond=cond, y=y)

        return pred_noise

    @torch.no_grad()
    def sampling(self,n_samples, clipped_reverse_diffusion=True,device="cpu", cond=None, y=None, idx=0, save=False):
        x_t=torch.randn((n_samples,self.in_channels,self.image_size,self.image_size)).to(device)
        #concat = x_t.clone()
        if cond is not None and self.cond_type == "sum": 
            gt, mask = cond[:n_samples,:3], cond[:n_samples,3][:,None]
            cond=None
        #print(x_t.shape, cond.shape, self.in_channels)
        for i in tqdm(range(self.timesteps-1,-1,-1),desc="Sampling"):
            noise=torch.randn_like(x_t).to(device)
            t=torch.tensor([i for _ in range(n_samples)]).to(device)

            if self.cond_type == "sum":
                gt_noised = self._forward_diffusion(gt,t,noise) # check data range-->try to redo it in (-1,1)
                x_t = mask*gt_noised + (1-mask)*x_t #(gt_noised)*2.-1.

            if i%25==0 and i<=200 or i%100==0 and i<=self.timesteps and save:
                save_image((x_t+1.)/2., f"results/prova/s{idx}_{i}_pred.png", nrow = int(math.sqrt(n_samples))) # save such that you have 1 row for each idx, play with color of grid
                if self.cond_type == "sum":
                    save_image(mask*gt_noised, f"results/prova/s{i}_masked.png", nrow = int(math.sqrt(n_samples))),
                    save_image(gt_noised, f"results/prova/s{i}_gt.png") #save_image((1-mask)*x_t, f"results/prova/s{i}_masked_pred.png")

            if clipped_reverse_diffusion:
                x_t=self._reverse_diffusion_with_clip(x_t,t,noise,cond=cond, y=y)
            else:
                x_t=self._reverse_diffusion(x_t,t,noise,cond=cond, y=y)

        #x_t=(x_t+1.)/2. if x_t.min() < 0 else x_t #[-1,1] to [0,1]

        return x_t
    
    def forward_only(self, img, device="cpu"):
        for i in tqdm(range(self.timesteps-1,-1,-1),desc="Noising"):
            noise=torch.randn_like(img).to(device)
            t=torch.tensor([i for _ in range(img.shape[0])]).to(device)
            img_noised = self._forward_diffusion(img,t,noise)
            if i%25==0 and i<=200 or i%100==0 and i<=self.timesteps:
               save_image(img_noised, f"results/prova/img{i}.png")
        breakpoint()

    
    def _cosine_variance_schedule(self,timesteps,epsilon= 0.008):
        steps=torch.linspace(0,timesteps,steps=timesteps+1,dtype=torch.float32)
        f_t=torch.cos(((steps/timesteps+epsilon)/(1.0+epsilon))*math.pi*0.5)**2
        betas=torch.clip(1.0-f_t[1:]/f_t[:timesteps],0.0,0.999)

        return betas

    def _forward_diffusion(self,x_0,t,noise):
        assert x_0.shape==noise.shape
        #q(x_{t}|x_{t-1})
        return self.sqrt_alphas_cumprod.gather(-1,t).reshape(x_0.shape[0],1,1,1)*x_0+ \
                self.sqrt_one_minus_alphas_cumprod.gather(-1,t).reshape(x_0.shape[0],1,1,1)*noise


    @torch.no_grad()
    def _reverse_diffusion(self,x_t,t,noise,cond=None, y=None):
        '''
        p(x_{t-1}|x_{t})-> mean,std

        pred_noise-> pred_mean and pred_std
        '''
        pred=self.model(x_t,t,cond=cond, y=y)

        alpha_t=self.alphas.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        alpha_t_cumprod=self.alphas_cumprod.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        beta_t=self.betas.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        sqrt_one_minus_alpha_cumprod_t=self.sqrt_one_minus_alphas_cumprod.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        mean=(1./torch.sqrt(alpha_t))*(x_t-((1.0-alpha_t)/sqrt_one_minus_alpha_cumprod_t)*pred)

        if t.min()>0:
            alpha_t_cumprod_prev=self.alphas_cumprod.gather(-1,t-1).reshape(x_t.shape[0],1,1,1)
            std=torch.sqrt(beta_t*(1.-alpha_t_cumprod_prev)/(1.-alpha_t_cumprod))
        else:
            std=0.0

        return mean+std*noise 


    @torch.no_grad()
    def _reverse_diffusion_with_clip(self,x_t,t,noise,cond=None, y=None): 
        '''
        p(x_{0}|x_{t}),q(x_{t-1}|x_{0},x_{t})->mean,std

        pred_noise -> pred_x_0 (clip to [-1.0,1.0]) -> pred_mean and pred_std
        '''
        pred=self.model(x_t,t,cond=cond, y=y)
        alpha_t=self.alphas.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        alpha_t_cumprod=self.alphas_cumprod.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        beta_t=self.betas.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        
        x_0_pred=torch.sqrt(1. / alpha_t_cumprod)*x_t-torch.sqrt(1. / alpha_t_cumprod - 1.)*pred
        x_0_pred.clamp_(-1., 1.)

        if t.min()>0:
            alpha_t_cumprod_prev=self.alphas_cumprod.gather(-1,t-1).reshape(x_t.shape[0],1,1,1)
            mean= (beta_t * torch.sqrt(alpha_t_cumprod_prev) / (1. - alpha_t_cumprod))*x_0_pred +\
                 ((1. - alpha_t_cumprod_prev) * torch.sqrt(alpha_t) / (1. - alpha_t_cumprod))*x_t

            std=torch.sqrt(beta_t*(1.-alpha_t_cumprod_prev)/(1.-alpha_t_cumprod))
        else:
            mean=(beta_t / (1. - alpha_t_cumprod))*x_0_pred #alpha_t_cumprod_prev=1 since 0!=1
            std=0.0

        return mean+std*noise 
    
class DiffusionWrapper(pl.LightningModule):
    def __init__(self, nn_model, conditioning_key):
        super().__init__()
        self.diffusion_model = nn_model
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm']

    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None):
        if self.conditioning_key is None:
            out = self.diffusion_model(x, t)
        elif self.conditioning_key == 'concat':
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t)
        elif self.conditioning_key == 'crossattn':
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(x, t, context=cc)
        elif self.conditioning_key == 'hybrid':
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc)
        elif self.conditioning_key == 'class':
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)
        else:
            raise NotImplementedError()

        return out