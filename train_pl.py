import argparse
from data import *
from pytorch_lightning.loggers import WandbLogger
from tqdm import tqdm

from pytorch_lightning import seed_everything
from omegaconf import OmegaConf

from backbones.unet_openai import UnetModel
from diffusion.model_pl import EO_Diffusion
from train_utils import CustomModelCheckpoint, EMACallback

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
  config_file = "models/ldm/semantic_synthesis64/unet64.yaml" # TODO, write yml file
  config = OmegaConf.load(config_file)
  first_stage_key, cond_stage_key = config.model_config.first_stage_key, config.model_config.cond_stage_key
  unet_config = config.unet_config
  base_lr, max_epochs, ckpt_path = config.train_config.lr, config.train.max_epochs, config.train_config.ckpt,
  image_size, batch_size, num_workers = config.data_config.image_size, config.data_config.batch_size, config.data_config.num_workers
  num_res_blocks, in_channels, cond_channels, out_channels, dim_mults, attention_resolutions, num_heads, =  unet_config.num_res_block, unet_config.channels, unet_config.cond_channels,unet_config.out_channels,
  unet_config.ch_mult, unet_config.att_mult, unet_config.num_heads
  ngpu = 1 #torch.cuda.device_count()
  device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
  # seed everything

  unet = UNetModel(image_size, in_channels=in_channels+cond_channels, model_channels=in_channels, out_channels=out_channels, channel_mult=dim_mults, 
                        attention_resolutions=attention_resolutions,num_res_blocks=num_res_blocks, num_heads=num_heads, num_classes=args.num_classes)
  model=EO_Diffusion(unet,
            timesteps=args.timesteps,
            image_size=image_size,
            in_channels=in_channels
            ).to(device)

  logger = WandbLogger(project="LDM-EO", log_model=True)
  #logger.experiment.name = name_run
  logger.watch(model)
  callback = CustomModelCheckpoint(dirpath=ckpt_path, # where the ckpt will be saved
                                      filename='sample-{epoch}-{step}', # the name of the best ckpt
                                      save_top_k=3,
                                      verbose=True,
                                      monitor="train/loss_simple", # multiple monitor?
                                      mode="min" # validation loos need to be min
                                      )
  callback_ema = EMACallback()
  callbacks = [callback]

  #dataset = MyDataset(images, masks, shape=(in_size, in_size), rgb_mask=False)
  train_dl, val_dl = create_inria_dataloaders(batch_size = batch_size, 
  num_workers=4*ngpu, num_patches=2000, patch_overlap=0, length=0)


if __name__ == "__main__":
    main()