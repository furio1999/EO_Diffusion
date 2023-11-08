# EO-Diffusion
<p align="left">
  <img   
  width="30%"
  height="30%" 
  src="assets/Sapienza_Roma_Logo.png">
</p>
<p align="right">
  <img   
  width="30%"
  height="30%" 
  src="assets/ESA_logo_2020_Deep.png">
</p>
This work is the result of a collaboration between [European Space Agency](https://www.esa.int/), [Φ-Lab](https://philab.esa.int/about/) and [Sapienza University of Rome](https://www.uniroma1.it/it/), [Alcor Lab](https://alcorlab.diag.uniroma1.it/) for my master thesis ([manuscript pdf](assets/Fulvio_Master_Thesis_Definitive.pdf))
Code for our [paper](assets/2023_BIDS_Diffusion_Models_for_EO.pdf) "Diffusion Models for Earth Observation Use-cases: from cloud removal to urban change detection", accepted as oral at Big Data From Space 2023 (BIDS 2023)
<p align="center">
  <img   
  width="30%"
  height="30%" 
  src="assets/diff1.gif">
</p>

## Demo
download the following checkpoint https://drive.google.com/file/d/1u415nF2ZzsNnJ8w-BdzT8FC8123R4LcJ/view?usp=drive_link in "results" folder and rename it "clouds_best.pt". </br>
You can find a demo at the following notebook [EO_Diffusion.ipynb](EO_Diffusion.ipynb)
## Use-Cases
### Cloud Removal
<p align="center">
  <img   
  width="30%"
  height="30%" 
  src="assets/slides_cr.png">
</p>

### Synthetic OSCD
<p align="center">
  <img   
  width="30%"
  height="30%" 
  src="assets/slides_oscd.png">
</p>

### Urban Replanning
<p align="center">
  <img   
  width="30%"
  height="30%" 
  src="assets/inpaint.png">
</p>

## Installation
Conda environment: 
- Conda 23.1.0
- CUDA toolkit: 11.7.1
- Pytorch: 11.3.0
- Torchvision + torchaudio: 0.14.0 + 0.13.0
- Tested on an NVIDIA RTX 4000 (49 GB)

GPU utilities installation:
I don't recomment using the exported eo_diffusion.yml file. It's better to install it directly from pytorch website with the required versions, as shown in the command below:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda create -n env_name
conda activate env_name
conda install pip
pip install -r requirements.txt
```

## Training

```bash
python train.py
```

Specify batch size, diffusion steps training hyperparameters from command line (you have default values otherwise)
```bash
python train.py --batch_size 4 --timesteps 1000 --lr 1e-05 --epochs 200
```

### Conditional Training
Based on the same concept expressed in RePaint https://arxiv.org/abs/2201.09865
```bash
python train.py --cond_type "sum"
```

Save your results
```bash
python train.py --dir path/to/your/dir --ckpt ckpt_name
```

## Testing
--save option if you want to store images
```bash
python inference.py --ckpt path/to/your/ckpt --outdir path/to/your/folder_samples --save
```

## Customization
Below you find the two relevant lines to modify concerning U-Net architecture and data loaders, in train.py and inference.py
```bash
base_dim, dim_mults, attention_resolutions,num_res_blocks, num_heads=128,[1,2,3,4],[4,8],2,8
train_dataloader,test_dataloader=create_cloud_dataloaders(batch_size=args.batch_size, num_workers=4, size=image_size,
                ratio=0.5, length=-1, num_patches=2000, percents=[99,0,70])
```
In data.py you find all the available dataloaders with the title create_{dataset_name}_dataloader. </br>
In data_load.py you find all the Dataset classes for the available datasets. 

Concerning U-Net and Diffusion, you can modify the parameters at this two lines:
```bash
unet = UNetModel(image_size, in_channels=in_channels+cond_channels, model_channels=base_dim, out_channels=out_channels, channel_mult=dim_mults, 
                attention_resolutions=attention_resolutions,num_res_blocks=num_res_blocks, num_heads=num_heads, num_classes=num_classes)
model=EODiffusion(unet,
            timesteps=args.timesteps,
            image_size=image_size,
            in_channels=in_channels
            ).to(device)
```

## References

Lilian Weng blog on Diffusion Models: https://lilianweng.github.io/posts/2021-07-11-diffusion-models/

The Denoising Diffusion Probabilistic Models paper: https://arxiv.org/pdf/2006.11239.pdf 

RePaint paper: https://arxiv.org/abs/2201.09865

