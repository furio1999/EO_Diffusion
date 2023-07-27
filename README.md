# EO-Diffusion
![60 epochs training from scratch](assets/diff1.gif "60 epochs training from scratch")

A simple codebase for my master thesis work about diffusion models for EO

## Installation
Conda environment: 
- Conda 23.1.0
- NVIDIA rtx 4000 (49 GB)
- CUDA toolkit: 11.7.1
- Pytorch: 11.3.0
- Torchvision + torchaudio: 0.14.0 + 0.13.0
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

## References

Lilian Weng blog on Diffusion Models: https://lilianweng.github.io/posts/2021-07-11-diffusion-models/

The Denoising Diffusion Probabilistic Models paper: https://arxiv.org/pdf/2006.11239.pdf 

RePaint paper: https://arxiv.org/abs/2201.09865

