from data_load import make_patches, InriaDataset, select_class, InriaData
import os,glob
import numpy as np
from torchvision.transforms import ToPILImage
from torch.utils.data import random_split
from torch import Generator
from einops import rearrange
path="../EO-Diffusion/data/AerialImageDataset"
vocab_images, vocab_gt = {}, {}
textfile, textfile2 = os.path.join(path, "train/images.txt"),os.path.join(path, "train/gt.txt")
images = sorted(glob.glob(os.path.join(path, "train/images", "*tif")))
masks = sorted(glob.glob(os.path.join(path, "train/gt", "*tif")))
print("files selected")
classes = ['vienna', 'austin', 'tyrol', 'chicago', 'kitsap']

for cat in classes:
    images, masks = select_class(images, masks, cat=cat)
    print("saving images from ", cat)
    for image, mask in zip(images,masks):
        print("save run...")
        make_patches([image], in_ch=3, num_patches = 10000, size=64, ratio=0,outpath = "../data/AerialImageDataset64/train/images", cat=cat), # cat not necessary
        make_patches([mask], num_patches=10000, in_ch=1, size=64, ratio=0, outpath = "../data/AerialImageDataset64/train/gt",cat=cat) #images, masks
"""
fim, fgt = open(textfile, "a"),open(textfile2, "a")
for cat in classes: 
    print("cat ", cat)
    ims, ms =  select_class(images, masks, cat=cat)
    vocab_images.update({os.path.split(key)[-1]:val for key,val in zip(ims,len(ims)*[cat])})
    vocab_gt.update({os.path.split(key)[-1]:val for key,val in zip(ms,len(ms)*[cat])})
    breakpoint()
for key, value in vocab_images.items(): 
        fim.write('%s %s\n' % (key, value))
for key, value in vocab_gt.items(): 
        fgt.write('%s %s\n' % (key, value))
"""
val_split = 0.15
dataset = InriaDataset(path = "../EO-Diffusion/data/AerialImageDataset", length=2, cond="class", num_patches=1000, size=64)
train_ds, test_ds = random_split(dataset, [1-val_split, val_split], generator=Generator().manual_seed(1200)) # custom split with predefined functs?
breakpoint()
to_pil = ToPILImage()
data = dataset[6]
img, target = data["image"], data["class"]
img = to_pil(rearrange(img, 'h w c -> c h w'),)
breakpoint()
## TODO txt file with orig_name - class, txt file with name_patch - class. Could have been better to divide it by folders??
        
