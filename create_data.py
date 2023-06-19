from data_load import make_patches, InriaDataset, select_class, InriaData
import os,glob
import numpy as np
from torchvision.transforms import ToPILImage
from torch.utils.data import random_split
from torchvision.utils import save_image, make_grid
from torch import Generator
from einops import rearrange
from data import *
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from PIL import Image
"""
path="../EO-Diffusion/data/AerialImageDataset"
vocab_images, vocab_gt = {}, {}
textfile, textfile2 = os.path.join(path, "train/images.txt"),os.path.join(path, "train/gt.txt")
images = sorted(glob.glob(os.path.join(path, "train/images", "*tif")))
masks = sorted(glob.glob(os.path.join(path, "train/gt", "*tif")))
print("files selected")
classes = ['vienna', 'austin', 'tyrol', 'chicago', 'kitsap']"""
"""
for cat in classes:
    images, masks = select_class(images, masks, cat=cat)
    print("saving images from ", cat)
    for image, mask in zip(images,masks):
        print("save run...")
        make_patches([image], in_ch=3, num_patches = 10000, size=64, ratio=0,outpath = "../data/AerialImageDataset64/train/images", cat=cat), # cat not necessary
        make_patches([mask], num_patches=10000, in_ch=1, size=64, ratio=0, outpath = "../data/AerialImageDataset64/train/gt",cat=cat) #images, masks
"""
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
## TODO txt file with orig_name - class, txt file with name_patch - class. Could have been better to divide it by folders??"""
def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()

def test_mask():
        dl, _ = create_inria_dataloaders(batch_size=1, num_workers=4)
        outdir = "results/prova"
        img_path, mask_path = os.path.join(outdir, "image.png"),  os.path.join(outdir, "mask.png")
        os.makedirs(outdir, exist_ok=True)
        for data in dl:
                image, mask = data["image"], data["segmentation"]
                inverse_mask = 1-mask
                print("loaded")
                if mask.mean()>=0.8 or mask.mean()<=0.2: 
                        print("aargh")
                        continue
                print("inpaint")
                inpainted = image*inverse_mask
                save_image(inpainted, img_path, nrow=1)
                save_image(image, os.path.join(outdir, "imageorig.png"), nrow=1)
                save_image(mask, mask_path, nrow=1)
                save_image(inverse_mask, os.path.join(outdir, "maskinverse.png"), nrow=1)
                breakpoint()

def test_cloud(idx=0,ds=None):
        img,mask = ds[idx]["image"],ds[idx]["segmentation"]
        img_orig, mask_orig = ds[idx]["orig_image"],ds[idx]["orig_segm"]
        img_orig = img_orig*(1-mask_orig)
        img_orig = img_orig/img_orig.max()
        if img.min() < 0: 
              img,mask = (img+1.)/2., (mask+1.)/2.
        save_image(img,"results/prova/img.png"),save_image(mask,"results/prova/mask.png")
        save_image(img_orig,"results/prova/imgorig.png"),save_image(mask_orig,"results/prova/maskorig.png")
        breakpoint()
        #show(img_orig),

def test_save():
      arr1 = torch.zeros((5,3,64,64))
      arr2 = torch.ones((5,3,64,64))
      arr3 = torch.randn((5,3,64,64))
      arr = torch.cat([arr1,arr2,arr3])
      save_image(arr,"results/prova/concat.png", nrow=5)

class tester():   
    def __init__(self):
        self.img = None 
    def test_hist(self,img):
        if img.min()<0: img = (img+1.)/2.
        img = img*255
        if self.img is None: self.img = torch.clone(img)[None]
        self.img = torch.mean(torch.cat([self.img, img[None]]), dim=0, keepdim=True)
        bins = torch.linspace(0, 256, 257)
        hist = [torch.histogram(c, bins=bins) for c in img] # self.img[0]
        plt.plot(hist[0].bin_edges[:-1], hist[0].hist, color="r")
        plt.plot(hist[1].bin_edges[:-1], hist[1].hist, color="g")
        plt.plot(hist[2].bin_edges[:-1], hist[2].hist, color="b")

def test_oscd():
        preprocess=transforms.Compose([ transforms.RandomHorizontalFlip(), transforms.RandomHorizontalFlip(), 
                                     transforms.RandomAdjustSharpness(p=0.3, sharpness_factor=0.3),transforms.RandomSolarize(threshold=0.5, p=0.1), transforms.RandomAdjustSharpness(p=0.3, sharpness_factor=1.5),
                                     #transforms.Normalize(mean=(0.5), std=(0.5))
                                   ])
        norm = transforms.Normalize(mean=(0.5), std=(0.5))
        ds = OSCD(transform=preprocess)
        print(len(ds))
        for i in range(1000):
          print(i)
          img, cond = ds[i]["image"], 1-ds[i]["segmentation"]
          breakpoint()
          if cond.min() < 0: img, cond = (img+1.)/2., (cond+1.)/2.
          show(img), show(img*cond)   

if __name__ == "__main__":
      test_oscd()
     
