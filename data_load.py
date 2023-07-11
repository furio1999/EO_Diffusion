import argparse, os, sys, datetime, glob, importlib, csv
import numpy as np
import time
import torch
import torchvision
import torchvision.transforms.functional as F
import pytorch_lightning as pl
from tqdm import tqdm
from einops import rearrange
from utils import *

from packaging import version
from omegaconf import OmegaConf 
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from functools import partial
from PIL import Image
from patchify import patchify, unpatchify
import kornia
from kornia.utils import image_to_tensor
import kornia.augmentation as KA
import pdb, random
import pandas as pd

# inria: 2 classes? multiclasses in an image?

# TODO delete rgb_mask, insert in_ch
# consider do everything with torch tensors, in cuda. Measure difference between cpu and gpu
# maybe separate functions for images and masks
def preprocess_data(img, msk, device, mode="RGB", mask_ch = 1, left=0, top=0, size=64, threshold=0.5, transform=torchvision.transforms.ToPILImage()):
    trh = threshold # problems
    right, bottom = left+size, top+size # bottom disturbed by multiple assignments on same line
    shape = (left, top, right, bottom) # shape(None) = (0)

    if type(img) == str:
      image = np.array(Image.open(img).convert(mode))
    else:
      image = transform(img).convert(mode)
      image = np.array(image.crop(shape))

    image = image.astype(np.float32)/255.0
    if len(image.shape)==2: 
      image = image[:,:,None]
    image = image.transpose(2,0,1)
    image = torch.from_numpy(image)


    if type(msk) == str:
      mask_image = Image.open(msk)
    elif msk is None:
        pass
    else:
      mask_image = transform(msk).convert("L")
      mask_image = np.array(mask_image.crop(shape))

    if mask_ch==3:
      mask = np.array(mask_image).astype(np.float32)/255.0 # careful, this command does notreturn error with mask_image=None
      if mask.shape[-1] == 3:
        mask = mask.transpose(2,0,1)
    elif mask_ch==1:
      mask = np.array(mask_image) # what is this L?? related to L in apply_image??
      mask = mask.astype(np.float32)/255.0
      mask = mask[None]
      mask[mask < trh] = 0
      mask[mask >= trh] = 1
      #masked_image = (1*mask) * image + mask*image # mask is only 2-class; how to do it with multiclass? Multiclass inpaint?
    else:
        mask = None
      

    if mask is not None:
        mask = torch.from_numpy(mask)
        batch = {"image": image, "mask": mask}
    else:
        batch = {"image": image}

    for k in batch:
        batch[k] = batch[k].to(device=device)
        batch[k] = batch[k]*2.0-1.0 # be careful
    return batch

def visualize_data(outpath):
    pass


class MyDataset(Dataset):
    def __init__(self, images, masks, size=64, device="cpu", nlabels = 2, rgb_mask=None):
        li, lm = len(images), len(masks)
        assert li==lm, print(f"different lengths {li} {lm}")
        self.length = len(images)
        self.ims, self.masks = images, masks
        self.nlabels = nlabels
        self.size = size
        self.device = device
        self.rgb_mask = rgb_mask
    def __len__(self):
        return self.length
    def __getitem__(self, i):
        #example = dict((k, self.labels[k][i]) for k in self.labels)
        example = dict()
        image, mask = self.ims[i], self.masks[i]
        batch = preprocess_data(image, mask, self.device, size = self.size, rgb_mask = self.rgb_mask) # TODO, very slow

        image,mask = batch["image"][0], batch["mask"][0]
        image, mask = rearrange(image, 'c h w -> h w c'), rearrange(mask, 'c h w -> h w c')


        processed = {"image": image, "mask": mask}
        example["image"] = processed["image"]
        segmentation = processed["mask"]
        example["segmentation"] = segmentation
        return example

class DerivedDataset(MyDataset):
    def __getitem__(self,i):
        batch = super().__getitem__(i)
        #breakpoint()
        batch["segmentation2"] = batch.pop("segmentation")
        #breakpoint()
        return batch

def preprocess():
    pass

def augment():
    pass

def select_class(image_files, mask_files, cat=None):
  images,masks = [],[]
  if cat is None:
    return image_files, mask_files
  if type(cat) is not list:
    categories = [cat]
  for cat in categories:
   images = images + [image.replace("\\","/") for image in image_files if os.path.split(image)[-1][:3]==cat[:3]]
   masks = masks + [mask.replace("\\", "/") for mask in mask_files if os.path.split(mask)[-1][:3]==cat[:3]]
  return sorted(images), sorted(masks)

def extract_patches(images, masks, h,w, n_patches=1, size=64, rgb_mask=None, ratio=0.4, device="cpu", threshold=0.5):
        ims, ms = [], []
        step = int(ratio*size)
        for i, (im,mask) in enumerate(zip(images,masks)):
          n = 0
          print(f"extracting patches from image {i} in dataset")
          # with this loop we discard the right and bottom extremum of the image
          for left,top in zip(range(0,w,step),range(0,h,step)):
            if left+size >=w or top+size>=h or n>=n_patches:
                break
            batch = preprocess_data(im, mask, device, left=left, top=top, size=size, rgb_mask = rgb_mask, threshold = threshold)
            ims.append(batch["image"]), ms.append(batch["mask"])
            imfile, maskfile = os.path.splitext(im)[0], os.path.splitext(mask)[0]
            imfile, maskfile = imfile + f"_{n}.tif", maskfile + f"{n}.tif"
            n+=1
          print(f"{n} patches extacted")

        return ims, ms

### IDEA: insert patchify into tranforms or insert transforms in make_patches instead of patchify
def make_patches(imfiles, device="cpu", in_ch=3, outpath=None, num_patches=1, size=64, ratio=0.5, step=None, bias = 0, cat=None, max_size = 5000):
  """
  Input: strings pointing to files; Output: Torch.Tensor of shape (N_patches X C X H X W)
  """
  if step is None:
    step = int((1-ratio)*size)
  transform = torchvision.transforms.ToPILImage()
  to_tensor = torchvision.transforms.ToTensor()
  length = len(imfiles)
  num_patches = min(num_patches, (int(max_size/step))**2)
  im_patches = torch.zeros((num_patches*(length), size, size, in_ch))#.to(device) # originally numpy
  print(f"extracting patches from {in_ch} channels images")
  vocab, sump = {}, 0
  for i, imfile in enumerate(imfiles):
    print(f"image {i+1}")
    if in_ch == 3:
      image = np.array(Image.open(imfile).convert("RGB"))
    else:
      image = np.array(Image.open(imfile).convert("L"))[:,:,None] # original: without .convert("L")
    image_patches = patchify(image, (size, size, image.shape[-1]), step) # torch function to patchify everything? Unfold?
    image_patches = torch.from_numpy(image_patches)#.to(device) # maybe translate all to numpy or all to torch (.astype substitute)
    image_patches = torch.flatten(image_patches, start_dim=0, end_dim=2)
    dim = image_patches.shape[0]
    n_patches, jump = min(num_patches,dim), dim//num_patches
    print(f"selected {n_patches} patches out of {image_patches.shape[0]} total patches")
    image_patches = image_patches[:n_patches*jump:jump] if jump>0 else image_patches[:n_patches]
    im_patches[n_patches*i:n_patches*(i+1)]=image_patches
    sump += n_patches

  im_patches = im_patches#.to(device, dtype=torch.uint8) # maybe rearrange dims, careful for next rearrange in preprocess_data
  im_patches = rearrange(im_patches, 'b h w c -> b c h w')

  if outpath is None:
    return im_patches[:sump]

  os.makedirs(outpath, exist_ok = True)
  textfile = os.path.join(outpath, "images.txt")
  vocab = open(textfile) if textfile in os.listdir(outpath) else {}
  assert im_patches.shape[0]%n_patches == 0
  bias = min(bias,len(im_patches)-1)
  for idx in range(im_patches[bias:].shape[0]):
    idx_file, n_patch = idx//n_patches, idx%n_patches
    imfile = imfiles[idx_file]
    img = transform(im_patches[idx]*255) # careful, im_patches are numpy, not torch.tensor
    #breakpoint()
    outfile = os.path.join(outpath, os.path.splitext(os.path.split(imfile)[-1])[0] + f"_{n_patch+bias}.tif").replace("\\","/")
    vocab[imfile] = cat
    img.save(outfile)


    

# TODO: load images and masks names directly into dataset  
class InriaDataset(Dataset):
    # would be nice having init working once per all
    def __init__(self, path="data/AerialImageDataset", val_split=0.15, length=0, num_patches=1, patch_overlap=0.5, transforms =None,
    size=64, device="cpu", nlabels = 2, img_ch=3, mask_ch=3, category = None, threshold=0.5, compact=True, cond=None, uncond="image", 
    use_val=None, ch_last=True, load_from_disk=False, use_int=False):
        images = sorted(glob.glob(os.path.join(path, "train/images", "*tif")))
        masks = sorted(glob.glob(os.path.join(path, "train/gt", "*tif")))
        li, lm = len(images), len(masks)
        self.h,self.w = 5000,5000
        assert li==lm, print(f"different lengths {li} {lm}")
        self.ch_last = ch_last
        self.img_ch, self.mask_ch, self.patch_overlap = img_ch, mask_ch, patch_overlap
        self.compact, self.cond, self.uncond = compact, cond, uncond
        self.transforms = transforms
        self.nlabels, self.num_patches = nlabels, num_patches
        self.size, self.patch_overlap = size, patch_overlap
        self.step = min(int(self.patch_overlap*self.size), int(self.h/self.num_patches))
        assert img_ch == 1 or img_ch==3, print("image channels must be 1 or 3")
        self.mode = "RGB" if img_ch==3 else "L"
        self.category = category
        self.device, self.threshold = device, threshold
        self.use_int = use_int
        images, masks = select_class(images, masks, cat=category)

        if length > 0 and length < len(images):
          jump = len(images)//length
          images, masks = images[:length*jump:jump], masks[:length*jump:jump]
        #temp = list(zip(images,masks))
        #random.shuffle(temp)
        #images,masks = zip(*temp)
        images, masks = list(images), list(masks)
        #print(images,masks)

        self.vocab = {}
        textfile = os.path.join(path, "train/images.txt")
        if os.path.exists(textfile):
          with open(textfile) as f: 
              for line in f.readlines(): 
                  key,val = line.split()
                  self.vocab[key] = val
        self.class_labels = {'austin':0,  'chicago':1, 'kitsap':2, 'tyrol':3, 'vienna':4} # otherwise for 1...5 gives CUDA INTERNAL ERROR, but it's just class numbering mismatch
        
        self.imfiles, self.maskfiles = images, masks
        self.ims, self.masks = self.imfiles, self.maskfiles
        if not load_from_disk:
          self.ims = make_patches(images, in_ch=img_ch, num_patches = self.num_patches, size=self.size, ratio=patch_overlap)
          self.masks = make_patches(masks, num_patches=self.num_patches, in_ch=mask_ch, size=self.size, ratio=patch_overlap) #images, masks
        l = len(self.ims) if type(self.ims) is list else self.ims.shape[0]
        assert l % len(self.imfiles) == 0, print(f"{l} total patches for {len(self.imfiles)} files")
        self.n_patches = l//len(self.imfiles)

        self.val_length = int(val_split*l) # subtle problem: val_length updated with same length accessed by TRAIN on gpu 0 for gpu 1!!!
        """
        if use_val:
          self.ims, self.masks = self.ims[:self.val_length], self.masks[:self.val_length] # :x: !!!
          self.imfiles, self.maskfiles = self.imfiles[:self.val_length], self.maskfiles[:self.val_length]
        elif not use_val:
          self.ims, self.masks = self.ims[self.val_length:], self.masks[self.val_length:]
          self.imfiles, self.maskfiles = self.imfiles[self.val_length:], self.maskfiles[self.val_length:]
        else:
          pass """
        
        self.length = len(self.ims) if type(self.ims) is list else self.ims.shape[0]  

        self.pil_to_tensor = torchvision.transforms.PILToTensor()
        self.process = KA.RandomGaussianBlur((3,3),(0.1,2.0))


    def __len__(self):
        return self.length

    # pytorch random split train val handle patch-label idx
    def __getitem__(self, n):
        #example = dict((k, self.labels[k][i]) for k in self.labels)
        example = dict()
        imfile, maskfile = self.ims[n], self.masks[n] #mask gets corrupted during training, but in debuggign is fine!!
        # make size/newsize patches in getitem
        class_label = self.class_labels[self.vocab[os.path.split(self.imfiles[n//self.n_patches])[-1]]] if self.vocab!={} else None # make sure imfiles and patches order correspond
        if type(imfile) == str:
          image, mask = self.pil_to_tensor(Image.open(imfile))/255, self.pil_to_tensor(Image.open(maskfile).convert("L"))/255
        else:
          image, mask = imfile/255, maskfile/255
        example["image_orig"], example["mask_orig"] = image, mask
        if self.transforms is not None:
          out = self.transforms(torch.cat((image,mask),0))
          image,mask = out[:3], out[3][None]
          #image,mask = image[0],mask[0]
        if image.shape[-1]>image.shape[0] and self.ch_last:
          image, mask = rearrange(image, 'c h w -> h w c'), rearrange(mask, 'c h w -> h w c')
        if self.use_int:
          image, mask = (image*255).type(torch.uint8),(mask*255).type(torch.uint8)
        
        #image = mask * image + (1-mask)*self.process(image)[0]
        # TODO select class label


        example["image"]=image
        example["segmentation"]=mask
        example["class"] = class_label
        if self.compact:
          return example
        elif self.cond is not None:
          return example[self.cond], example[self.uncond]
        else:
          return example[self.uncond]

# must redefine also len?
class trainInriaDataset(InriaDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.val_length>0:
            self.ims, self.masks = self.ims[self.val_length:], self.masks[self.val_length:]

        #self.ims, self.masks = extract_patches(self.ims, self.masks, self.h, self.w, size=self.size, device=self.device, rgb_mask=self.rgb_mask, 
        #threshold=self.threshold, n_patches=self.n_patches) # doubled in tr and val
        self.length = len(self.ims)

    def __getitem__(self,i):
        batch = super().__getitem__(i)
        return batch

class valInriaDataset(InriaDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.val_length>0:
            self.ims, self.masks = self.ims[:self.val_length], self.masks[:self.val_length]

        #self.ims, self.masks = extract_patches(self.ims, self.masks, self.h, self.w, size=self.size, device=self.device, rgb_mask=self.rgb_mask, 
        #threshold=self.threshold, n_patches=self.n_patches) # doubled in tr and val
        self.length = len(self.ims)
    def __getitem__(self,i):
        batch = super().__getitem__(i)
        return batch

class InriaData(InriaDataset):
    def __init__(self,*args,**kwargs):
      super().__init__(args,kwargs)


    def __len__(self):
        return self.length*self.num_patches
    def __getitem__(self, n):
        #example = dict((k, self.labels[k][i]) for k in self.labels)
        example = dict()
        n_patch, idx = n % self.num_patches, n//self.num_patches
        if n_patch == 0:
          imfiles, maskfiles = make_patches(self.ims[idx]), make_patches(self.masks[idx])
        imfile, maskfile = imfiles[n_patch], maskfiles[n_patch] #mask gets corrupted during training, but in debuggign is fine!!
        #batch = preprocess_data(imfile, maskfile, self.device, mode=self.mode, size=self.size) # image, mask = process() instead of batch=process damn!! for LDM
        #image,mask = batch["image"], batch["mask"]
        if type(imfile) == "str":
          image, mask = self.pil_to_tensor(Image.open(imfile))/255, self.pil_to_tensor(Image.open(maskfile).convert("RGB"))/255
        else:
          image, mask = imfile/255, maskfile/255
        if self.transforms is not None and self.cond is not None:
          out = self.input_T(image, mask)
          image,mask = out[0][0], out[1][0]
        elif self.cond is None:
          image = self.input_T(image)[0]
        if image.shape[-1]>image.shape[0] and self.ch_last:
          image, mask = rearrange(image, 'c h w -> h w c'), rearrange(mask, 'c h w -> h w c')


        example["image"]=image
        example["segmentation"]=mask
        if self.compact:
          return example
        elif self.cond is not None:
          return example[self.cond], example[self.uncond]
        else:
          return example[self.uncond]

class myMNIST(torchvision.datasets.MNIST):
  def __getitem__(self,i):
    x,y = super().__getitem__(i)
    #x = rearrange(x, "c h w->h w c")
    vocab = {}
    vocab["image"], vocab["class_label"] = x, y
    return vocab

class myCIFAR10(torchvision.datasets.CIFAR10):
  def __getitem__(self,i):
    x,y = super().__getitem__(i)
    vocab = {}
    vocab["image"], vocab["class_label"] = x, y
    return vocab

# to handle space limitations: provide custom windowing for patches + custom choice of tile (instead of :length) 
class CloudMaskDataset(Dataset):
   def __init__(self, root="../data/Sentinel-2-CMC", classes=['agricultural', 'urban/developed', 'hills/mountains'], percents=[50,25,70], 
   size=64, num_patches=200, ratio=0, length=3, load_from_disk=True,
                transforms=None):
      self.orig_size, self.step = (1022,1022), int((1-ratio)*size)
      self.np_I, self.np_J = int((self.orig_size[0]-size)/self.step+1),int((self.orig_size[1]-size)/self.step+1) # may be the opposite
      self.num_patches, self.size = min(num_patches, (self.np_J*self.np_I)), size # np_I*np_J
      self.load_from_disk = load_from_disk
      self.transforms, self.eq = transforms, torchvision.transforms.RandomEqualize(p=0.7)
      self.img_path, self.mask_path = os.path.join(root,"subscenes"), os.path.join(root,"masks")
      db = pd.read_csv(root + "/classification_tags.csv", index_col="index")

      key_percents = ["clear_percent", "cloud_percent", "shadow_percent"]
      cover_dict = {key:val for key,val in zip(key_percents,percents)}
      #'agricultural', 'urban/developed', 'hills/mountains'
      db_constr = (db["snow/ice"]==0)*(db["clear_percent"]>=percents[0])*(db["cloud_percent"]>=percents[1])
      db_truth = (db[classes[0]]==1) if classes is not [] else (db["agricultural"]==0) | (db["agricultural"]==1)
      for key in classes: db_truth += (db[key]==1)
      db_truth = db_truth * db_constr
      db = db[db_truth]

      names = db["scene"]
      self.names = names[:length] if length > 0 and length<len(names) else names
      self.index, self.db = self.names.index, db
      self.length = len(self.names)*self.num_patches
      print(f"extracting {self.length} patches from {len(self.names)} tiles")
      if self.num_patches > 0 and not load_from_disk:
        self.ims, self.ms = self.make_patches()
        assert len(self.ims)==self.length/self.num_patches, print(f"expected length {self.length} vs actual length {len(self.ims)}")

   def __len__(self):
      return self.length
   def __getitem__(self,i):
      n, npatches = int(i/self.num_patches), i%self.num_patches
      if self.load_from_disk:
        self.imgs, self.masks = np.load(os.path.join(self.img_path, self.names[self.index[n]]+".npy")),\
        np.load(os.path.join(self.mask_path, self.names[self.index[n]]+".npy"))
        self.imgs = self.imgs[...,[3,2,1]]
        self.imgs = np.clip(self.imgs, 0,1)
        self.imgs, self.masks = torch.from_numpy(self.imgs).permute(2,0,1), torch.from_numpy(self.masks).permute(2,0,1).to(torch.float32)[None,1]
      else: self.imgs,self.masks = self.ims[n],self.ms[n]
      #self.imgs = self.imgs.clip(0.,1.)

      idx_i, idx_j=int(npatches/self.np_J)*self.step, int(npatches%self.np_J)*self.step # added int
      #self.step original
      img,mask = self.imgs[:,idx_i:idx_i+self.size, idx_j:idx_j+self.size], self.masks[:,idx_i:idx_i+self.size, idx_j:idx_j+self.size]
      #img = self.eq((img*255).to(torch.uint8))/255
      #if img.mean() < 0.2: img = F.adjust_brightness(img, 3)

      if self.transforms is not None:
        out = self.transforms(torch.cat((img,mask),0))
        img,mask = out[:3], out[3][None]
      
      #img,mask = img*2.-1., mask*2.-1.
      batch = {}
      batch["image"],batch["segmentation"]=img,mask
      batch["orig_image"],batch["orig_segm"]=self.imgs, self.masks # debug output
      return batch
   
   def make_patches(self):
      ims, ms = [],[]
      for n in range(len(self.names)):
        print(f"appending image {n}")
        self.imgs, self.masks = np.load(os.path.join(self.img_path, self.names[self.index[n]]+".npy")), np.load(os.path.join(self.mask_path, self.names[self.index[n]]+".npy"))
        self.imgs = self.imgs[...,[3,2,1]]
        self.imgs = np.clip(self.imgs, 0,1)
        self.imgs, self.masks = torch.from_numpy(self.imgs).permute(2,0,1), torch.from_numpy(self.masks).permute(2,0,1).to(torch.float32)[None,1]
        ims.append(self.imgs), ms.append(self.masks)
      return ims, ms
   
class OSCD(Dataset):
  def __init__(self,pw=64,ph=64,sw=32,sh=32,mnh=10,mnw=10,mxw=50,mxh=50, clip=0.3, mult=1, transform=None, length=None, fake=False, train=True):
    self.fake = fake
    if fake: self.path='../data/OSCD_p_dataset_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(pw,ph,sw,sh,mnw, mnh, mxw, mxh,clip)
    if fake and mult > 1: self.path='../data/OSCD_p_dataset_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(pw,ph,sw,sh,mnw, mnh, mxw, mxh,clip, mult)
    if not fake:
       if train: self.path = "../data/OSCD_{}_{}/train".format(pw,sw)
       else: self.path = "../data/OSCD_{}_{}/test".format(pw,sw)
    #self.traindirs = os.open("train.text")
    self.img_names = sorted(glob.glob(os.path.join(self.path,"*imgs_2_rect-rgb*")))
    self.gt_names = sorted(glob.glob(os.path.join(self.path,"*imgs_1_rect-rgb*")))
    self.label_names = sorted(glob.glob(os.path.join(self.path, '*lbl*')))
    self.to_tensor = torchvision.transforms.ToTensor()
    self.transforms, self.normalize = transform, torchvision.transforms.Normalize(mean=(0.5), std=(0.5))
    if length is not None:
       self.img_names, self.label_names = self.img_names[:length],self.label_names[:length]

  def __len__(self):
      return len(self.img_names)
    
  def __getitem__(self,n):
      batch = {}
      img, label = Image.open(self.img_names[n]), Image.open(self.label_names[n]).convert("L")
      img, label = self.to_tensor(img), self.to_tensor(label) # :-1
      if img.shape[0]==4: img = img[:-1] # RGBA image
      #img = img/255
      if self.transforms is not None:
        #out = self.transforms(torch.cat((img,label),0))
        #img,label = out[:3], out[3][None]
        img = self.transforms(img)
      batch["image"], batch["segmentation"] = img, label
      return batch
  
class SARWakeDataset(Dataset):
  def __init__(self, root="../data/SARWake", mode="train", size=64, num_patches=200, ratio=0.5, orig = (501,501), length=1, transforms=None):
    self.root = root + "/train2017" if mode == "train" else root + "/val2017"
    self.db = pd.read_csv(self.root + "/train_csv.csv") if mode == "train" else pd.read_csv(self.root + "/val_csv.csv")
    self.names = self.db["filename"][:length]
    self.size = size
    self.orig_size, self.step = orig, int((1-ratio)*size)
    self.np_I, self.np_J = int((self.orig_size[0]-self.size)/self.step+1)+1*int(self.orig_size[0]>self.size), int((self.orig_size[1]-self.size)/self.step+1)+1*int(self.orig_size[1]>self.size) # may be the opposite
    self.np_I, self.np_J = max(self.np_I,1), max(self.np_J,1)
    self.num_patches, self.size = min(num_patches, (self.np_J*self.np_I)), size # np_I*np_J
    self.transforms = transforms
    self.to_tensor = torchvision.transforms.ToTensor()
    self.orig_sizes, self.tot_patches, self.single_patches, patch_to_tile = self.make_patches(self.size, self.num_patches)
    self.sump = sum(self.single_patches)
  
  def __len__(self):
     return self.tot_patches[-1]
  
  def __getitem__(self,i):
    # retrieve n from i
    num_patches = min(self.tot_patches[self.tot_patches>i])
    n = int(i/num_patches)
    patch_idx = i%self.single_patches[n]
    tile = self.to_tensor(Image.open(os.path.join(self.root, self.names[n])))[0][None] # be careful to grayscale opening
    orig_size = tile.shape[1:]
    np_I, np_J = int((orig_size[0]-self.size)/self.step+1)+1*int(orig_size[0]>self.size), int((orig_size[1]-self.size)/self.step+1)+1*int(orig_size[1]>self.size)
    np_I, np_J = max(np_I,1), max(np_J,1)
    idx_i, idx_j=int(patch_idx/np_J)*self.step, int(patch_idx%np_J)*self.step
    idx_i, idx_j = max(min(idx_i, orig_size[0]-self.size-1),0), max(min(idx_j, orig_size[1]-self.size-1),0)
    print(idx_i, idx_j)
    img = tile[:,idx_i:idx_i+self.size, idx_j:idx_j+self.size]

    if self.transforms is not None:
      out = self.transforms(img)
      img = out
    if len(img.shape)==2: img = img[None]
    
    #img,mask = img*2.-1., mask*2.-1.
    batch = {}
    batch["image"]=img
    return batch

  def make_patches(self,size,n_patches, ratio=0.5):
    sizes, num_patches, single_patches, patch_to_tile = [], [], [], {}
    for name in self.names:
      tile = Image.open(os.path.join(self.root, name))
      orig_size = tile.size
      np_I, np_J = int((orig_size[0]-self.size)/self.step+1)+1*int(orig_size[0]>self.size), int((orig_size[1]-self.size)/self.step+1)+1*int(orig_size[1]>self.size)
      np_I, np_J = max(np_I,1), max(np_J,1)
      n_patches = min(n_patches, (np_J*np_I)) # 3 corner patches
      sizes.append(orig_size), num_patches.append(n_patches+num_patches[-1]) if len(num_patches)>0 else num_patches.append(n_patches), single_patches.append(n_patches) # better if included in csv file
      patch_to_tile[n_patches]=name
    return np.array(sizes), np.array(num_patches), single_patches, patch_to_tile
  
class EuroSAT(Dataset):
   def __init__(self, transforms = None, root = "../data/EuroSAT_RGB", size=64, ratio=0.5, length=None):
      self.folders = os.listdir(root)
      self.files = []
      for folder in self.folders:
         print(folder)
         files = glob.glob(os.path.join(root, folder,"*.jpg"))
         print(files[0])
         for file in files: 
            self.files.append(file)
      self.length = len(self.files) # hw is it handled by random split?
      self.to_tensor = torchvision.transforms.ToTensor()
      self.transforms = transforms

   def __len__(self):
    return self.length
   
   def __getitem__(self, n):
      img = Image.open(self.files[n])
      batch = {}
      img = self.to_tensor(img)

      if self.transforms is not None:
        out = self.transforms(img)
        img = out
      if len(img.shape)==2: img = img[None]

      batch["image"] = img

      return batch

    