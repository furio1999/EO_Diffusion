import argparse, os, sys, datetime, glob, importlib, csv
import numpy as np
import time
import torch
import torchvision
import pytorch_lightning as pl
from tqdm import tqdm
from einops import rearrange

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
  vocab = {}
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

  im_patches = im_patches#.to(device, dtype=torch.uint8) # maybe rearrange dims, careful for next rearrange in preprocess_data
  im_patches = rearrange(im_patches, 'b h w c -> b c h w')

  if outpath is None:
    return im_patches

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
    use_val=None, ch_last=True, load_from_disk=False):
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
        images, masks = select_class(images, masks, cat=category)

        if length > 0 and length < len(images):
          jump = len(images)//length
          images, masks = images[:length*jump:jump], masks[:length*jump:jump]
        temp = list(zip(images,masks))
        random.shuffle(temp)
        images,masks = zip(*temp)
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
          self.ims, self.masks = make_patches(images, in_ch=img_ch, num_patches = self.num_patches, size=self.size, ratio=patch_overlap), make_patches(masks, num_patches=self.num_patches, in_ch=mask_ch, size=self.size, ratio=patch_overlap) #images, masks
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
        

        if self.transforms is not None:
            if self.cond is not None:
                data_keys=2*['input']
            else:
                data_keys=['input']

            self.input_T=KA.container.AugmentationSequential(
                *self.transforms,
                data_keys=data_keys,
                same_on_batch=False
            )  

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
        #batch = preprocess_data(imfile, maskfile, self.device, mode=self.mode, size=self.size) # image, mask = process() instead of batch=process damn!! for LDM
        #image,mask = batch["image"], batch["mask"]
        if type(imfile) == str:
          image, mask = self.pil_to_tensor(Image.open(imfile))/255, self.pil_to_tensor(Image.open(maskfile).convert("L"))/255
        else:
          image, mask = imfile/255, maskfile/255
        if self.transforms is not None:
          out = self.input_T(image, mask)
          image,mask = out[0], out[1]
          #image,mask = image[0],mask[0]
        if image.shape[-1]>image.shape[0] and self.ch_last:
          image, mask = rearrange(image, 'c h w -> h w c'), rearrange(mask, 'c h w -> h w c')
        
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
    x = x[:,2:26,2:26]
    return x

class myCIFAR10(torchvision.datasets.CIFAR10):
  def __getitem__(self,i):
    x,y = super().__getitem__(i)
    return x

class TestDataset(Dataset):
  def __init__(self, num_images=2000, size=64, img_ch=3, mask_ch=1,uncond="image", cond="class"):
    self.cond, self.uncond = cond, uncond
    self.ims, self.masks, self.targets = torch.ones((num_images,img_ch, size,size)),torch.ones((num_images,mask_ch, size,size)), torch.full((num_images,),1)

  def __len__(self):
    return self.ims.shape[0]
  
  def __getitem__(self,n):
    vocab = dict()
    vocab["image"],vocab["mask"],vocab["class"] = self.ims[n],self.masks[n],self.targets[n]
    return vocab[self.cond],vocab[self.uncond]