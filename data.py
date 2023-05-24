import os
import numpy as np
from PIL import Image
import scipy, scipy.io
from easydict import EasyDict
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch import Generator
from torch.utils.data import random_split
#os.sys.path.append("../EO-Diffusion/scripts")
from data_load import *
from torch import nn

class MyTransform(nn.Module):
    def __init__(self, transforms):
        super().__init__()
        self.transforms=transforms
    def __call__(self,imgs):
        t = self.transforms
        return [t(img) for img in imgs]


def create_mnist_dataloaders(batch_size,image_size=28,num_workers=4):
    
    preprocess=transforms.Compose([transforms.Resize(image_size),\
                                    transforms.ToTensor(),\
                                    transforms.Normalize([0.5],[0.5])]) #[0,1] to [-1,1]

    train_dataset=myMNIST(root="../data/mnist_data",\
                        train=True,\
                        download=True,\
                        transform=preprocess
                        )
    test_dataset=myMNIST(root="../data/mnist_data",\
                        train=False,\
                        download=True,\
                        transform=preprocess
                        )

    return DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers),\
            DataLoader(test_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)

def create_cifar10_dataloaders(batch_size,image_size=32,num_workers=4):
    
    preprocess=transforms.Compose([
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),]) #[0,1] to [-1,1]

    train_dataset=datasets.CIFAR10(root="./cifar_data",\
                        train=True,\
                        download=True,\
                        transform=preprocess
                        )
    test_dataset=datasets.CIFAR10(root="./cifar_data",\
                        train=False,\
                        download=True,\
                        transform=preprocess
                        )

    return DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers),\
            DataLoader(test_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)

def create_inria_dataloaders(batch_size, size=64, patch_overlap=0.5, num_workers=0, val_split = 0.15, SEED=4097, test=False, device="cpu", length=3, num_patches=200,
return_dataset = False):
        preprocess=transforms.Compose([
                                   transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip(), #grayscale gives problems
                                   #A.Normalize([0.5],[0.5]),
                                   ]) if not test or return_dataset else transforms.Compose([transforms.CenterCrop(size=size)])
        print("loading dataset...")
        dataset = InriaDataset(path = "../EO-Diffusion/data/AerialImageDataset", transforms=preprocess, compact=True, size = size,length=length,
        ch_last=False,img_ch=3, mask_ch=1, uncond="image", num_patches=num_patches, patch_overlap=patch_overlap, load_from_disk=False, use_int=return_dataset)

        train_ds, test_ds = random_split(dataset, [1-val_split, val_split], generator=Generator().manual_seed(SEED))
        if return_dataset: return train_ds, test_ds
        print("Loaded!!")
        return DataLoader(train_ds,batch_size=batch_size,shuffle=True,num_workers=num_workers),\
                DataLoader(test_ds,batch_size=batch_size,shuffle=True,num_workers=num_workers)

def create_cloud_dataloaders(batch_size, num_workers=0, val_split=0.15, SEED=4097, return_dataset=False, test=False, **kwargs):
            preprocess=transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip(),
                                            #A.Normalize([0.5],[0.5]), and brightness
]) if not test else None
            dataset = CloudMaskDataset(transforms=preprocess,**kwargs)
            train_ds, test_ds = random_split(dataset, [1-val_split, val_split], generator=Generator().manual_seed(SEED))
            if return_dataset: return train_ds, test_ds
            return DataLoader(train_ds,batch_size=batch_size,shuffle=True,num_workers=num_workers),\
                DataLoader(test_ds,batch_size=batch_size,shuffle=True,num_workers=num_workers)


def create_test_dataloaders(batch_size, image_size=64, num_workers=0, val_split = 0.15, SEED=4097, num_images=20):
        preprocess=[
                                   transforms.RandomHorizontalFlip(),
                                   transforms.Normalize([0.5],[0.5]),
                                   ]
        dataset = TestDataset(num_images=num_images)

        train_ds, test_ds = random_split(dataset, [1-val_split, val_split], generator=Generator().manual_seed(SEED))

        return DataLoader(train_ds,batch_size=batch_size,shuffle=True,num_workers=num_workers),\
                DataLoader(test_ds,batch_size=batch_size,shuffle=True,num_workers=num_workers)


def get_metadata(name):
    if name == "mnist":
        metadata = EasyDict(
            {
                "image_size": 28,
                "num_classes": 10,
                "train_images": 60000,
                "val_images": 10000,
                "num_channels": 1,
            }
        )
    elif name == "mnist_m":
        metadata = EasyDict(
            {
                "image_size": 28,
                "num_classes": 10,
                "train_images": 60000,
                "val_images": 10000,
                "num_channels": 3,
            }
        )
    elif name == "cifar10":
        metadata = EasyDict(
            {
                "image_size": 32,
                "num_classes": 10,
                "train_images": 50000,
                "val_images": 10000,
                "num_channels": 3,
            }
        )
    elif name == "melanoma":
        metadata = EasyDict(
            {
                "image_size": 64,
                "num_classes": 2,
                "train_images": 33126,
                "val_images": 0,
                "num_channels": 3,
            }
        )
    elif name == "afhq":
        metadata = EasyDict(
            {
                "image_size": 64,
                "num_classes": 3,
                "train_images": 14630,
                "val_images": 1500,
                "num_channels": 3,
            }
        )
    elif name == "celeba":
        metadata = EasyDict(
            {
                "image_size": 64,
                "num_classes": 4,
                "train_images": 109036,
                "val_images": 12376,
                "num_channels": 3,
            }
        )
    elif name == "cars":
        metadata = EasyDict(
            {
                "image_size": 64,
                "num_classes": 196,
                "train_images": 8144,
                "val_images": 8041,
                "num_channels": 3,
            }
        )
    elif name == "flowers":
        metadata = EasyDict(
            {
                "image_size": 64,
                "num_classes": 102,
                "train_images": 2040,
                "val_images": 6149,
                "num_channels": 3,
            }
        )
    elif name == "gtsrb":
        metadata = EasyDict(
            {
                "image_size": 32,
                "num_classes": 43,
                "train_images": 39252,
                "val_images": 12631,
                "num_channels": 3,
            }
        )
    else:
        raise ValueError(f"{name} dataset nor supported!")
    return metadata


class oxford_flowers_dataset(Dataset):
    def __init__(self, indexes, labels, root_dir, transform=None):
        self.images = []
        self.targets = []
        self.transform = transform

        for i in indexes:
            self.images.append(
                os.path.join(
                    root_dir,
                    "jpg",
                    "image_" + "".join(["0"] * (5 - len(str(i)))) + str(i) + ".jpg",
                )
            )
            self.targets.append(labels[i - 1] - 1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        target = self.targets[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, target


# TODO: Add datasets imagenette/birds/svhn etc etc.
def get_dataset(name, data_dir, metadata):
    """
    Return a dataset with the current name. We only support two datasets with
    their fixed image resolutions. One can easily add additional datasets here.

    Note: To avoid learning the distribution of transformed data, don't use heavy
        data augmentation with diffusion models.
    """
    if name == "mnist":
        preprocess=transforms.Compose([transforms.Resize(metadata.image_size),\
                                        transforms.ToTensor(),\
                                        transforms.Normalize([0.5],[0.5])]) #[0,1] to [-1,1]
        train_set = datasets.MNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=preprocess,
        )
        test_dataset=datasets.MNIST(root="./mnist_data",\
                            train=False,\
                            download=True,\
                            transform=preprocess
                            )
    elif name == "mnist_m":
        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    metadata.image_size, scale=(0.8, 1.0), ratio=(0.8, 1.2)
                ),
                transforms.ToTensor(),
            ]
        )
        train_set = datasets.ImageFolder(
            data_dir,
            transform=transform_train,
        )
    elif name == "cifar10":
        transform_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        train_set = datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=transform_train,
        )
    elif name in ["imagenette", "melanoma", "afhq"]:
        transform_train = transforms.Compose(
            [
                transforms.Resize(74),
                transforms.RandomCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        train_set = datasets.ImageFolder(
            data_dir,
            transform=transform_train,
        )
    elif name == "celeba":
        # celebA has a large number of images, avoiding randomcropping.
        transform_train = transforms.Compose(
            [
                transforms.Resize(64),
                transforms.CenterCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        train_set = datasets.ImageFolder(
            data_dir,
            transform=transform_train,
        )
    elif name == "cars":
        transform_train = transforms.Compose(
            [
                transforms.Resize(64),
                transforms.RandomCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        train_set = datasets.ImageFolder(
            data_dir,
            transform=transform_train,
        )
    elif name == "flowers":
        transform_train = transforms.Compose(
            [
                transforms.Resize(64),
                transforms.RandomCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        splits = scipy.io.loadmat(os.path.join(data_dir, "setid.mat"))
        labels = scipy.io.loadmat(os.path.join(data_dir, "imagelabels.mat"))
        labels = labels["labels"][0]
        train_set = oxford_flowers_dataset(
            np.concatenate((splits["trnid"][0], splits["valid"][0]), axis=0),
            labels,
            data_dir,
            transform_train,
        )
    elif name == "gtsrb":
        # celebA has a large number of images, avoiding randomcropping.
        transform_train = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ]
        )
        train_set = datasets.ImageFolder(
            data_dir,
            transform=transform_train,
        )
    else:
        raise ValueError(f"{name} dataset nor supported!")
    return train_set


def remove_module(d):
    return OrderedDict({(k[len("module.") :], v) for (k, v) in d.items()})


def fix_legacy_dict(d):
    keys = list(d.keys())
    if "model" in keys:
        d = d["model"]
    if "state_dict" in keys:
        d = d["state_dict"]
    keys = list(d.keys())
    # remove multi-gpu module.
    if "module." in keys[1]:
        d = remove_module(d)
    return d
