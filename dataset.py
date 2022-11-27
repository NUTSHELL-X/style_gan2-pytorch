import torch
from torch.utils.data import Dataset,DataLoader
import os
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.datasets import ImageFolder
ds_folder='./MineCraft-RT_1280x720_v12'

class SingleFolderDataset(Dataset): #返回一个Dataset的实例，用于读取单个文件夹内的图片
    def __init__(self,folder,transform):
        self.folder=folder
        self.transform=transform

    def __len__(self):
        return len(os.listdir(self.folder))
    
    def __getitem__(self, idx):
        image=Image.open(os.path.join(self.folder,os.listdir(self.folder)[idx]))
        image=np.array(image)
        if self.transform:
            image=self.transform(image=image)['image']
        return image

class Transforms:
    def __init__(self, transforms: A.Compose):
        self.transforms = transforms

    def __call__(self, img, *args, **kwargs):
        return self.transforms(image=np.array(img))['image']

def create_dataloader(res,batch_size):
    if isinstance(res,list) or isinstance(res,tuple):
        h,w=res
    else:
        h=res
        w=res
    transforms=A.Compose(
        [
            A.Resize(height=int(h/0.9),width=int(w/0.9)),
            A.RandomCrop(h,w),
            A.RGBShift(10,10,10),
            A.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.1,hue=0),
            A.Normalize(mean=(0,0,0),std=(1,1,1)),
            ToTensorV2(),
        ]
    )
    ds=ImageFolder(root=ds_folder,transform=Transforms(transforms))
    dl=DataLoader(dataset=ds,batch_size=batch_size,shuffle=True,drop_last=True)
    return dl