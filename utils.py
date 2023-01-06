import torch
from torch.utils.data import Dataset,DataLoader
from torchvision.io import read_image,ImageReadMode
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# import cv2

def plot_images(images,max_num=16,n_col=4):
    if isinstance(images,tf.Tensor):
        if len(images.shape)==3:
            images=tf.expand_dims(images,0)
        images=images.numpy()
    elif isinstance(images,torch.Tensor):
        if len(images.shape)==3:
            images=images.unsqueeze(0)
        images=images.permute(0,2,3,1)
        images=images.numpy()
    elif isinstance(images,np.ndarray):
        if len(images.shape)==3:
            images=np.expand_dims(images,0)
    
    image_num=min(images.shape[0],max_num)
    assert n_col>=1
    n_row=np.ceil(image_num/n_col)
    for i in range(image_num):
        plt.subplot(n_row,n_col,i+1)
        plt.imshow(images[i])
    plt.show()