import torch
from torch.utils.data import Dataset,DataLoader
from torchvision.io import read_image,ImageReadMode
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# import cv2

def plot_images(images,max_num=16,n_col=4):
    if isinstance(images,torch.Tensor):
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

def print_networks(net):
    """Print the total number of parameters in the network and (if verbose) network architecture

    Parameters:
        verbose (bool) -- if verbose: print the network architecture
    """
    print('---------- Networks initialized -------------')
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('[Network] Total number of parameters : %.3f M' % (num_params / 1e6))
    print('-----------------------------------------------')