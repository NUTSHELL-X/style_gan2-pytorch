import torch
import torch.nn as nn
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

    """
    if not isinstance(net,nn.Module):
        print('net type error')
    print('net layers and weights:')
    for name,param in net.named_parameters():
        print('layer: ',name,'param dtype: ',param.dtype)
    print('-----------------------------------------------')
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('[Network] Total number of parameters : %.3f M' % (num_params / 1e6))
    print('-----------------------------------------------')

def BCE_loss_disc(disc_real,disc_fake):
    loss_f = nn.BCELoss()
    bs = disc_real.shape[0]
    device = disc_real.device
    dtype= disc_real.dtype
    ones = torch.ones(bs).to(device).to(dtype)
    zeros = torch.zeros(bs).to(device).to(dtype)
    loss = loss_f(disc_real,ones)+loss_f(disc_fake,zeros)

    return loss

def BCE_loss_gen(disc_fake):
    loss_f = nn.BCELoss()
    bs = disc_fake.shape[0]
    device = disc_fake.device
    dtype = disc_fake.dtype
    ones = torch.ones(bs).to(device).to(dtype)
    loss = loss_f(disc_fake,ones)
    
    return loss 

def WGAN_loss_disc(disc_real,disc_fake):
    loss_f = torch.mean
    loss = -loss_f(disc_real)+loss_f(disc_fake)

    return loss 

def WGAN_loss_gen(disc_fake):
    loss_f = torch.mean
    loss = -loss_f(disc_fake)

    return loss

def save_weights(net,save_path,gpus):
    if len(gpus) > 1 and torch.cuda.is_available():
        torch.save(net.module.cpu().state_dict(), save_path)
    else:
        torch.save(net.cpu().state_dict(), save_path)

def save_training_params(opt_gen,opt_disc,total_epochs,save_path):
    torch.save({
        'opt_disc_state_dict':opt_disc.state_dict(),
        'opt_gen_state_dict':opt_gen.state_dict(),
        'total_epochs':total_epochs,
    },save_path)