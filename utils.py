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

def config_parser():
    import configargparse
    parser=configargparse.ArgumentParser()
    parser.add_argument('--config',is_config_file=True,help='config file path',default='config.txt')
    parser.add_argument('--model_path',type=str) # path to save weights(.pt file)
    parser.add_argument('--generated_image_folder',type=str) # path to save generated images(.jpg file) 
    parser.add_argument('--dataset_path',type=str) # path containing dataset
    parser.add_argument('--epoch',type=int)
    parser.add_argument('--dtype',type=str) # data type used for training(fp16,fp32,mixed)
    parser.add_argument('--start_res',nargs='+',type=int) # 
    parser.add_argument('--upscale_times',type=int)
    parser.add_argument('--start_c',type=int) # channels for input constant(torch.ones)
    parser.add_argument('--w_c',type=int) # channels for style code
    parser.add_argument('--batch_size',type=int)
    parser.add_argument('--lr',type=float) # learning rate
    parser.add_argument('--lr_decay',type=bool) # learning rate decay or not
    parser.add_argument('--milestones',type=int,nargs='+',action='append') # milestones for learning rate decay
    parser.add_argument('--gamma',type=float) # learning rate decay ratio

    return parser