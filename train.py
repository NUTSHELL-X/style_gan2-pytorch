import torch
import torch.nn as nn
import numpy
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
from utils import plot_images
import os
from tqdm import tqdm
from dataset import create_dataloader
from model import Generator,Discriminator
import torch.optim as optim
from torchvision.utils import save_image
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import MultiStepLR

epoch=20
batch_size=32
start_res=(4,4)
start_c=256
w_c=512
save_path='Diana_saved2.pt'
device='cuda' if torch.cuda.is_available() else "cpu"
gen=Generator(start_res=start_res,w_c=w_c,start_c=start_c).to(device)
disc=Discriminator(start_res=start_res,start_c=start_c).to(device)
opt_gen=optim.Adam(gen.parameters(),lr=1e-4)
opt_disc=optim.Adam(disc.parameters(),lr=1e-4)
sche_gen=MultiStepLR(opt_gen, milestones=[30,80,150,200,250], gamma=0.7)
sche_disc=MultiStepLR(opt_disc, milestones=[30,80,150,200,250], gamma=0.7)
total_epochs=0
if os.path.exists(save_path):
    print('loading data from saved weights')
    checkpoint=torch.load(save_path)
    disc.load_state_dict(checkpoint['disc_state_dict'])
    gen.load_state_dict(checkpoint['gen_state_dict'])
    opt_disc.load_state_dict(checkpoint['disc_opt_state_dict'])
    opt_gen.load_state_dict(checkpoint['gen_opt_state_dict'])
    sche_gen.load_state_dict(checkpoint['sche_gen_state_dict'])
    sche_disc.load_state_dict(checkpoint['sche_disc_state_dict'])
    total_epochs=checkpoint['total_epochs']
    print('total_epochs:',total_epochs)
else:
    print('initializing from scratch')

scaler_gen=torch.cuda.amp.GradScaler()
scaler_disc=torch.cuda.amp.GradScaler()
loss_fn=nn.BCEWithLogitsLoss()
base_tensor=torch.ones((batch_size,start_c,start_res[0],start_res[1])).to(device)
def train_fn(epochs):
    for i in range(epochs):
        dataloader=create_dataloader(256,batch_size)
        for idx,(real,labels) in tqdm(enumerate(dataloader)):
            #train disc
            real=real.to(device)
            # print('real shape:',real.shape)
            latent=torch.randn(batch_size,w_c).to(device)
            fake=gen([base_tensor,latent])
            opt_disc.zero_grad()
            with autocast(dtype=torch.bfloat16):
                disc_real=disc(real)
                disc_fake=disc(fake.detach())
                # d_loss=-torch.mean(disc(real))+torch.mean(disc(fake.detach()))
                d_loss=loss_fn(disc_real.reshape(batch_size),torch.ones(batch_size).to(device))+loss_fn(disc_fake.reshape(batch_size),torch.zeros(batch_size).to(device))
                print('d_loss:',d_loss.item())
            scaler_disc.scale(d_loss).backward()
            scaler_disc.step(opt_disc)
            scaler_disc.update()
            #train gen
            opt_gen.zero_grad()
            # g_loss=-torch.mean(disc(fake))
            with autocast(dtype=torch.bfloat16):
                g_loss=loss_fn(disc(fake).reshape(batch_size),torch.ones(batch_size).to(device))
                print('g_loss',g_loss.item())
            scaler_gen.scale(g_loss).backward()
            scaler_gen.step(opt_gen)
            scaler_gen.update()

        sche_disc.step()
        sche_gen.step()
        save_image(fake.cpu().detach()[0],f'generated_Diana2/generated_img_{total_epochs+i}.jpg')


if __name__=='__main__':
    epochs=40
    train_fn(epochs)
    total_epochs+=epochs
    # n_examples=4
    # latent=torch.randn(n_examples,256).to(device)
    # generated_imgs=gen(latent).cpu().detach()
    # for i in range(n_examples):
    #     save_image(generated_imgs[0],f'generated_img/generated_img_{3}_{i}.jpg')
    torch.save({
        'disc_state_dict':disc.state_dict(),
        'gen_state_dict':gen.state_dict(),
        'disc_opt_state_dict':opt_disc.state_dict(),
        'gen_opt_state_dict':opt_gen.state_dict(),
        'total_epochs':total_epochs,
        'sche_gen_state_dict':sche_gen.state_dict(),
        'sche_disc_state_dict':sche_disc.state_dict(),
    },save_path)