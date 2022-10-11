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

epoch=20
batch_size=32
start_res=(4,4)
start_c=256
w_c=512
save_path='Diana_saved.pt'
device='cuda' if torch.cuda.is_available() else "cpu"
gen=Generator(start_res=start_res,start_c=start_c).to(device)
disc=Discriminator(start_res=start_res,start_c=start_c).to(device)
opt_gen=optim.Adam(gen.parameters(),lr=3e-4)
opt_disc=optim.Adam(disc.parameters(),lr=1e-4)
total_epochs={}
if os.path.exists(save_path):
    print('loading data from saved weights')
    checkpoint=torch.load(save_path)
    disc.load_state_dict(checkpoint['disc_state_dict'])
    gen.load_state_dict(checkpoint['gen_state_dict'])
    opt_disc.load_state_dict(checkpoint['disc_opt_state_dict'])
    opt_gen.load_state_dict(checkpoint['gen_opt_state_dict'])
    total_epochs=checkpoint['total_epochs']
    print('total_epochs:',total_epochs)
else:
    print('initializing from scratch')

loss_fn=nn.BCELoss()
base_tensor=torch.ones((batch_size,start_c,start_res[0],start_res[1])).to(device)
def train_fn(epochs):
    for step in epochs:
        epoch=epochs[step]
        dataloader=create_dataloader((start_res[0]*2**step,start_res[1]*2**step),batch_size)
        for i in range(epoch):
            for idx,(real,labels) in tqdm(enumerate(dataloader)):
                alpha=i/epoch if step not in total_epochs or total_epochs[step]==0 else 1.0
                print('alpha:',alpha)
                #train disc
                real=real.to(device)
                # print('real shape:',real.shape)
                latent=torch.randn(batch_size,w_c).to(device)
                fake=gen([base_tensor,latent],step,alpha)
                opt_disc.zero_grad()
                disc_real=disc(real,step,alpha)
                disc_fake=disc(fake.detach(),step,alpha)
                # d_loss=-torch.mean(disc(real))+torch.mean(disc(fake.detach()))
                d_loss=loss_fn(disc_real.sigmoid().reshape(batch_size),torch.ones(batch_size).to(device))+loss_fn(disc_fake.sigmoid().reshape(batch_size),torch.zeros(batch_size).to(device))
                print('d_loss:',d_loss.item())
                d_loss.backward()
                opt_disc.step()
                #train gen
                opt_gen.zero_grad()
                # g_loss=-torch.mean(disc(fake))
                g_loss=loss_fn(disc(fake,step,alpha).sigmoid().reshape(batch_size),torch.ones(batch_size).to(device))
                print('g_loss',g_loss.item())
                g_loss.backward()
                opt_gen.step()
                # latent=torch.randn(batch_size,w_c).to(device)
                # fake=gen([base_tensor,latent],5)

            save_image(fake.cpu().detach()[0],f'generated_Diana/generated_img_{step}_{i}.jpg')



if __name__=='__main__':
    epochs={5:60}
    train_fn(epochs)
    for i in epochs:
        if i not in total_epochs:
            total_epochs[i]=epochs[i]
        else:
            total_epochs[i]+=epochs[i]
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
    },save_path)