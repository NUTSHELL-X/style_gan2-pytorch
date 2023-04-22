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
from utils import config_parser

parser=config_parser()
args=parser.parse_args()
batch_size=args.batch_size
start_res=args.start_res
print(start_res,type(start_res))
start_c=args.start_c
w_c=args.w_c
save_images=True
auto_scale=True
model_path=args.model_path
lr=args.lr
upscale_times=args.upscale_times
final_h=start_res[0]*2**upscale_times
final_w=start_res[1]*2**upscale_times
generated_image_folder=args.generated_image_folder
# milestones=args.milestones
# print(milestones)
device=args.device if torch.cuda.is_available() else "cpu"
gen=Generator(start_res=start_res,w_c=w_c,start_c=start_c).to(device)
disc=Discriminator(start_res=start_res,start_c=start_c).to(device)
opt_gen=optim.Adam(gen.parameters(),lr=lr)
opt_disc=optim.Adam(disc.parameters(),lr=lr)
sche_gen=MultiStepLR(opt_gen, milestones=[30,80,150,200,250], gamma=0.7)
sche_disc=MultiStepLR(opt_disc, milestones=[30,80,150,200,250], gamma=0.7)
total_epochs=0
if os.path.exists(model_path):
    print('loading data from saved weights')
    checkpoint=torch.load(model_path)
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
if auto_scale:
    scaler_disc = torch.cuda.amp.GradScaler()
    scaler_gen = torch.cuda.amp.GradScaler()
dataloader=create_dataloader((final_h,final_w),batch_size)
def train_fn(epochs):
    for i in range(epochs):
        for idx,(real,labels) in tqdm(enumerate(dataloader)):
            #train disc
            real=real.to(device)
            latent=torch.randn(batch_size,w_c).to(device)
            fake=gen([base_tensor,latent])
            opt_disc.zero_grad()
            disc_real=disc(real)
            disc_fake=disc(fake.detach())
            # d_loss=-torch.mean(disc(real))+torch.mean(disc(fake.detach()))
            d_loss=loss_fn(disc_real.reshape(batch_size),torch.ones(batch_size).to(device))+loss_fn(disc_fake.reshape(batch_size),torch.zeros(batch_size).to(device))
            print('d_loss:',d_loss.item())
            d_loss.backward()
            opt_disc.step()
            # print("1:",next(iter(disc.parameters())).data)
            #train gen
            opt_gen.zero_grad()
            # g_loss=-torch.mean(disc(fake))
            g_loss=loss_fn(disc(fake).reshape(batch_size),torch.ones(batch_size).to(device))
            print('g_loss',g_loss.item())
            g_loss.backward()
            opt_gen.step()

            if idx%600 == 0:
                save_image(fake.cpu().detach()[0],generated_image_folder+f'generated_img_{total_epochs+i}_{idx}.jpg')
                
        if save_images:
            if not os.path.exists(generated_image_folder):
                os.mkdir(generated_image_folder)
            save_image(fake.cpu().detach()[0],generated_image_folder+f'generated_img_{total_epochs+i}.jpg')

def train_fn_auto_scaler(epochs):
    for i in range(epochs):
        for idx,(real,labels) in tqdm(enumerate(dataloader)):
            #train disc
            real=real.to(device)
            latent=torch.randn(batch_size,w_c).to(device)
            opt_disc.zero_grad()
            with torch.cuda.amp.autocast():
                fake=gen([base_tensor,latent])
                disc_real=disc(real)
                disc_fake=disc(fake.detach())
                # d_loss=-torch.mean(disc(real))+torch.mean(disc(fake.detach()))
                d_loss=loss_fn(disc_real.reshape(batch_size),torch.ones(batch_size).to(device))+loss_fn(disc_fake.reshape(batch_size),torch.zeros(batch_size).to(device))
            print('d_loss:',d_loss.item())
            scaler_disc.scale(d_loss).backward()
            scaler_disc.step(opt_disc)
            scaler_disc.update()
            # print("1:",next(iter(disc.parameters())).data)
            #train gen
            opt_gen.zero_grad()
            with torch.cuda.amp.autocast():
                # g_loss=-torch.mean(disc(fake))
                g_loss=loss_fn(disc(fake).reshape(batch_size),torch.ones(batch_size).to(device))
            print('g_loss',g_loss.item())
            scaler_gen.scale(g_loss).backward()
            scaler_gen.step(opt_gen)
            scaler_gen.update()
        
        if save_images:
            save_image(fake.cpu().detach()[0],generated_image_path+f'generated_img_{total_epochs+i}.jpg')

if __name__=='__main__':
    epochs=args.epoch
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
    },model_path)