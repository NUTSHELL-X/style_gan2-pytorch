import torch
import torch.nn as nn
import numpy
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
from utils import plot_images,save_weights,save_training_params,print_networks,BCE_loss_disc,BCE_loss_gen,\
WGAN_loss_disc,WGAN_loss_gen
import os
from tqdm import tqdm
from dataset import create_dataloader
from model import Generator,Discriminator
import torch.optim as optim
from torchvision.utils import save_image
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import MultiStepLR,StepLR
from options import config_parser

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

parser=config_parser()
args=parser.parse_args()
batch_size=args.batch_size
start_res=args.start_res
start_channels=args.start_channels
w_channels=args.w_channels
save_images=args.save_images
# auto_scale=True
model_path=args.model_path
lr=args.lr
upscale_times=args.upscale_times
final_h=start_res[0]*2**upscale_times
final_w=start_res[1]*2**upscale_times
dataset_type = args.dataset_type
dtype = torch.float16 if args.dtype=='float16' else torch.float32
generated_image_folder=args.generated_image_folder
device=args.device if torch.cuda.is_available() else "cpu"
print('using device {device}'.format(device=device))
# if os.path.exists(model_path):
#     print('loading data from saved weights')
#     checkpoint=torch.load(model_path)
#     disc.load_state_dict(checkpoint['disc_state_dict'])
#     gen.load_state_dict(checkpoint['gen_state_dict'])
#     opt_disc.load_state_dict(checkpoint['disc_opt_state_dict'])
#     opt_gen.load_state_dict(checkpoint['gen_opt_state_dict'])
#     sche_gen.load_state_dict(checkpoint['sche_gen_state_dict'])
#     sche_disc.load_state_dict(checkpoint['sche_disc_state_dict'])
#     total_epochs=checkpoint['total_epochs']
#     print('total_epochs:',total_epochs)
# else:
#     print('initializing from scratch')

scaler_gen=torch.cuda.amp.GradScaler()
scaler_disc=torch.cuda.amp.GradScaler()
loss_fn=nn.BCEWithLogitsLoss()
base_tensor=torch.ones((batch_size,start_channels,start_res[0],start_res[1])).to(device).to(dtype)
# if auto_scale:
#     scaler_disc = torch.cuda.amp.GradScaler()
#     scaler_gen = torch.cuda.amp.GradScaler()
dataloader=create_dataloader((final_h,final_w),batch_size,dataset_type)
def train_gen(net_gen,net_disc,opt_gen,opt,_disc):
    pass

def train_fn(gen,disc,opt_gen,opt_disc,sche_gen,sche_disc,loss_fn_gen,loss_fn_disc,total_epochs,epochs,dtype):
    for i in range(epochs):
        print('updating epoch: {}'.format(total_epochs+i))
        print('current generator learning rate: ',sche_gen.get_last_lr())
        print('current discriminator learning rate: ',sche_disc.get_last_lr())
        for idx,(real,labels) in tqdm(enumerate(dataloader)):
            #train disc
            real=real.to(device).to(dtype)
            batch_size = real.shape[0]
            latent=torch.randn(batch_size,w_channels).to(device).to(dtype)
            fake=gen([base_tensor,latent])
            opt_disc.zero_grad()
            disc_real=disc(real).squeeze()
            disc_fake=disc(fake.detach()).squeeze()
            d_loss=loss_fn_disc(disc_real,disc_fake)
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.)
            opt_disc.step()
            #train gen
            opt_gen.zero_grad()
            g_loss=loss_fn_gen(disc(fake).squeeze())
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.)
            opt_gen.step()
            if type(args.save_freq_inside_epoch) == int and idx%args.save_freq_inside_epoch == 0:
                save_image(fake.cpu().detach()[0].to(torch.float32),os.path.join(generated_image_folder,f'generated_img_{total_epochs+i}_{idx}.jpg'))

        print('d_loss: ',d_loss.item())
        print('g_loss: ',g_loss.item())
        sche_gen.step()
        sche_disc.step()
        if save_images:
            save_image(fake.cpu().detach()[0].to(torch.float32),os.path.join(generated_image_folder,f'generated_img_{total_epochs+i}.jpg'))

# def train_fn_auto_scaler(epochs):
#     for i in range(epochs):
#         for idx,(real,labels) in tqdm(enumerate(dataloader)):
#             #train disc
#             real=real.to(device)
#             latent=torch.randn(batch_size,w_channels).to(device)
#             opt_disc.zero_grad()
#             with torch.cuda.amp.autocast():
#                 fake=gen([base_tensor,latent])
#                 disc_real=disc(real)
#                 disc_fake=disc(fake.detach())
#                 # d_loss=-torch.mean(disc(real))+torch.mean(disc(fake.detach()))
#                 d_loss=loss_fn(disc_real.reshape(batch_size),torch.ones(batch_size).to(device))+loss_fn(disc_fake.reshape(batch_size),torch.zeros(batch_size).to(device))
#             print('d_loss:',d_loss.item())
#             scaler_disc.scale(d_loss).backward()
#             scaler_disc.step(opt_disc)
#             scaler_disc.update()
#             # print("1:",next(iter(disc.parameters())).data)
#             #train gen
#             opt_gen.zero_grad()
#             with torch.cuda.amp.autocast():
#                 # g_loss=-torch.mean(disc(fake))
#                 g_loss=loss_fn(disc(fake).reshape(batch_size),torch.ones(batch_size).to(device))
#             print('g_loss',g_loss.item())
#             scaler_gen.scale(g_loss).backward()
#             scaler_gen.step(opt_gen)
#             scaler_gen.update()
        
#         if save_images:
#             save_image(fake.cpu().detach()[0],generated_image_folder+f'generated_img_{total_epochs+i}.jpg')

if __name__=='__main__':
    epochs=args.epochs
    print('using dtype: ',dtype)
    gen=Generator(start_res=2*start_res,w_c=w_channels,start_c=start_channels,steps=upscale_times).to(device).to(dtype)
    gen=torch.nn.DataParallel(gen,device_ids=args.gpus)
    disc=Discriminator(start_res=start_res,start_c=start_channels,steps=upscale_times).to(device).to(dtype)
    disc=torch.nn.DataParallel(disc,device_ids=args.gpus)
    adam_eps = 1e-4 if dtype == torch.float16 else 1e-8
    opt_gen=optim.Adam(gen.parameters(),lr=lr,eps=adam_eps)
    opt_disc=optim.Adam(disc.parameters(),lr=lr,eps=adam_eps)
    loss_fn_gen = WGAN_loss_gen
    loss_fn_disc = WGAN_loss_disc
    if not os.path.exists(generated_image_folder) and save_images:
        os.mkdir(generated_image_folder)
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    if args.continues: # load weights
        if len(args.gpus) > 1:
            gen.module.load_state_dict(torch.load(args.gen_weights_path)) # 可能还有问题
            disc.module.load_state_dict(torch.load(args.disc_weights_path))
        else:
            gen.load_state_dict(torch.load(args.gen_weights_path))
            disc.load_state_dict(torch.load(args.disc_weights_path))
        print('loaded saved weights ; Generator:{},Discriminator:{}'.format(args.gen_weights_path,args.disc_weights_path))
        training_params=torch.load(args.training_params_path)
        opt_disc.load_state_dict(training_params['opt_disc_state_dict'])
        opt_gen.load_state_dict(training_params['opt_gen_state_dict'])
        # sche_gen.load_state_dict(training_params['sche_gen_state_dict'])
        # sche_disc.load_state_dict(training_params['sche_disc_state_dict'])
        total_epochs=training_params['total_epochs']
        print('total_epochs:',total_epochs)
    else:
        total_epochs=0
        print('initializing from scratch')
    sche_gen=StepLR(opt_gen,step_size=args.lr_step_size,gamma=args.gamma)
    sche_disc=StepLR(opt_disc,step_size=args.lr_step_size,gamma=args.gamma)
    print('total_epochs:',total_epochs)
    train_fn(gen,disc,opt_gen,opt_disc,sche_gen,sche_disc,loss_fn_gen,loss_fn_disc,total_epochs,epochs,dtype)
    total_epochs+=epochs

    save_weights(gen,args.gen_weights_path,args.gpus)
    save_weights(disc,args.disc_weights_path,args.gpus)
    save_training_params(opt_gen,opt_disc,total_epochs,args.training_params_path)