import torch
import torch.nn as nn
from model import Generator
from utils import plot_images
import imageio

start_c=256
w_c=512
batch_size=12
device='cuda' if torch.cuda.is_available() else 'cpu'
start_res=(5,8)
gen=Generator(start_res=start_res,start_c=start_c).to(device)
checkpoint=torch.load('MC_saved.pt')
gen.load_state_dict(checkpoint['gen_state_dict'])
base_tensor=torch.ones((1,start_c,start_res[0],start_res[1])).to(device)
# w=torch.randn((batch_size,w_c)).to(device)
# images=gen([base_tensor,w],4,1).to('cpu').detach()
# plot_images(images)
x=torch.randn((1,w_c)).to(device)
y=torch.randn((1,w_c)).to(device)
n_frames=200
images=[]
for i in range(n_frames+1):
    a=i/n_frames
    z=a*x+(1-a)*y
    image=gen([base_tensor,z]).squeeze().to('cpu').detach().permute(1,2,0).numpy()
    print(image.shape)
    images.append(image)

imageio.mimsave('gif/generated_mc_landscape.gif',images,fps=20)