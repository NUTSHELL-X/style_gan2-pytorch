import torch
import torch.nn as nn
from model import Generator
from utils import plot_images
import imageio
import os
from utils import config_parser

parser=config_parser()
args=parser.parse_args()
start_c=args.start_c
w_c=args.w_c
batch_size=12
device='cuda' if torch.cuda.is_available() else 'cpu'
start_res=args.start_res
gen=Generator(start_res=start_res,w_c=w_c,start_c=start_c).to(device)
model_path=args.model_path
checkpoint=torch.load(model_path)
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
gif_folder='gif'
if not os.path.exists(gif_folder):
    os.mkdir(gif_folder)
imageio.mimsave(os.path.join(gif_folder,'generated_mc_landscape.gif'),images,fps=20)