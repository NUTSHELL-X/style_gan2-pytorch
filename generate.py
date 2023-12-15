import torch
import torch.nn as nn
from model import Generator
from utils import plot_images
import imageio
import os
from utils import config_parser

parser=config_parser()
args=parser.parse_args()
start_channels=args.start_channels
w_channels=args.w_channels
batch_size=8
device=args.device if torch.cuda.is_available() else 'cpu'
dtype = torch.float16 if args.dtype=='float16' else torch.float32
start_res=args.start_res
gen=Generator(start_res=start_res,w_c=w_channels,start_c=start_channels).to(device).to(dtype)
if len(args.gpus) > 1:
    gen.module.load_state_dict(torch.load(args.gen_weights_path)) # 可能还有问题
else:
    gen.load_state_dict(torch.load(args.gen_weights_path))
print('loaded saved weights ; Generator:{}'.format(args.gen_weights_path))
base_tensor=torch.ones((1,start_channels,start_res[0],start_res[1])).to(device).to(dtype)
# w=torch.randn((batch_size,w_c)).to(device)
# images=gen([base_tensor,w],4,1).to('cpu').detach()
# plot_images(images)
x=torch.randn((1,w_channels)).to(device).to(dtype)
y=torch.randn((1,w_channels)).to(device).to(dtype)
print(x,y)
n_frames=200
images=[]
for i in range(n_frames+1):
    a=i/n_frames
    z=a*x+(1-a)*y
    image=gen([base_tensor,z]).squeeze().to('cpu').to(torch.float32).detach().permute(1,2,0).numpy()
    images.append(image)
gif_folder='gif'
if not os.path.exists(gif_folder):
    os.mkdir(gif_folder)
imageio.mimsave(os.path.join(gif_folder,'generated.gif'),images,fps=20)