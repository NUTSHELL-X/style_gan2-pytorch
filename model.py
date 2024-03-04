import torch
import torch.nn as nn
import numpy as np
from options import config_parser
from utils import print_networks
# from torch.nn.parallel import DistributedDataParallel

parser=config_parser()
args=parser.parse_args()
device=args.device if torch.cuda.is_available() else "cpu"
def fade_in(alpha,a,b):
    return alpha*a+(1.0-alpha)*b

def pixel_norm(x,dim,epsilon=1e-8):
    return x/torch.sqrt(torch.mean(x**2,dim=dim,keepdim=True)+epsilon)

class EqualizedConv(nn.Module):
    def __init__(self,in_c,out_c,kernel_size=3,stride=1,padding=1,gain=2,bias=False):
        super().__init__()
        self.conv=nn.Conv2d(in_c,out_c,kernel_size,stride,padding,bias=bias)
        self.scale=(gain/(in_c*kernel_size**2))**0.5
        
    def forward(self,x):
        return self.conv(x*self.scale)

class EqualizedDense(nn.Module):
    def __init__(self,in_units,out_units,gain=2,learning_rate_multiplier=1,**kwards):
        super().__init__()
        self.in_units=in_units
        self.out_units=out_units
        self.gain=gain
        self.learning_rate_multiplier=learning_rate_multiplier
        self.dense=nn.Linear(in_units,out_units)
        self.scale=np.sqrt(self.gain/self.in_units)
        nn.init.normal_(self.dense.weight,mean=0.0,std=1.0/self.learning_rate_multiplier*self.scale)
        nn.init.zeros_(self.dense.bias)

    def forward(self,x):
        outputs=self.dense(x)
        return outputs*self.learning_rate_multiplier

class AddNoise(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.conv=nn.Conv2d(c,c,kernel_size=1,groups=c,bias=False)
        self.c=c
        # self.b=torch.randn((1,c,1,1),requires_grad=True)

    def forward(self,inputs):
        x,noise=inputs
        # print('x,noise:',x.device,noise.device)
        noise=noise.tile((1,self.c,1,1))
        outputs=x+self.conv(noise)
        return outputs

class AddBias(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.b=nn.Parameter(torch.zeros(1,c,1,1,requires_grad=True))
    
    def forward(self,x):
        return x+self.b

class AdaIN(nn.Module):     
    def __init__(self,x_c,w_c,gain=1,**kwargs):
        super().__init__()
        self.x_c=x_c
        self.dense_0=EqualizedDense(w_c,x_c,gain=gain)
        self.dense_1=EqualizedDense(w_c,x_c,gain=gain)

    def forward(self,inputs):
        x,w=inputs
        ys=self.dense_0(w).reshape(-1,self.x_c,1,1)
        yb=self.dense_1(w).reshape(-1,self.x_c,1,1)
        return ys*x+yb

class Mapping(nn.Module):
    def __init__(self,z_dim=512):
        super().__init__()
        self.z_dim=z_dim
        self.net=nn.ModuleList()
        for i in range(8):
            self.net.append(EqualizedDense(z_dim,z_dim,learning_rate_multiplier=0.01))
            self.net.append(nn.LeakyReLU(0.2))

    def forward(self,x):
        w=pixel_norm(x,-1)
        for layer in self.net:
            w=layer(w)
        return w

class G_block(nn.Module):
    def __init__(self,in_c,out_c,w_c,need_upsample=False):
        super().__init__()
        self.in_c=in_c
        self.out_c=out_c
        self.w_c=w_c
        self.need_upsample=need_upsample
        self.upsample=nn.Upsample(scale_factor=2)
        self.eq_conv0=EqualizedConv(in_c,out_c,kernel_size=3,bias=True)
        # self.add_bias=AddBias(out_c)
        self.add_noise0=AddNoise(out_c)
        self.leaky0=nn.LeakyReLU(0.2)
        self.ins_norm0=nn.InstanceNorm2d(out_c)
        self.ada_in0=AdaIN(in_c,w_c)

    def forward(self,inputs):
        x,w,noise=inputs
        x=self.ins_norm0(x)
        x=self.ada_in0([x,w])
        if self.need_upsample:
            x=self.upsample(x)
        
        x=self.eq_conv0(x)
        # x=self.add_bias(x)
        x=self.add_noise0([x,noise])
        x=self.leaky0(x)
        return x

class Generator(nn.Module):
    def __init__(self,start_res=(4,4),w_c=512,start_c=512,steps=6):
        super().__init__()
        self.g_blocks=nn.ModuleList()
        self.to_rgbs=nn.ModuleList()
        self.up_samples=nn.ModuleList()
        self.n_channels={
            0:start_c,
            1:start_c,
            2:start_c,
            3:start_c,
            4:int(start_c/2),
            5:int(start_c/4),
            6:int(start_c/8),
        }
        self.start_res=start_res
        self.w_c=w_c
        self.steps=steps
        self.map_net=Mapping(z_dim=w_c)
        self.sigmoid=nn.Sigmoid()
        for i in range(steps+1):
            c=self.n_channels[i]
            to_rgb=EqualizedConv(c,3,kernel_size=1,padding=0)
            self.to_rgbs.append(to_rgb)
            if i>=1:
                self.g_blocks.append(G_block(self.n_channels[i-1],c,w_c,need_upsample=True))
            self.g_blocks.append(G_block(c,c,w_c,need_upsample=False))
        for i in range(steps):
            self.up_samples.append(nn.Upsample(scale_factor=2))
       
    def forward(self,inputs):
        x,w=inputs
        w=self.map_net(w)
        rgb_out=None
        bc=x.shape[0]
        for i in range(self.steps+1):
            height=self.start_res[0]*2**i
            width=self.start_res[1]*2**i
            if i==0:
                noise=torch.randn((bc,1,height,width)).to(x.device).to(x.dtype)
                x=self.g_blocks[0]([x,w,noise])
                rgb_out=self.to_rgbs[0](x)
            else:
                noise=torch.randn((bc,1,height,width)).to(x.device).to(x.dtype)
                x=self.g_blocks[2*i-1]([x,w,noise])
                noise=torch.randn((bc,1,height,width)).to(x.device).to(x.dtype)
                x=self.g_blocks[2*i]([x,w,noise])
                rgb_out=self.up_samples[i-1](rgb_out)
                rgb_out+=self.to_rgbs[i](x)

        return self.sigmoid(rgb_out)

class D_block(nn.Module):
    def __init__(self,in_c,out_c):
        super().__init__()
        self.down_sample0=nn.AvgPool2d(2)
        self.down_sample1=nn.AvgPool2d(2)
        self.eq_conv=EqualizedConv(in_c,out_c,bias=True)
        self.conv=nn.Conv2d(in_c,out_c,kernel_size=3,padding=1)
        self.bn=nn.BatchNorm2d(out_c)
        self.leaky=nn.LeakyReLU(0.2)

    def forward(self,x):
        x_skip=self.down_sample0(x)
        x_skip=self.conv(x_skip)
        x=self.eq_conv(x)
        x=self.down_sample1(x)
        x=x+x_skip
        x=self.bn(x)
        x=self.leaky(x)
        return x

class Discriminator(nn.Module):
    def __init__(self,start_res=(4,4),start_c=512,steps=6):
        super().__init__()
        self.start_res=start_res
        self.steps=steps
        self.d_blocks=nn.ModuleList()
        self.n_channels={
            0:start_c,
            1:start_c,
            2:start_c,
            3:start_c,
            4:int(start_c/2),
            5:int(start_c/4),
            6:int(start_c/8),
        }
        self.from_rgb=EqualizedConv(in_c=3,out_c=self.n_channels[steps])
        self.flatten=nn.Flatten()
        self.dense0=nn.Linear(start_res[0]*start_res[1]*self.n_channels[0],512)
        self.leaky=nn.LeakyReLU(0.2)
        self.dense1=nn.Linear(512,1)
        self.sigmoid=nn.Sigmoid()
        for i in range(steps,0,-1):
            self.d_blocks.append(D_block(self.n_channels[i],self.n_channels[i-1]))

    def forward(self,x):
        x=self.from_rgb(x)
        for i in range(self.steps):
            x=self.d_blocks[i](x)
        x=self.flatten(x)
        x=self.dense0(x)
        x=self.leaky(x)
        x=self.dense1(x)
        x=self.sigmoid(x)
        return x

if __name__=='__main__':
    device='cuda'
    start_res=[4,4]
    batch_size=2
    start_c=512
    w_c=512
    dtype = torch.float16 if args.dtype=='float16' else torch.float32
    x=torch.randn((batch_size,start_c,start_res[0],start_res[1])).to(device).to(dtype)
    w=torch.randn((batch_size,w_c)).to(device).to(dtype)
    gen=Generator(start_res=start_res,start_c=start_c,steps=4).to(device)
    gen.to(dtype)
    gen=torch.nn.DataParallel(gen,device_ids=args.gpus)
    print_networks(gen)
    out=gen([x,w])
    print('out shape:',out.shape,out.device)
    disc=Discriminator(start_res=start_res,start_c=start_c,steps=4).to(device).to(dtype)
    print_networks(disc)
    disc_out=disc(out)
    print(disc_out)
    