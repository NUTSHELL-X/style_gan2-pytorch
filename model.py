import torch
import torch.nn as nn
import numpy as np

device='cuda' if torch.cuda.is_available() else "cpu"
def fade_in(alpha,a,b):
    return alpha*a+(1.0-alpha)*b

def pixel_norm(x,dim,epsilon=1e-8):
    return x/torch.sqrt(torch.mean(x**2,dim=dim,keepdim=True)+epsilon)

class EqualizedConv(nn.Module):
    def __init__(self,in_c,out_c,kernel_size=3,stride=1,padding=1,gain=2):
        super().__init__()
        self.conv=nn.Conv2d(in_c,out_c,kernel_size,stride,padding)
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
    def __init__(self,in_c,out_c,w_c,is_base):
        super().__init__()
        self.in_c=in_c
        self.out_c=out_c
        self.w_c=w_c
        self.is_base=is_base
        self.upsample=nn.Upsample(scale_factor=2)
        self.eq_conv0=EqualizedConv(in_c,out_c,kernel_size=3)
        self.add_noise0=AddNoise(out_c)
        self.leaky0=nn.LeakyReLU(0.2)
        self.ins_norm0=nn.InstanceNorm2d(out_c)
        self.ada_in0=AdaIN(out_c,w_c)
        self.eq_conv1=EqualizedConv(out_c,out_c)
        self.add_noise1=AddNoise(out_c)
        self.leaky1=nn.LeakyReLU(0.2)
        self.ins_norm1=nn.InstanceNorm2d(out_c)
        self.ada_in1=AdaIN(out_c,w_c)

    def forward(self,inputs):
        x,w,noise=inputs
        if not self.is_base:
            x=self.upsample(x)
            x=self.eq_conv0(x)
        x=self.add_noise0([x,noise])
        x=self.leaky0(x)
        x=self.ins_norm0(x)
        x=self.ada_in0([x,w])

        x=self.eq_conv1(x)
        x=self.add_noise1([x,noise])
        x=self.leaky1(x)
        x=self.ins_norm1(x)
        x=self.ada_in1([x,w])

        return x

class Generator(nn.Module):
    def __init__(self,start_res=(4,4),w_c=512,start_c=512,max_steps=6):
        super().__init__()
        self.g_blocks=nn.ModuleList()
        self.to_rgbs=nn.ModuleList()
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
        self.map_net=Mapping(z_dim=w_c)
        for i in range(max_steps+1):
            c=self.n_channels[i]
            to_rgb=EqualizedConv(c,3,kernel_size=1,padding=0)
            self.to_rgbs.append(to_rgb)
            is_base=i==0
            self.g_blocks.append(G_block(self.n_channels[i-1] if i!=0 else start_c,c,w_c,is_base))

    def forward(self,inputs,step,alpha):
        x,w=inputs
        w=self.map_net(w)
        if step==0:
            height=self.start_res[0]
            width=self.start_res[1]
            noise=torch.randn((1,1,height,width)).to(device)
            x=self.g_blocks[0]([x,w,noise])
            return self.to_rgbs[0](x)
        else:
            # print('x,step,alpha:',x.shape,step,alpha)
            for i in range(step):
                height=self.start_res[0]*2**i
                width=self.start_res[1]*2**i
                noise=torch.randn((1,1,height,width)).to(device)
                x=self.g_blocks[i]([x,w,noise])

            old_rgb=self.to_rgbs[step-1](x)
            old_rgb=nn.Upsample(scale_factor=2)(old_rgb)
            height=self.start_res[0]*2**step
            width=self.start_res[1]*2**step
            noise=torch.randn((1,1,height,width)).to(device)
            x=self.g_blocks[step]([x,w,noise])
            new_rgb=self.to_rgbs[step](x)
            return fade_in(alpha,new_rgb,old_rgb)

class ResBlock(nn.Module):
    def __init__(self,in_c,out_c,pool_factor=2):
        super(ResBlock,self).__init__()
        self.pool_factor=pool_factor
        self.conv0=EqualizedConv(in_c,out_c,3,padding='same')
        self.leaky_relu0=nn.LeakyReLU(0.2)
        self.conv1=EqualizedConv(out_c,out_c,3,padding='same')
        self.leaky_relu1=nn.LeakyReLU(0.2)
        self.conv2=EqualizedConv(in_c,out_c,1,padding='same')
        self.ins_norm=nn.InstanceNorm2d(out_c)
        self.avg_pool=nn.AvgPool2d(pool_factor)

    def forward(self,x):
        x_skip=self.conv2(x)
        x=self.conv0(x)
        x=self.leaky_relu0(x)
        x=self.conv1(x)
        x=self.leaky_relu1(x)
        x=x+x_skip
        x=self.ins_norm(x)
        x=self.avg_pool(x)
        return x

# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator,self).__init__()
#         self.net=nn.Sequential(
#             ResBlock(3,64),
#             ResBlock(64,128),
#             ResBlock(128,128),
#             ResBlock(128,256),
#             ResBlock(256,256),
#             nn.Flatten(),
#             nn.Linear(5*8*256,512),
#             nn.LeakyReLU(0.1),
#             nn.Dropout(0.1),
#             nn.Linear(512,1),
#             # nn.Sigmoid(),
#         )

#     def forward(self,x):
#         return self.net(x)

class D_block_base(nn.Module):
    def __init__(self,in_c,out_c,hidden_units):
        super().__init__()
        self.eq_conv=EqualizedConv(in_c+1,out_c)
        self.leaky0=nn.LeakyReLU(0.2)
        self.flatten=nn.Flatten()
        self.eq_dense0=EqualizedDense(hidden_units,out_c)
        self.leaky1=nn.LeakyReLU(0.2)
        self.eq_dense1=EqualizedDense(out_c,1)

    def minibatch_std(self,x):
        batch_statistics=x.std(dim=0).mean().repeat(x.shape[0],1,x.shape[2],x.shape[3])
        return torch.cat([x,batch_statistics],dim=1)

    def forward(self,x):
        x=self.minibatch_std(x)
        x=self.eq_conv(x)
        x=self.leaky0(x)
        x=self.flatten(x)
        x=self.eq_dense0(x)
        x=self.leaky1(x)
        x=self.eq_dense1(x)

        return x

class Discriminator(nn.Module):
    def __init__(self,start_res=(4,4),start_c=512,max_step=6):
        super().__init__()
        self.start_res=start_res
        self.d_blocks=nn.ModuleList()
        self.from_rgbs=nn.ModuleList()
        self.n_channels={
            0:start_c,
            1:start_c,
            2:start_c,
            3:start_c,
            4:int(start_c/2),
            5:int(start_c/4),
            6:int(start_c/8),
        }

        for i in range(max_step+1):
            from_rgb=EqualizedConv(3,self.n_channels[i],kernel_size=1,padding=0)
            self.from_rgbs.append(from_rgb)
            if i==0:
                self.d_blocks.append(D_block_base(start_c,start_c,start_res[0]*start_res[1]*start_c))
            else:
                self.d_blocks.append(ResBlock(self.n_channels[i],self.n_channels[i-1]))

    def forward(self,x,step,alpha):
        if step==0:
            x=self.from_rgbs[0](x)
            x=self.d_blocks[0](x)
            return x
        else:
            downsized_image=nn.AvgPool2d(2)(x)
            x0=self.from_rgbs[step-1](downsized_image)
            x1=self.from_rgbs[step](x)
            x1=self.d_blocks[step](x1)
            # print('x0,x1:',x0.shape,x1.shape)
            x=fade_in(alpha,x1,x0)
            for i in range(step-1,-1,-1):
                x=self.d_blocks[i](x)
            return x

if __name__=='__main__':
    device='cuda'
    start_res=[5,8]
    batch_size=16
    start_c=256
    w_c=512
    x=torch.randn((batch_size,start_c,start_res[0],start_res[1])).to(device)
    w=torch.randn((batch_size,w_c)).to(device)
    gen=Generator(start_res=start_res,start_c=start_c).to(device)
    out=gen([x,w],0,0)
    print('out shape:',out.shape)
    disc=Discriminator(start_res=start_res,start_c=start_c).to(device)
    disc_out=disc(out,0,0)
    print(disc_out)
    