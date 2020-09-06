import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

def glu(input,gate):
    return torch.mul(input,torch.sigmoid(gate))

class PixelShuffle(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelShuffle, self).__init__()
        # Custom Implementation because PyTorch PixelShuffle requires,
        # 4D input. Whereas, in this case we have have 3D array
        self.upscale_factor = upscale_factor

    def forward(self, input):
        n = input.shape[0]
        c_out = input.shape[1] // 2
        w_new = input.shape[2] * 2
        return input.view(n, c_out, w_new)


class G_Downsample(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(G_Downsample,self).__init__()
        self.pad=nn.ReplicationPad1d((4,0))
        self.conv=nn.Conv1d(in_channels=in_channels,out_channels=out_channels,kernel_size=5,stride=2)
        self.conv_gate=nn.Conv1d(in_channels=in_channels,out_channels=out_channels,kernel_size=5,stride=2)
        self.ins_norm=nn.InstanceNorm1d(out_channels)
        self.ins_gate=nn.InstanceNorm1d(out_channels)
    def forward(self,x):
        x=self.pad(x)
        x_gate=self.conv_gate(x)
        x_gate=self.ins_gate(x_gate)
        x=self.conv(x)
        x=self.ins_norm(x)
        x=glu(x,x_gate)
        return x

class Resblock(nn.Module):
    def __init__(self):
        super(Resblock,self).__init__()
        self.pad=nn.ReplicationPad1d((2,0))
        self.conv1=nn.Conv1d(256,512,kernel_size=3,stride=1)
        self.ins_norm_1=nn.InstanceNorm1d(512)
        self.conv_gate_1=nn.Conv1d(256,512,kernel_size=3,stride=1)
        self.ins_gate_1=nn.InstanceNorm1d(512)
        self.conv2=nn.Conv1d(512,256,kernel_size=3,stride=1)
        self.ins_norm_2=nn.InstanceNorm1d(256)


    def forward(self,x):
        x_0=self.pad(x)
        x_gate=self.conv_gate_1(x_0)
        x_gate=self.ins_gate_1(x_gate)
        x_1=self.conv1(x_0)
        x_1=self.ins_norm_1(x_1)
        y=glu(x_1,x_gate)
        y=self.pad(y)
        y=self.conv2(y)
        y=self.ins_norm_2(y)
        out=torch.add(x,y)
        return out

class Res6block(nn.Module):
    def __init__(self):
        super(Res6block,self).__init__()
        self.res_1=Resblock()
        self.res_2=Resblock()
        self.res_3=Resblock()
        self.res_4=Resblock()
        self.res_5=Resblock()
        self.res_6=Resblock()
        self.res_all=[self.res_1,self.res_2,self.res_3,self.res_4,self.res_5,self.res_6]
    def forward(self,x):
        for res in self.res_all:
            x=res(x)
        return x

class G_Upsample(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(G_Upsample,self).__init__()
        self.pad=nn.ReplicationPad1d((4,0))
        self.conv=nn.Conv1d(in_channels=in_channels,out_channels=out_channels,kernel_size=5,stride=1)
        self.conv_gate=nn.Conv1d(in_channels=in_channels,out_channels=out_channels,kernel_size=5,stride=1)
        self.pix_shuf=PixelShuffle(2)
        self.pix_shuf_gate=PixelShuffle(2)
        self.ins_norm=nn.InstanceNorm1d(out_channels)
        self.ins_gate=nn.InstanceNorm1d(out_channels)
    def forward(self,x):
        x=self.pad(x)
        x1=self.conv(x)
        x1=self.pix_shuf(x1)
        x1=self.ins_norm(x1)
        x2=self.conv_gate(x)
        x2=self.pix_shuf_gate(x2)
        x2=self.ins_gate(x2)
        out=glu(x1,x2)
        return out


class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.pad=nn.ReplicationPad1d((14,0))
        self.conv_first=nn.Conv1d(in_channels=24,out_channels=128,kernel_size=15,stride=1)
        self.conv_first_gate=nn.Conv1d(in_channels=24,out_channels=128,kernel_size=15,stride=1)
        self.down_sample1=G_Downsample(128,256)
        self.down_sample2=G_Downsample(256,256)
        self.res_block=Res6block()
        self.up_sample1=G_Upsample(256,512)
        self.up_sample2=G_Upsample(256,256)
        self.conv_last=nn.Conv1d(in_channels=128,out_channels=24,kernel_size=15,stride=1)

    def forward(self,x):
        x=self.pad(x)
        x_gate=self.conv_first_gate(x)
        x=self.conv_first(x)
        x=glu(x,x_gate)
        x=self.down_sample1(x)
        x=self.down_sample2(x)
        x=self.res_block(x)
        x=self.up_sample1(x)
        x=self.up_sample2(x)
        x=self.pad(x)
        x=self.conv_last(x)
        return x

class D_Downsample(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride):
        super(D_Downsample,self).__init__()
        self.conv=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(kernel_size,3),stride=(stride,2),padding=1)
        self.conv_gate=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(kernel_size,3),stride=(stride,2),padding=1)
        self.ins_norm=nn.InstanceNorm2d(out_channels)
        self.ins_gate=nn.InstanceNorm2d(out_channels)
    def forward(self,x):
        x1=self.conv(x)
        x1=self.ins_norm(x1)
        x2=self.conv_gate(x)
        x2=self.ins_gate(x2)
        out=glu(x1,x2)
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.conv_first=nn.Conv2d(in_channels=1,out_channels=128,kernel_size=3,stride=(1,2),padding=(1,1))
        self.down_sample1=D_Downsample(128,256,3,2)
        self.down_sample2=D_Downsample(256,512,3,2)
        self.down_sample3=D_Downsample(512,1024,6,1)
        self.flatten=nn.Flatten()
        self.linear=nn.Linear(1024*3*8,1)


    def forward(self,x):
        x=self.conv_first(x)
        x=self.down_sample1(x)
        x=self.down_sample2(x)
        x=self.down_sample3(x)
        x=self.flatten(x)
        x=self.linear(x)
        x=torch.sigmoid(x)
        return x
