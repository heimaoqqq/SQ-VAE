import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from networks.util import ResBlock


class EncoderVqResnet256(nn.Module):
    """256×256编码器，遵循原项目风格"""
    def __init__(self, dim_z, cfgs, flg_bn=True, flg_var_q=False):
        super(EncoderVqResnet256, self).__init__()
        self.flg_variance = flg_var_q

        # 原版下采样方式：使用4×4卷积 + stride=2
        # 256×256 → 32×32 需要3次下采样 (256→128→64→32)
        
        layers_conv = []
        # 第一层：3 → dim_z//4, 256→128
        layers_conv.append(nn.Conv2d(3, dim_z // 4, 4, stride=2, padding=1))
        if flg_bn:
            layers_conv.append(nn.BatchNorm2d(dim_z // 4))
        layers_conv.append(nn.ReLU())
        
        # 第二层：dim_z//4 → dim_z//2, 128→64
        layers_conv.append(nn.Conv2d(dim_z // 4, dim_z // 2, 4, stride=2, padding=1))
        if flg_bn:
            layers_conv.append(nn.BatchNorm2d(dim_z // 2))
        layers_conv.append(nn.ReLU())
        
        # 第三层：dim_z//2 → dim_z, 64→32
        layers_conv.append(nn.Conv2d(dim_z // 2, dim_z, 4, stride=2, padding=1))
        if flg_bn:
            layers_conv.append(nn.BatchNorm2d(dim_z))
        layers_conv.append(nn.ReLU())
        
        # 最后一层：保持分辨率
        layers_conv.append(nn.Conv2d(dim_z, dim_z, 3, stride=1, padding=1))
        
        self.conv = nn.Sequential(*layers_conv)
        
        # ResBlocks - 完全原版
        num_rb = cfgs.num_rb
        layers_resblocks = []
        for i in range(num_rb-1):
            layers_resblocks.append(ResBlock(dim_z))
        self.res = nn.Sequential(*layers_resblocks)
        self.res_m = ResBlock(dim_z)
        if self.flg_variance:
            self.res_v = ResBlock(dim_z)

    def forward(self, x):
        out_conv = self.conv(x)
        out_res = self.res(out_conv)
        mu = self.res_m(out_res)
        if self.flg_variance:
            log_var = self.res_v(out_res)
            return mu, log_var
        else:
            return mu


class DecoderVqResnet256(nn.Module):
    """256×256解码器，遵循原项目风格"""
    def __init__(self, dim_z, cfgs, flg_bn=True):
        super(DecoderVqResnet256, self).__init__()
        
        # ResBlocks - 完全原版
        num_rb = cfgs.num_rb
        layers_resblocks = []
        for i in range(num_rb-1):
            layers_resblocks.append(ResBlock(dim_z))
        self.res = nn.Sequential(*layers_resblocks)
        
        # 原版上采样方式：使用转置卷积
        # 32×32 → 256×256 需要3次上采样 (32→64→128→256)
        
        layers_convt = []
        # 第一层：保持分辨率
        layers_convt.append(nn.ConvTranspose2d(dim_z, dim_z, 3, stride=1, padding=1))
        if flg_bn:
            layers_convt.append(nn.BatchNorm2d(dim_z))
        layers_convt.append(nn.ReLU())
        
        # 第二层：dim_z → dim_z//2, 32→64
        layers_convt.append(nn.ConvTranspose2d(dim_z, dim_z // 2, 4, stride=2, padding=1))
        if flg_bn:
            layers_convt.append(nn.BatchNorm2d(dim_z // 2))
        layers_convt.append(nn.ReLU())
        
        # 第三层：dim_z//2 → dim_z//4, 64→128
        layers_convt.append(nn.ConvTranspose2d(dim_z // 2, dim_z // 4, 4, stride=2, padding=1))
        if flg_bn:
            layers_convt.append(nn.BatchNorm2d(dim_z // 4))
        layers_convt.append(nn.ReLU())
        
        # 第四层：dim_z//4 → 3, 128→256
        layers_convt.append(nn.ConvTranspose2d(dim_z // 4, 3, 4, stride=2, padding=1))
        layers_convt.append(nn.Sigmoid())
        
        self.convt = nn.Sequential(*layers_convt)

    def forward(self, z):
        out_res = self.res(z)
        out = self.convt(out_res)
        return out
