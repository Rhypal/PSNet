# import sys
# sys.path.append('/path/to/your/module')
import torch
from matplotlib import pyplot as plt
import numpy as np
from model.resnet_model import *
from torch.cuda.amp import autocast
from torch import nn
from torch.nn import functional as F

from mamba_ssm import Mamba
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat
import pdb
import os
# import rasterio
from timm.models.layers import trunc_normal_, DropPath

#full image decode'output guide patch image encode

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
torch.cuda.empty_cache()

class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.2))

class ConvBN1(nn.Sequential):
    def __init__(self, in_channels, out_channels, bias=False):
        super(ConvBN1, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())


class ConvDown(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ConvDown, self).__init__(
            # # nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            # # nn.BatchNorm2d(out_channels),
            # # nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(), nn.Dropout(0.2),
        )

class ConvUp(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ConvUp, self).__init__(
            # # nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            # # nn.BatchNorm2d(out_channels), nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels), nn.ReLU(),
            # nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels), nn.ReLU(), nn.Dropout(0.2),
        )


class Multi_Attention(nn.Module):
    def __init__(self):
        super(Multi_Attention, self).__init__()

        # 定义模块中用到的各个层
        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=9, dilation=d, padding=p)
            for (d, p) in zip([1, 2, 4, 8], [4, 8, 16, 32])
        ])
        self.relu = nn.ReLU()
        self.linear = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 连接整合输入特征图
        map1 = torch.cat(x, dim=1)
        # 进行2D全局池化
        global_pool = nn.functional.adaptive_avg_pool2d(map1, (1, 1))
        # 进行Transpose和Squeeze
        transposed = global_pool.permute(0, 2, 3, 1).squeeze(dim=1)
        # 进行一维卷积操作
        conv1d_outputs = [conv(transposed) for conv in self.conv1d_list]
        conv1d_results = torch.cat(conv1d_outputs, dim=1)

        # 进行转置操作
        transposed_conv = conv1d_results.permute(0, 2, 1)

        # 进行ReLU操作
        relu_output = self.relu(transposed_conv)

        # 进行Linear操作
        linear_output = self.linear(relu_output)

        # 进行Sigmoid操作
        sigmoid_output = self.sigmoid(linear_output)

        # 进行unsqueeze操作
        unsqueezed = sigmoid_output.unsqueeze(-1)

        output = unsqueezed * map1

        return output

class CascadeFusion (nn.Module):
    def __init__(self, in_channels1, in_channels2, in_channels3):
        super(CascadeFusion, self).__init__()

        # 定义各个操作所需要的模块
        self.conv1_map1 = nn.Conv2d(in_channels=in_channels1, out_channels=128, kernel_size=3, padding=2, dilation=2)
        self.bn1_map1 = nn.BatchNorm2d(128)
        self.relu1_map1 = nn.ReLU(inplace=True)
        self.conv1_map2 = nn.Conv2d(in_channels=in_channels2, out_channels=128, kernel_size=3, padding=1)
        self.bn1_map2 = nn.BatchNorm2d(128)
        self.relu1_map2 = nn.ReLU(inplace=True)
        self.conv1_map3 = nn.Conv2d(in_channels=in_channels3, out_channels=128, kernel_size=3, padding=1)
        self.bn1_map3 = nn.BatchNorm2d(128)
        self.relu1_map3 = nn.ReLU(inplace=True)

        self.conv_concat = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=5, padding=4, dilation=2)
        self.bn_concat = nn.BatchNorm2d(256)
        self.relu_concat = nn.ReLU(inplace=True)

        self.conv_final = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.bn_final = nn.BatchNorm2d(128)
        self.relu_final = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, map1, map2, map3):
        # in_channels = map2.shape[1]
        # 对 map1 的操作
        # map1 = nn.functional.interpolate(map1, scale_factor=2, mode='bilinear', align_corners=False)
        x_low = self.relu1_map1(self.bn1_map1(self.conv1_map1(map1)))
        # 对 map2 的操作
        x_high = self.relu1_map2(self.bn1_map2(self.conv1_map2(map2)))

        x_high3 = self.relu1_map3(self.bn1_map3(self.conv1_map3(map3)))
        # 执行concat操作
        concat_features = torch.cat((x_low, x_high), dim=1)
        concat_features3 = torch.cat((concat_features, x_high3), dim=1)
        concat_output = self.relu_concat(self.bn_concat(self.conv_concat(concat_features3)))

        # 使用Sigmoid获得x_ga
        x_ga = self.sigmoid(self.relu_final(self.bn_final(self.conv_final(concat_output))))
        # 分支1：x_ga与x_low做Hadamard积得到output10
        output1 = x_ga * x_low
        # 分支2：(1 - x_ga)与x_high做Hadamard积得到output2
        output2 = (1 - x_ga) * x_high

        # 将output1和output2相加得到模块的结果
        module_output = output1 + output2
        module_output = F.interpolate(module_output, scale_factor=2, mode='bilinear', align_corners=False)

        return module_output

class DiffNet(nn.Module):
    def __init__(self, in_ch, n_classes):
        super(DiffNet, self).__init__()

        self.conv0 = nn.Conv2d(in_ch, 64, 3, padding=1)

        self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1 = SynchronizedBatchNorm2d(64)
        self.relu1 = nn.ReLU()

        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = SynchronizedBatchNorm2d(64)
        self.relu2 = nn.ReLU()

        self.pool2 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = SynchronizedBatchNorm2d(64)
        self.relu3 = nn.ReLU()

        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = SynchronizedBatchNorm2d(64)
        self.relu4 = nn.ReLU()

        self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=True)


        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn5 = SynchronizedBatchNorm2d(64)
        self.relu5 = nn.ReLU()

        self.conv_d4 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d4 = SynchronizedBatchNorm2d(64)
        self.relu_d4 = nn.ReLU()

        self.conv_d3 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d3 = SynchronizedBatchNorm2d(64)
        self.relu_d3 = nn.ReLU()

        self.conv_d2 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d2 = SynchronizedBatchNorm2d(64)
        self.relu_d2 = nn.ReLU()

        self.conv_d1 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d1 = SynchronizedBatchNorm2d(64)
        self.relu_d1 = nn.ReLU()

        self.conv_d0 = nn.Conv2d(64, n_classes, 3, padding=1) #2
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False)


    def forward(self,x):
        # x = Resize([4, 1, 320, 320])
        hx = x
        # print('x:', hx.shape)
        hx = self.conv0(hx)
        # print('conv0:', hx.shape)
        hx1 = self.relu1(self.bn1(self.conv1(hx)))
        # print('relu:', hx1.shape)
        hx = self.pool1(hx1)
        # print('pool1:', hx.shape)

        hx2 = self.relu2(self.bn2(self.conv2(hx)))
        # print('conv2:', hx2.shape)
        hx = self.pool2(hx2)
        # print('pool2:', hx.shape)

        hx3 = self.relu3(self.bn3(self.conv3(hx)))
        # print('conv3', hx3.shape)
        hx = self.pool3(hx3)
        # print('pool3:', hx.shape)

        hx4 = self.relu4(self.bn4(self.conv4(hx)))
        # print('conv4:', hx4.shape)
        hx = self.pool4(hx4)
        # print('pool4:', hx.shape)

        hx5 = self.relu5(self.bn5(self.conv5(hx)))
        # print('conv5:', hx5.shape)
        hx = self.upscore2(hx5)
        # print('upscore2:', hx.shape)

        d4 = self.relu_d4(self.bn_d4(self.conv_d4(torch.cat((hx,hx4),1))))
        # print('conv_d4:', d4.shape)
        hx = self.upscore2(d4)
        # print('upscore:', hx.shape)

        d3 = self.relu_d3(self.bn_d3(self.conv_d3(torch.cat((hx,hx3),1))))
        # print('conv_d3:', d3.shape)
        hx = self.upscore2(d3)
        # print('upscore:', hx.shape)
        d2 = self.relu_d2(self.bn_d2(self.conv_d2(torch.cat((hx,hx2),1))))
        # print('conv_d2:', d2.shape)
        hx = self.upscore2(d2)
        # print('upscore:', hx.shape)

        d1 = self.relu_d1(self.bn_d1(self.conv_d1(torch.cat((hx,hx1),1))))
        # print('conv_d1:', d1.shape)
        #这个卷积报错了，试试用d2 d2可能放进 cov_d0(d2)

        x = self.conv_d0(d1)
        # print('conv_d0:', residual.shape)
        # x = x+residual
        return x

class Fusion_basic(nn.Module):
    def __init__(self):
        super(Fusion_basic, self).__init__()
        self.conv = nn.Conv2d(1,1,1)

    def find_coordination(self, l_map, mode=1):# l_map：(b,1,384,384)
        # 暂时没有在batch维上批量处理的方法，目前以单batch设计,尝试中！
        non_zero_indices = np.argwhere(l_map[0] != 0) # 找到非0的地方
        new_shape = non_zero_indices[-1] - non_zero_indices[0] + 1   # 大图中对应切片的width 和 length
        # print(1, non_zero_indices,non_zero_indices[-1],non_zero_indices[0], new_shape)
        min_cor = np.min(non_zero_indices,axis=0) # 左上角坐标，用于patch索引
        if mode != 1:
            min_cor = (min_cor - 0.5 * (mode // 2) * new_shape).astype(int)  # 保持整数！
            for i in range(2):
                if min_cor[i]<0:
                    min_cor[i]=0
            new_shape = mode * new_shape

        for i in range(2):
            if new_shape[i] + min_cor[i] > l_map.shape[2]:
                new_shape[i] =  l_map.shape[2] - min_cor[i]

        return new_shape, min_cor

    def create_new(self, f_b_b, local_map_b, mode=1): # 仅在一个batchsize上进行，用于简化操作
        c, h, w = f_b_b.shape
        c0, h0, w0 = local_map_b.shape # 暂不进行resize
        assert h0 == h , "mask尺寸不匹配"

        # 创建新的图像
        new_shape, min_index = self.find_coordination(local_map_b,mode=mode)
        # print(new_shape,min_index)
        new_img = np.zeros((c,new_shape[0],new_shape[1]))
        for i in range(c):
            new_img[i] = f_b_b[i][min_index[0]:min_index[0] + new_shape[0], min_index[1]:min_index[1] + new_shape[1]]
        new_img = torch.from_numpy(new_img).unsqueeze(0)
        # return new_img
        return F.interpolate(new_img, (h,w), mode='bilinear')  # 此时的new_img 的形状与位置掩码中的patch大小相同，需要进行双线性插值映射到f_s的尺度

    def batch_new(self, f_b, local_map,mode=1):
        f_b = f_b.detach().cpu().numpy()
        local_map = local_map.detach().cpu().numpy()

        batch_new_img = torch.zeros(f_b.shape)
        for i in range(f_b.shape[0]):  # batchsize 的索引应该与numpy数组相同，从0开始
            batch_new_img[i] = self.create_new(f_b[i], local_map[i],mode)[0]

        return batch_new_img.cuda()

    def forward(self, f_b, local_map):  # 如果local_map 需要进行resize，应该采用最近邻插值
        # f_s : patch feature map; f_b : global feature map; local_map : location information
        # 特征融合过程以 patch feature map 为基准
        b,c,h,w = f_b.shape  # b 和 c 需要用于辅助特征获取
        # 特征图尺度对齐采用 双线性插值（参考现有大尺度语义分割设计）
        local_map = F.interpolate(local_map, size=(h,w), mode='bilinear', align_corners=False)

        # feature_min 为固定patch的局部特征
        # f_local = f_b * local_map  # 张量操作与numpy应该没有太大区别，无法直接提取，需要通过坐标索引
        # interpolation does not change the shape of tensor
        feature_min = F.interpolate(self.batch_new(f_b,local_map,mode=1),(h,w), mode='bilinear')
        return feature_min


class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class AttentionPooling(nn.Module):
    def __init__(self, in_channels):
        super(AttentionPooling, self).__init__()
        self.attn_conv = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        attn_map = self.attn_conv(x)
        attn_map = F.softmax(attn_map.view(x.size(0), -1), dim=-1)
        attn_map = attn_map.view(x.size(0), 1, x.size(2), x.size(3))
        weighted_x = x * attn_map
        return weighted_x, attn_map

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates= [4, 8, 12]):
        super(ASPP, self).__init__()

        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ))
        for rate in atrous_rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()))
        # self.global_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
        #                                  nn.Conv2d(in_channels, out_channels, 1, bias=False),
        #                                  nn.BatchNorm2d(out_channels),
        #                                  nn.ReLU())
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(modules) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        # pool = self.global_pool(x)
        # pool = F.interpolate(pool, size=x.shape[2:], mode='bilinear', align_corners=False)
        # res.append(pool)
        res = torch.cat(res, dim=1)
        return self.project(res)



class CrissCrossAttention(nn.Module):   # Criss-Cross Attention Module
    def __init__(self, in_dim):
        super(CrissCrossAttention,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.ones(1))
        self.topk = True

    def INF(self, B, H, W):
        return -torch.diag(torch.tensor(float("inf")).to('cuda:1').repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)

    def forward(self, x1, x2):
        m_batchsize, _, height, width = x1.size()
        proj_query = self.query_conv(x1)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        proj_key = self.key_conv(x2)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        proj_value = self.value_conv(x2)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        if self.topk:
            topk_values_h, topk_indices_h = torch.topk(energy_H, k=height // 2, dim=-1)
            mask_h = torch.zeros_like(energy_H, dtype=torch.float32)
            mask_h.scatter(-1, topk_indices_h, 1.0)
            energy_H = torch.where(mask_h > 0, energy_H, torch.full_like(energy_H, float('inf')))
            topk_values_w, topk_indices_w = torch.topk(energy_W, k=width // 2, dim=-1)
            mask_w = torch.zeros_like(energy_W, dtype=torch.float32)
            mask_w.scatter(-1, topk_indices_w, 1.0)
            energy_W = torch.where(mask_w > 0, energy_W, torch.full_like(energy_W, float('inf')))

        concate = self.softmax(torch.cat([energy_H, energy_W], -1))
        del proj_query, proj_query_H, proj_query_W, proj_key, proj_key_H, proj_key_W, proj_value, energy_H, energy_W
        torch.cuda.empty_cache()

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        return self.gamma*(out_H + out_W) + x1


class CrossAttention(nn.Module):
    def __init__(self, in_dim):
        super(CrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.ones(1))
        # self.bn1 = nn.BatchNorm2d(in_dim)
        # self.bn2 = nn.BatchNorm2d(in_dim)
        self.visualize = True
        self.att_map = None
        self.conv1 = nn.Sequential(nn.Conv2d(in_dim * 2, in_dim * 2, 3, padding=1, groups=in_dim),
                                   nn.Conv2d(in_dim * 2, in_dim, 1, padding=0, groups=1),
                                   nn.BatchNorm2d(in_dim), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_dim * 2, in_dim * 2, 3, padding=1, groups=in_dim),
                                   nn.Conv2d(in_dim * 2, in_dim, 1, padding=0, groups=1),
                                   nn.BatchNorm2d(in_dim), nn.ReLU())

    def forward(self, x1, x2):
        batchsize, C, height, width = x1.size()
        sub_x = x1 - x2
        concat1 = torch.cat([sub_x, x1], dim=1)
        concat2 = torch.cat([sub_x, x2], dim=1)
        x3 = self.conv1(concat1)
        x4 = self.conv2(concat2)
        proj_query = self.query_conv(x4)
        proj_key = self.key_conv(x3)
        proj_value = self.value_conv(x3)
        proj_query = proj_query.view(batchsize, -1, height * width).permute(0, 2, 1).contiguous()
        proj_key = proj_key.view(batchsize, -1, height * width).contiguous()
        proj_value = proj_value.view(batchsize, -1, height * width).contiguous()     # [B, C, H*W]
        energy = torch.bmm(proj_query, proj_key)        # [B, H*W, H*W]
        del proj_query, proj_key, concat1, concat2
        torch.cuda.empty_cache()

        # # topk
        # topk_values, topk_indices = torch.topk(energy, k=1500, dim=2)
        # mask = torch.zeros_like(energy, dtype=torch.float16)
        # mask.scatter_(2, topk_indices, 1.0)
        # energy = torch.where(mask > 0, energy, torch.full_like(energy, float('-inf')))

        attn = self.softmax(energy)     # [2, 2304, 2304]

        if self.visualize:
            self.att_map = attn.view(batchsize, height, width, height, width)

        out = torch.bmm(proj_value, attn.permute(0, 2, 1))
        out = out.view(batchsize, -1, height, width)
        return self.gamma * out + x1


class LSConv(nn.Module):
    def __init__(self, dim, lks=7, sks=3, groups=8):
        super(LSConv, self).__init__()
        self.sks = sks
        self.groups = groups
        self.dim = dim

        self.cv1 = nn.Sequential(nn.Conv2d(dim, dim // 2, 1), nn.BatchNorm2d(dim // 2))
        self.act = nn.ReLU()
        self.cv2 = nn.Sequential(
            nn.Conv2d(dim // 2, dim // 2, kernel_size=lks, padding=(lks - 1) // 2, groups=dim // 2),
            nn.BatchNorm2d(dim // 2))
        self.cv3 = nn.Sequential(nn.Conv2d(dim // 2, dim // 2, 1), nn.BatchNorm2d(dim // 2))
        self.cv4 = nn.Conv2d(dim // 2, sks ** 2 * dim // groups, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=dim // groups, num_channels=sks ** 2 * dim // groups)
        self.bn = nn.BatchNorm2d(dim)

    def LKP(self, z):
        z = self.act(self.cv3(self.cv2(self.act(self.cv1(z)))))
        z = self.norm(self.cv4(z))
        b, _, h, width = z.size()
        z = z.view(b, self.dim // self.groups, self.sks ** 2, h, width)
        return z

    def SKA(self, x, w):
        b, c, h, width = x.size()
        x = F.unfold(x, kernel_size=self.sks, padding=self.sks // 2, stride=1)
        x = x.view(b, self.groups, c // self.groups, self.sks ** 2, h, width)
        w = w.softmax(dim=2)
        x = (x * w.unsqueeze(1)).sum(dim=3)
        x = x.view(b, c, h, width)
        return x

    def forward(self, x):
        w = self.LKP(x)
        y = self.SKA(x, w)
        y = self.bn(y) + x
        return y


class ARM(torch.nn.Module):        #Asymmetrical Refinement Module
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(in_channels)
        self.lsconv = LSConv(in_channels)
        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1,3), padding=(0,1))
        self.conv4 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3,1), padding=(1,0))

    def forward(self, x0):
        x0 = self.bn(x0)
        x1 = self.lsconv(x0)     # [2, 256, 48, 48]
        x2 = self.conv3(x0)
        x2 = self.conv4(x2)     # [2, 256, 48, 48]
        x2 = self.bn(x2)
        x = x1 + x2
        # x = torch.mul(x1, x2)       # [2, 256, 48, 48]
        x = self.relu(x)
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class PSnet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(PSnet, self).__init__()
        self.conv1 = nn.Sequential(ConvBN(n_channels, 16),
                                   ConvBN(16, 32),
                                   ASPP(32,64),)
        self.conv2 = nn.Sequential(ConvBN(64, 64),
                                   ASPP(64,64),
                                   ConvBN(64, 64),)
        self.conv3 = nn.Sequential(ConvBN(128, 128),
                                   ASPP(128, 128),
                                   ConvBN(128, 128),)
        self.conv4 = nn.Sequential(ConvBN(256, 256),
                                   ASPP(256, 256))
        self.mamba = Mamba(d_model=4, d_state=16, d_conv=4, expand=2)
        self.mamba1 = Mamba(d_model=64, d_state=16, d_conv=4, expand=2)
        self.mamba2 = Mamba(d_model=64, d_state=16, d_conv=4, expand=2)
        self.mamba3 = Mamba(d_model=128, d_state=16, d_conv=4, expand=2)
        self.mamba4 = Mamba(d_model=256, d_state=16, d_conv=4, expand=2)

        self.down1 = ConvDown(64, 64)
        self.down2 = ConvDown(64, 128)
        self.down3 = ConvDown(128, 256)
        self.down4 = ConvDown(256, 512)
        # self.attention_pool = AttentionPooling(4)

        self.conv_seg0 = ConvBN1(256, 256)   # 256 128
        self.conv_seg1 = nn.Sequential(ConvBN1(512, 384), ConvBN1(384, 256))
        self.conv_seg2 = nn.Sequential(ConvBN1(512, 256),ConvBN1(256, 128))
        self.conv_seg3 = ConvBN1(256,256)    # 64 128
        self.conv_seg4 = ConvBN1(128,128)
        self.conv_seg5 = ConvBN1(128*2,64)
        self.conv_seg6 = ConvBN1(64 *2,64)
        self.conv_seg7 = nn.Conv2d(32, n_classes, kernel_size=3, padding=1)
        self.diffnet = DiffNet(64, n_classes)   #32
        self.up1 = ConvUp(256, 256)
        self.up2 = ConvUp(256, 128)
        self.up3 = nn.Sequential(ConvUp(128, 128), ConvBN(128, 64))
        self.up4 = ConvUp(64, 64)
        self.up5 = ConvUp(256, 128)

        self.fusion_bs = Fusion_basic()
        # self.fusion2_2 = CascadeFusion(128, 384, 128)
        # self.fusion2_3 = CascadeFusion(128, 128, 64)
        # self.profusion1 = ProFusion(128, 128, 128)
        # self.profusion2 = ProFusion(128, 64, 128)
        self.arm1 = ARM(256)
        self.arm2 = ARM(128)
        self.arm3 = ARM(64)
        self.arm4 = ARM(64)
        self.ca0 = CrossAttention(in_dim=256)
        self.ca1 = CrossAttention(in_dim=256)
        self.ca2 = CrossAttention(in_dim=128)
        self.ca3 = CrossAttention(in_dim=64)
        # self.ca4 = CrossAttention(in_dim=64)

        self.lsconv1 = LSConv(dim=256, lks=7, sks=3, groups=16)
        self.lsconv2 = LSConv(dim=128, lks=7, sks=3, groups=8)
        self.lsconv3 = LSConv(dim=64, lks=7, sks=3, groups=8)
        self.lsconv4 = LSConv(dim=64, lks=7, sks=3, groups=8)
        # self.heat1 = HorBlock(dim=128)
        # self.heat2 = HorBlock(dim=64)
        self.loss_f = nn.MSELoss()

    def mamba_block(self, x, n):
        x_raw = x.permute(0, 2, 3, 1).contiguous()
        B, H, W, C = x_raw.shape
        x_flat = x_raw.view(1, -1, C)
        mamba_func = {1: self.mamba1, 2: self.mamba2, 3: self.mamba3, 4:self.mamba4}
        if n in mamba_func:
            x_flat = mamba_func[n](x_flat)

        y = x_flat.view(B, H, W, C)
        y = y.permute(0, 3, 1, 2).contiguous()
        return y

    def b_encoder(self, x):     # [b, 4, 384, 384]
        x0 = self.conv1(x)  # [b, 64, 384, 384]
        x0 = self.mamba_block(x0, 1)
        x1 = self.down1(x0)  # [b, 64, 192, 192]
        x2 = self.conv2(x1)  # 64dim
        x2 = self.mamba_block(x2, 2)
        x2 = self.down2(x2)  # [b, 128, 96, 96]
        x3 = self.conv3(x2)
        x3 = self.mamba_block(x3, 3)
        x3 = self.down3(x3)  # [b, 256, 48, 48]
        # x4 = self.conv4(x3)
        # x4 = self.up5(x4)
        return x3, x2, x1

    def s_encoder(self, x):     # [b, 4, 384, 384]
        x1 = self.conv1(x)      # [b, 64, 384, 384]
        x1 = self.mamba_block(x1, 1)
        x1 = self.down1(x1)     # [b, 64, 192, 192]
        x2 = self.conv2(x1)     # 64dim
        x2 = self.mamba_block(x2, 2)
        x2 = self.down2(x2)      # [b, 128, 96, 96]
        x3 = self.conv3(x2)
        x3 = self.mamba_block(x3, 3)
        x3 = self.down3(x3)     # [b, 256, 48, 48]
        x4 = self.conv4(x3)
        x4 = self.mamba_block(x4, 4)
        x4 = self.down4(x4)     # [b, 512, 24, 24]
        return x4, x3, x2, x1

    def fusion2(self, x_s, x_p, location_map):       # x_b=[3, 256, 48, 48], x_s=24,[3, 256, 48, 48],96,192,384
        x_w = self.fusion_bs(x_p, location_map)  # 从大图的特征图中扣出小图 x_w为[feature_min]
        x_w1 = self.conv_seg0(x_w)  # ([3, 256, 48, 48]) #feature_min经过1*1的卷积
        x_s0 = self.conv_seg1(x_s[0])
        x_s0 = self.up1(x_s0)
        f_loss = self.loss_f(x_s0, x_w1)
        # x_fusion0 = torch.cat((x_s0, x_w1), dim=1)
        # x_fusion0 = self.conv_seg3(x_fusion0)
        x_fusion0 = self.ca0(x_s0, x_w1)
        x_fusion0 = self.lsconv1(x_fusion0)    # [3, 256, 48, 48]
        # x_fusion1 = torch.cat((x_fusion0, x_s[1]), dim=1)  # torch.Size([3, 512, 48, 48]) # 拼接大图中的小图和小图
        x_fusion1 = self.ca1(x_s[1], x_fusion0)

        x_fusion1 = self.conv_seg3(x_fusion1)   # x_fusion=[3, 256, 48, 48]
        x_fusion1 = self.up2(x_fusion1)    # [3, 128, 96, 96]
        # x_w2 = F.interpolate(x_w1, size=([96, 96]), mode='bilinear', align_corners=False)
        # x_w2 = self.conv_seg3(x_w2)      # [3, 128, 96, 96]
        x_fusion2 = self.ca2(x_s[2], x_fusion1)
        # x_fusion2 = self.ca2(x_w1, x_fusion2)
        x_fusion2 = self.lsconv2(x_fusion2)
        # x_fusion2 = self.arm2(x_fusion2)
        # x_fusion2 = torch.cat((x_fusion2, x_s[2]), dim=1)
        x_fusion2 = self.conv_seg4(x_fusion2)
        x_fusion2 = self.up3(x_fusion2)     # [3, 64, 192, 192]
        # x_w3 = F.interpolate(x_w, size=([192, 192]), mode='bilinear', align_corners=False)   # [3,256,192,192]
        # x_w3 = self.conv_seg5(x_w3)   # [3,64,192,192]
        # x_fusion31 = self.ca2(x_fusion2, x_w3)    # x_s[2]=[b,64,192,192]
        # x_fusion3 = self.ca3(x_s[3], x_fusion2)       # cuda out of memory
        x_fusion3 = self.lsconv3(x_fusion2)
        # x_fusion3 = self.arm3(x_fusion2)
        x_fusion3 = torch.cat((x_fusion3, x_s[3]), dim=1)
        x_fusion3 = self.conv_seg6(x_fusion3)
        x_fusion3 = self.up4(x_fusion3)
        # x_fusion3 = F.interpolate(x_fusion3, size=([384, 384]), mode='bilinear', align_corners=False)
        # x_fusion3 = torch.cat([x_fusion3, x_s[4]], dim=1)
        # x_fusion4 = self.ca4(x_s[4], x_fusion3)
        # x_fusion3 = self.lsconv4(x_fusion3)     # [3,64,384,384]
        # x_fusion4 = self.arm4(x_fusion3)
        # x_fusion3 = self.conv_seg7(x_fusion3)
        x_fusion4 = self.diffnet(x_fusion3)
        return x_fusion4, f_loss

    def forward(self, x_s, x_b, location_map):
        # input: image_s,image_b=[2, 4, 384, 384],location=[2, 1, 384, 384]
        x_small = self.s_encoder(x_s)
        x_big = self.b_encoder(x_b)

        x, f_loss = self.fusion2(x_small, x_big[0], location_map)
        output = torch.sigmoid(x)
        return output, f_loss


class MambaOne(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(MambaOne, self).__init__()
        self.conv1 = nn.Sequential(ConvBN(n_channels, 16),
                                   ConvBN(16, 32),
                                   ASPP(32, 64))
        self.conv2 = nn.Sequential(ConvBN(64, 64),
                                   ASPP(64, 64))
        self.conv3 = nn.Sequential(ConvBN(128, 128),
                                   ASPP(128, 128))
        self.conv4 = nn.Sequential(ConvBN(256, 256),
                                   ASPP(256, 256))

        self.down1 = ConvDown(64, 64)
        self.down2 = ConvDown(64, 128)
        self.down3 = ConvDown(128, 256)
        self.down4 = ConvDown(256, 512)

        self.mamba = Mamba(d_model=4, d_state=16, d_conv=4, expand=2)
        self.mamba1 = Mamba(d_model=32, d_state=16, d_conv=4, expand=2)
        self.down5 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.mamba2 = Mamba(d_model=64, d_state=16, d_conv=4, expand=2)
        self.mamba3 = Mamba(d_model=128, d_state=16, d_conv=4, expand=2)
        self.mamba4 = Mamba(d_model=256, d_state=16, d_conv=4, expand=2)

        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv5 = nn.Sequential(ConvBN(256 *2, 256), ConvBN(256,256))
        self.conv6 = nn.Sequential(ConvBN1(128 *2,128), ConvBN(128,128))
        self.conv7 = nn.Sequential(ConvBN1(64 * 2,64), ConvBN(64,64))
        self.conv8 = ConvBN(32,64)        # 96 32
        self.conv9 = ConvBN(128, 128)
        self.ca1 = CrossAttention(in_dim=256)       # CrissCrossAttention
        self.ca2 = CrossAttention(in_dim=128)
        # self.ca3 = CrissCrossAttention(in_dim=64)
        # self.arm1 = ARM(256)
        # self.arm2 = ARM(128)
        # self.arm3 = ARM(64)
        self.lsconv1 = LSConv(256)
        self.lsconv2 = LSConv(128)
        self.lsconv3 = LSConv(64)
        self.lsconv4 = LSConv(64)

        # self.proj_head = ProjectionHead(dim_in=512, proj_dim=256)        # 64
        self.diffnet = DiffNet(64, n_classes)
        # self.head = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(), nn.Linear(512, 256),
        #                           nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, n_classes))


    def mamba_block(self, x, n):
        x_raw = x.permute(0, 2, 3, 1).contiguous()
        B, H, W, C = x_raw.shape
        x_flat = x_raw.view(1, -1, C)
        mamba_func = {1: self.mamba1, 2: self.mamba2, 3: self.mamba3, 4: self.mamba4}
        if n in mamba_func:
            x_flat = mamba_func[n](x_flat)

        y = x_flat.view(B, H, W, C)
        y = y.permute(0, 3, 1, 2).contiguous()
        return y


    def forward(self, x_s):     # [b, 4, 384, 384]
        x1 = self.conv1(x_s)  # [b, 64, 384, 384]
        x1 = self.down1(x1)  # [b, 64, 192, 192]
        x2 = self.conv2(x1)  # 64dim
        x2 = self.mamba_block(x2, 2)
        x2 = self.down2(x2)  # [b, 128, 96, 96]
        x3 = self.conv3(x2)
        x3 = self.mamba_block(x3, 3)
        x3 = self.down3(x3)  # [b, 256, 48, 48]
        x4 = self.conv4(x3)
        x4 = self.mamba_block(x4, 4)
        x4 = self.down4(x4)     # [b, 512, 24, 24]

        x5 = self.up4(x4)  # [b, 256, 48, 48]
        x5 = torch.cat([x5, x3], dim=1)
        x5 = self.conv5(x5)
        # x5 = self.ca1(x3, x5)
        x5 = self.lsconv1(x5)
        x6 = self.up3(x5)  # [b, 128, 96, 96]
        x6 = torch.cat([x6, x2], dim=1)
        x6 = self.conv6(x6)
        # x6 = self.ca2(x2, x6)
        x6 = self.lsconv2(x6)
        # x6 = self.conv9(x6)
        x7 = self.up2(x6)  # [b, 64, 192, 192]
        # x7 = self.ca3(x7, x1)
        x7 = torch.cat([x7, x1], dim=1)
        x7 = self.conv7(x7)
        x7 = self.lsconv3(x7)
        x8 = self.up1(x7)   # [b, 32, 384, 384]
        # x8 = torch.cat([x8, x0], dim=1)
        x8 = self.conv8(x8)     # [b, 16, 384, 384]
        # x8 = self.lsconv4(x8)
        # x8 = self.seg(x8)
        out = self.diffnet(x8)       # [b, 1, 384, 384]
        out = torch.sigmoid(out)
        # emb = self.proj_head(x8)
        return out
        # return x4


class Encoder(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(Encoder, self).__init__()
        self.conv1 = nn.Sequential(ConvBN(n_channels, 16),
                                   ConvBN(16, 32),
                                   ASPP(32, 64))
        self.conv2 = nn.Sequential(ConvBN(64, 64),
                                   ASPP(64, 64))
        self.conv3 = nn.Sequential(ConvBN(128, 128),
                                   ASPP(128, 128))
        self.conv4 = nn.Sequential(ConvBN(256, 256),
                                   ASPP(256, 256))

        self.down1 = ConvDown(64, 64)
        self.down2 = ConvDown(64, 128)
        self.down3 = ConvDown(128, 256)
        self.down4 = ConvDown(256, 512)

        self.mamba = Mamba(d_model=4, d_state=16, d_conv=4, expand=2)
        self.mamba1 = Mamba(d_model=32, d_state=16, d_conv=4, expand=2)
        self.down5 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.mamba2 = Mamba(d_model=64, d_state=16, d_conv=4, expand=2)
        self.mamba3 = Mamba(d_model=128, d_state=16, d_conv=4, expand=2)
        self.mamba4 = Mamba(d_model=256, d_state=16, d_conv=4, expand=2)

    def mamba_block(self, x, n):
        x_raw = x.permute(0, 2, 3, 1).contiguous()
        B, H, W, C = x_raw.shape
        x_flat = x_raw.view(1, -1, C)
        mamba_func = {1: self.mamba1, 2: self.mamba2, 3: self.mamba3, 4: self.mamba4}
        if n in mamba_func:
            x_flat = mamba_func[n](x_flat)

        y = x_flat.view(B, H, W, C)
        y = y.permute(0, 3, 1, 2).contiguous()
        return y


    def forward(self, x_s):     # [b, 4, 384, 384]
        x1 = self.conv1(x_s)  # [b, 64, 384, 384]
        x1 = self.down1(x1)  # [b, 64, 192, 192]
        x2 = self.conv2(x1)  # 64dim
        x2 = self.mamba_block(x2, 2)
        x2 = self.down2(x2)  # [b, 128, 96, 96]
        x3 = self.conv3(x2)
        x3 = self.mamba_block(x3, 3)
        x3 = self.down3(x3)  # [b, 256, 48, 48]
        x4 = self.conv4(x3)
        x4 = self.mamba_block(x4, 4)
        x4 = self.down4(x4)     # [b, 512, 24, 24]
        return x4


if __name__ == "__main__":
    device = torch.device('cuda:0')

    a = torch.randn(2, 4, 384, 384).to(device)
    b = torch.randn(2, 4, 384, 384).to(device)
    c = torch.zeros(2, 1, 384, 384).to(device)
    c[:, :, :30, :30] = torch.ones(2, 1, 30, 30).to(device)

    # net1 = PSnet(4, 1).to(device)
    # out1, loss = net1(a, b, c)
    # print('out1', out1.shape)

    net2 = MambaOne(n_channels=4, n_classes=2).to(device)
    out2 = net2(a)
    print('out2', out2.shape)
    # print(out2['embed'].shape)
