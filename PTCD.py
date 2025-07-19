# import sys
# sys.path.append('/path/to/your/module')
import torch
from matplotlib import pyplot as plt
# from cloud_dection.model.FCN.FCN_2 import FCN_basic
from model.TransGA.basic_model.unet import UNet
import numpy as np
from model.TransGA.basic_model.backbone import mit_b4
from model.TransGA.basic_model.resnet_model import *

from model.TransGA.basic_model.sync_batchnorm import SynchronizedBatchNorm2d
from torch import nn
from torchvision.transforms import Resize
from torch.nn import functional as F

from model.unet_model import DoubleConv
from model.BoundaryNets_ori import BoundaryNets
from model.modules import BNReLU,ProjectionHead
from mamba_ssm import Mamba
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat
import pdb
import os
# import rasterio
#full image decode'output guide patch image encode

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
torch.cuda.empty_cache()

class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )

class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1,
                 norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU6, bias=False, inplace=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )
class ConvBNx(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNx, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), bias=bias,
                      dilation=(dilation, dilation), stride=(stride, stride),
                      padding=(((stride - 1) + dilation * (kernel_size - 1)) // 2, 0)
                      ),
            norm_layer(out_channels),
        )
class ConvBNy(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNy, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), bias=bias,
                      dilation=(dilation, dilation), stride=(stride, stride),
                      padding=(0, ((stride - 1) + dilation * (kernel_size - 1)) // 2)
                      ),
            norm_layer(out_channels),
        )
class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU6()
        )

class Channel_Selection(nn.Module):
    def __init__(self, channels, ratio=8):
        super(Channel_Selection, self).__init__()

        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_pooling = nn.AdaptiveMaxPool2d(1)

        self.fc_layers = nn.Sequential(
            Conv(channels, channels // ratio, kernel_size=1),
            nn.ReLU(),
            Conv(channels // ratio, channels, kernel_size=1)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        avg_x = self.avg_pooling(x).view(b, c, 1, 1)
        max_x = self.max_pooling(x).view(b, c, 1, 1)
        v = self.fc_layers(avg_x) + self.fc_layers(max_x)
        v = self.sigmoid(v).view(b, c,1,1)

        return v

class AdaptiveLocalFeatureExtraction(nn.Module):
    def __init__(self, dim, ratio=8,mode='v'):
        super(AdaptiveLocalFeatureExtraction, self).__init__()

        self.preconv = ConvBN(in_channels=dim,out_channels=dim,kernel_size=3)

        self.Channel_Selection = Channel_Selection(channels = dim, ratio=ratio)

        if mode=='v':
            self.convbase = ConvBNx(in_channels=dim,out_channels=dim,kernel_size=3)
            self.convlarge = ConvBNx(in_channels=dim,out_channels=dim,kernel_size=5)
        else:
            self.convbase = ConvBNy(in_channels=dim, out_channels=dim, kernel_size=3)
            self.convlarge = ConvBNy(in_channels=dim, out_channels=dim, kernel_size=5)


        self.post_conv = SeparableConvBNReLU(dim, dim, 3)


    def forward(self, x):

        s = self.Channel_Selection(self.preconv(x))
        x = self.post_conv(s * self.convbase(x) + (1 - s) * self.convlarge(x))

        return x

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
    def __init__(self, n_classes, in_ch):
        super(DiffNet, self).__init__()

        self.conv0 = nn.Conv2d(in_ch, 64, 3, padding=1)

        self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1 = SynchronizedBatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = SynchronizedBatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = SynchronizedBatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)

        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = SynchronizedBatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)

        self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=True)


        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn5 = SynchronizedBatchNorm2d(64)
        self.relu5 = nn.ReLU(inplace=True)


        self.conv_d4 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d4 = SynchronizedBatchNorm2d(64)
        self.relu_d4 = nn.ReLU(inplace=True)

        self.conv_d3 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d3 = SynchronizedBatchNorm2d(64)
        self.relu_d3 = nn.ReLU(inplace=True)

        self.conv_d2 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d2 = SynchronizedBatchNorm2d(64)
        self.relu_d2 = nn.ReLU(inplace=True)

        self.conv_d1 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d1 = SynchronizedBatchNorm2d(64)
        self.relu_d1 = nn.ReLU(inplace=True)

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

        residual = self.conv_d0(d1)
        # print('conv_d0:', residual.shape)
        x = x+residual
        # add = self.conv_d10(ori)
        # print(x.shape)
        # exit()
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

class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class Up(nn.Module):
    # bilinear是否采用双线性插值
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            # 使用双线性插值上采样
            # 上采样率为2，双线性插值模式
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        # else:
        #     # 使用转置卷积上采样
        #     self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        #     self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:

        # [N, C, H, W]
        # 上采样之后的特征图与要拼接的特征图，高度方向的差值
        diff_y = x1.size()[2] - x2.size()[2]
        # 上采样之后的特征图与要拼接的特征图，宽度方向的差值
        diff_x = x1.size()[3] - x2.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        # 1.填充差值
        x2 = F.pad(x2, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        # 2.拼接
        x = torch.cat([x1, x2], dim=1)
        x = self.up(x)
        # 3.卷积，两次卷积
        x = self.conv(x)
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

class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class MambaCNN(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(MambaCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
            )
        self.conv3 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU())
        self.mamba = Mamba(d_model=4, d_state=16, d_conv=4, expand=2)
        self.mamba1 = Mamba(d_model=32, d_state=16, d_conv=4, expand=2)
        self.mamba2 = Mamba(d_model=64, d_state=16, d_conv=4, expand=2)
        self.mamba3 = Mamba(d_model=128, d_state=16, d_conv=4, expand=2)
        # self.ssd1 = SS2D(d_model=4)
        # self.ssd2 = SS2D(d_model=64)
        # self.ssd3 = SS2D(d_model=128)

        self.down1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.down2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU())
        self.down3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                                   nn.BatchNorm2d(256),
                                   nn.LeakyReLU())
        # self.linear = nn.Linear(256, 128)
        # self.attention_pool = AttentionPooling(4)
        # self.BoundaryNets = BoundaryNets(n_channels, n_classes)

        self.conv_fusion = nn.Conv2d(512, 128, kernel_size=1)
        self.conv_seg = nn.Conv2d(256, 128, kernel_size=1)
        self.conv_seg2 = nn.Conv2d(768, 128, kernel_size=1)
        self.conv_seg1 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
                                   nn.GroupNorm(num_groups=16, num_channels=64), nn.LeakyReLU())
        self.conv_seg0 = nn.Conv2d(64, 1, kernel_size=1)
        # self.up3 = nn.ConvTranspose2d(384, 192, kernel_size=2, stride=2)
        # self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        # self.up1 = nn.ConvTranspose2d(64, 4, kernel_size=2, stride=2)

        self.fusion_bs = Fusion_basic()
        self.fusion2_2 = CascadeFusion(128, 384, 128).cuda()
        self.fusion2_3 = CascadeFusion(128, 128, 64).cuda()
        # self.multiattention = Multi_Attention()

        # self.BoundaryNets = BoundaryNets(4,1)
        self.local_v = AdaptiveLocalFeatureExtraction(384, ratio=8, mode='v')
        self.local_h = AdaptiveLocalFeatureExtraction(384, ratio=8, mode='h')
        self.diffnet = DiffNet(n_classes, 1)
        self.dropout = nn.Dropout(p=0.2)

    # def get_local_features(self, f_b, local_map):
    #     b, c, h, w = f_b.shape
    #     local_map = F.interpolate(local_map, size=(h, w), mode='bilinear', align_corners=False)
    #     _, _, h_map, w_map = local_map.shape
    #     assert h == h_map and w == w_map, "global feature map and local feature map must have same size"
    #
    #     local_features = torch.zeros_like(f_b)
    #     for i in range(b):
    #         global_feature_map = f_b[i]
    #         local_map_sample = local_map[i]
    #         local_map_sample = local_map_sample.squeeze(0)
    #
    #         for y in range(h):
    #             for x in range(w):
    #                 channel_idx = int(local_map_sample[y, x].item())
    #                 local_features[i, 0, y, x] = global_feature_map[channel_idx, y, x]
    #     # restore resolution to [b,4,384,384]
    #     local_features = F.interpolate(local_features, size=(h, w), mode='bilinear', align_corners=False)
    #     local_features = local_features.expand(-1, c, -1, -1)
    #     return local_features

    def mamba_block(self, x, n):
        x_raw = x.permute(0, 2, 3, 1).contiguous()
        B, H, W, C = x_raw.shape
        x_flat = x_raw.view(1, -1, C)
        mamba_func = {1: self.mamba1, 2: self.mamba2, 3: self.mamba3}
        if n in mamba_func:
            x_flat = mamba_func[n](x_flat)

        y = x_flat.view(B, H, W, C)
        y = y.permute(0, 3, 1, 2).contiguous()
        return y

    # def b_encoder(self, x):
    #     x0 = x.permute(0, 2, 3, 1)     # [b,384,384,4]
    #     x1 = self.ssd1(x0)
    #     x1 = x1.permute(0, 3, 1, 2)
    #     x1 = self.down1(x1)
    #     x1 = x1.permute(0, 2, 3, 1)     # [b,192,192,64]
    #     x2 = self.ssd2(x1)
    #     x2 = x2.permute(0, 3, 1, 2)
    #     x2 = self.down2(x2)
    #     x2 = x2.permute(0, 2, 3, 1)     # [b,96,96,128]
    #     x3 = self.ssd3(x2)
    #     x3 = x3.permute(0, 3, 1, 2)
    #     x3 = self.down3(x3)         # [b,256,48,48]
    #     return x3

    def s_encoder(self, x):     # [b, 4, 384, 384]
        x1 = self.conv1(x)      # [b, 32, 384, 384]
        x1 = self.mamba_block(x1, 1)
        x1 = self.down1(x1)     # [b, 64, 192, 192]
        x2 = self.conv2(x1)     # 64dim
        x2 = self.mamba_block(x2, 2)
        x2 = self.down2(x2)      # [b, 128, 96, 96]
        x3 = self.conv3(x2)
        x3 = self.mamba_block(x3, 3)
        x3 = self.down3(x3)     # [b, 256, 48, 48]

        return x3, x2, x1, x

    def encoder(self, x):     # [b, 4, 384, 384]
        x1 = self.conv1(x)      # [b, 64, 384, 384]
        x1 = self.dropout(x1)
        x1 = self.down1(x1)     # [b, 64, 192, 192]
        x2 = self.conv2(x1)     # 64dim
        x2 = self.dropout(x2)
        x2 = self.mamba_block(x2, 2)
        x2 = self.down2(x2)      # [b, 128, 96, 96]
        x3 = self.conv3(x2)
        x3 = self.mamba_block(x3, 3)
        x3 = self.down3(x3)     # [b, 256, 48, 48]

        return x3, x2, x1, x


    def fusion(self, x_s, x_b, location_map):       # x_b=[3, 256, 48, 48], x_s=[3, 256, 48, 48],96,192,384
        x_w = self.fusion_bs(x_b, location_map)  # 从大图的特征图中扣出小图 x_w为[feature_min]
        x_w1 = self.conv_seg(x_w)  # ([3, 128, 48, 48]) #feature_min经过1*1的卷积

        x_fusion = torch.cat((x_w1, x_s[0]), dim=1)  # torch.Size([3, 384, 48, 48]) # 拼接大图中的小图和小图
        x_fusion_v = self.local_v(x_fusion)  # 提取垂直的局部特征
        x_fusion_h = self.local_h(x_fusion)  # 提取水平的局部特征
        x_fusion = self.conv_seg2(torch.cat((x_fusion_v, x_fusion_h), dim=1))   # x_fusion=[3, 128, 48, 48]

        x_cas1 = torch.cat((x_fusion, x_s[0]), dim=1)  # 将x_fusion和x_s再次拼接, x_cas1=[3, 128+256, 48, 48]
        x_cas2 = F.interpolate(x_cas1, size=([96, 96]), mode='bilinear', align_corners=False)   # [3, 384, 96, 96]
        x_w2 = F.interpolate(x_b, size=([96, 96]), mode='bilinear', align_corners=False)    # [3, 256, 96, 96]

        x_w2 = self.fusion_bs(x_w2, location_map)
        x_w2 = self.conv_seg(x_w2)      # [3, 128, 96, 96]
        x_cas3 = self.fusion2_2(x_w2, x_cas2, x_s[1])       # [3,128,96,96]

        x_w3 = F.interpolate(x_b, size=([192, 192]), mode='bilinear', align_corners=False)   # [3,256,192,192]
        x_w3 = self.fusion_bs(x_w3, location_map)
        x_w3 = self.conv_seg(x_w3)   # [3,128,192,192]
        x_cas4 = self.fusion2_3(x_w3, x_cas3, x_s[2])
        x_cas4 = self.conv_seg1(x_cas4)
        x_cas4 = self.conv_seg0(x_cas4)

        return x_cas4

    def forward(self, x_s, x_b, location_map):
        # input: image_s,image_b=[2, 4, 384, 384],location=[2, 1, 384, 384]
        x_small = self.encoder(x_s)
        x_big3, x_big2, x_big1, x_big = self.encoder(x_b)

        x = self.fusion(x_small, x_big3, location_map)
        x = self.diffnet(x)
        output = torch.sigmoid(x)
        return output

class MambaOne(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(MambaOne, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(),
                                   nn.Conv2d(64, 64, kernel_size=1, padding=0, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(),
                                   nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU())
        self.down1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.mamba = Mamba(d_model=4, d_state=16, d_conv=4, expand=2)
        self.mamba1 = Mamba(d_model=32, d_state=16, d_conv=4, expand=2)
        self.down4 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.mamba2 = Mamba(d_model=64, d_state=16, d_conv=4, expand=2)
        self.mamba3 = Mamba(d_model=128, d_state=16, d_conv=4, expand=2)
        self.down2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(),
                                   nn.Conv2d(128, 128, kernel_size=1, padding=0, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU())
        self.down3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                                   nn.BatchNorm2d(256),
                                   nn.LeakyReLU(),
                                   nn.Conv2d(256, 256, kernel_size=1, padding=0, bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.LeakyReLU())

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                                   nn.GroupNorm(num_groups=32, num_channels=128), nn.SiLU())
        self.conv5 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                   nn.GroupNorm(num_groups=4, num_channels=64), nn.SiLU())
        self.conv6 = nn.Sequential(nn.Conv2d(32, 16, kernel_size=1, stride=1),
                                   nn.GroupNorm(num_groups=4, num_channels=16), nn.SiLU())
        self.seg = nn.Conv2d(16, n_classes, kernel_size=1, stride=1, padding=0)
        self.proj_head = ProjectionHead(dim_in=32, proj_dim=256)
        # self.diffnet = DiffNet(n_classes, 1)

    def mamba_block(self, x, n):
        x_raw = x.permute(0, 2, 3, 1).contiguous()
        B, H, W, C = x_raw.shape
        x_flat = x_raw.view(1, -1, C)
        mamba_func = {1: self.mamba1, 2: self.mamba2, 3: self.mamba3}
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

        x4 = self.up3(x3)    # [b, 128, 96, 96]
        x4 = self.conv4(x4)
        x5 = self.up2(x4)   # [b, 64, 192, 192]
        x5 = self.conv5(x5)
        x6 = self.up1(x5)   # [b, 32, 384, 384]
        x7 = self.conv6(x6)     # [b, 16, 384, 384]
        out = self.seg(x7)       # [b, 1, 384, 384]
        # out = torch.sigmoid(out)
        emb = self.proj_head(x6)

        return {'seg': out, 'embed': emb}
        # return out


class MambaContrast(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(MambaContrast, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(n_channels, 32, kernel_size=3, stride=1, padding=1),
                                   nn.GroupNorm(num_groups=8, num_channels=32), nn.SiLU())
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                   nn.GroupNorm(num_groups=16, num_channels=64), nn.SiLU())
        self.conv3 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                                   nn.GroupNorm(num_groups=32, num_channels=128), nn.SiLU())
        self.mamba = Mamba(d_model=4, d_state=16, d_conv=4, expand=2)
        self.mamba1 = Mamba(d_model=32, d_state=16, d_conv=4, expand=2)
        self.down4 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.mamba2 = Mamba(d_model=64, d_state=16, d_conv=4, expand=2)
        self.mamba3 = Mamba(d_model=128, d_state=16, d_conv=4, expand=2)
        self.down3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.down2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.proj_head = ProjectionHead(dim_in=128, proj_dim=256)
        self.seg = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            BNReLU(256, bn_type='torchbn'),
            nn.Dropout2d(0.10),
            nn.Conv2d(256, n_classes, kernel_size=1, stride=1, padding=0, bias=False)
        )


    def visualize_attention_map(self, x):
        feature = x[0]
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        for i in range(16):
            ax = axes[i//4, i%4]
            channel = feature[i]
            channel = (channel - channel.min()) / (channel.max() - channel.min())
            ax.imshow(channel.cpu().detach().numpy(), cmap='viridis')
            ax.axis('off')
            ax.set_title(f'channel{i+1}')

        # attn_map = x.permute(1,2,0).cpu().detach().numpy()  # 从GPU转到CPU并转换为NumPy数组
        # plt.imshow(attn_map, cmap='jet', alpha=0.5)
        # plt.colorbar()
        plt.tight_layout()
        plt.show()

    def mamba_block(self, x, n):
        x_raw = x.permute(0, 2, 3, 1).contiguous()
        B, H, W, C = x_raw.shape
        x_flat = x_raw.view(1, -1, C)
        mamba_func = {1: self.mamba1, 2: self.mamba2, 3: self.mamba3}
        if n in mamba_func:
            x_flat = mamba_func[n](x_flat)

        y = x_flat.view(B, H, W, C)
        y = y.permute(0, 3, 1, 2).contiguous()
        return y

    def forward(self, x_s):     # [b, 4, 384, 384]
        x1 = self.conv1(x_s)  # [b, 32, 384, 384]
        x1 = self.mamba_block(x1, 1)
        x1 = self.down4(x1)  # [b, 64, 192, 192]
        x2 = self.conv2(x1)
        x2 = self.mamba_block(x2, 2)
        x2 = self.down2(x2)  # [b, 128, 96, 96]
        x3 = self.conv3(x2)
        x3 = self.mamba_block(x3, 3)
        x3 = self.down3(x3)  # [b, 256, 48, 48]

        x4 = self.up2(x3)   # [b, 128, 96, 96]
        x5 = self.up1(x4)   # [b, 128, 192, 192]
        x6 = self.up1(x5)   # [b, 128, 384, 384]
        x7 = self.seg(x6)   # [b, 3, 384, 384]
        out = F.softmax(x7, dim=0)
        emb = self.proj_head(x6)

        return {'seg': out, 'embed': emb}

class ContrastNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(ContrastNet,self).__init__()
        self.cls_head = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            BNReLU(256, bn_type='torchbn'),
            nn.Dropout2d(0.10),
            nn.Conv2d(256, n_classes, kernel_size=1, stride=1, padding=0, bias=False)
        )
        self.up = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1,bias=False) # 上采样

        self.proj_head = ProjectionHead(dim_in=128, proj_dim=256)
        #  -------------Bilinear Upsampling--------------
        # self.diffnet = DiffNet(n_classes, 2)     # n_classes
        # self.backbone = {'b4': mit_b4}['b4'](True)
        # self.seg_net = FCN_basic(4, 1)
        self.BoundaryNets = BoundaryNets(n_channels,n_classes)

    def forward(self, x):
        x_p1, x_p2, x_p3, x_p4 = self.BoundaryNets(x)  # 小图经过bounarynets，x_p1是最终输出，其余都是encoder倒数三层的输出
        x_up1 = self.up(x_p1)   # [4,128,96,96]
        x_up2 = self.up(x_up1)  # [4,128,192,192]
        x_up3 = self.up(x_up2)  # [4,128,384,384]

        out = self.cls_head(x_up3)
        out = F.softmax(out, dim=0)     # [4,3,384,384]

        emb = self.proj_head(x_up3)     # [8,256,384,384]

        return {'seg':out, 'embed':emb}      # seg输出3分类，embed经过MLP输出256维

class MemoryBank:
    def __init__(self, capacity, feature_dim):
        self.capacity = capacity
        self.feature_dim = feature_dim
        self.bank = torch.zeros(capacity, feature_dim).to(device)
        self.ptr = 0

    def update(self, features):
        batch_size = features.shape[0]
        if self.ptr + batch_size > self.capacity:
            remainder = self.capacity - self.ptr
            self.bank[self.ptr:self.capacity] = features[:remainder]
            self.ptr = 0
            remaining = batch_size - remainder
            self.bank[self.ptr:self.ptr + remaining] = features[remainder:]
            self.ptr += remaining
        else:
            self.bank[self.ptr:self.ptr + batch_size] = features
            self.ptr += batch_size

    def get_bank(self):
        return self.bank


if __name__ == "__main__":
    device = torch.device('cuda:0')

    a = torch.randn(6, 4, 384, 384).to(device)
    b = torch.randn(6, 4, 384, 384).to(device)
    c = torch.zeros(6, 1, 384, 384).to(device)
    c[:, :, :30, :30] = torch.ones(6, 1, 30, 30).to(device)

    # net = TransGANets(4, 1).to(device)
    net1 = MambaCNN(4, 1).to(device)
    out1 = net1(a, b, c)
    print(out1.shape)

    # print(module['seg'].shape)
    # print(module['embed'].shape)    # [b, c, h, w]
    # print(module.get('q_mem'), "hhh")
    # net = TransGANets(n_channels=4, n_classes=2).cuda()

    net2 = MambaOne(n_channels=4, n_classes=1).to(device)
    out2 = net2(a)
    print(out2.shape)
    # print(output['seg'].shape, output['seg'])
    # print(output['embed'].shape)
