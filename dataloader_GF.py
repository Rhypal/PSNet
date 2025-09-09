# data loader
from __future__ import print_function, division
import os
import torch
from PIL import Image
# from skimage import io
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

# from osgeo import gdal
import rasterio
from rasterio.merge import merge
from tqdm import tqdm

import torch
import random
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class MultiInputAugmentation:
    def __init__(self, hflip_prob=0.5, vflip_prob=0.5, noise_level=0.02, bright=0.5):
        """
        为多输入多标签设计的数据增强
        Args:
            hflip_prob: 水平翻转概率 (0~1)
            vflip_prob: 垂直翻转概率 (0~1)
            noise_level: 高斯噪声的标准差 (0~0.1)
        """
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.noise_level = noise_level
        self.bright = bright
        # self.bright_jitter = transforms.ColorJitter(brightness=bright)

    def __call__(self, *tensors):
        """
        输入顺序:
            [图像1, 图像2, ..., 标签1, 标签2, ...]
        输出顺序与输入相同

        规则:
            1. 所有张量应用相同的翻转操作
            2. 只对图像张量(前N个)添加噪声
            3. 标签张量不添加噪声
        """
        assert len(tensors) >= 2, "至少需要一张图像和一个标签"
        tensors = list(tensors)  # 转为可变列表

        # 随机水平翻转 (所有张量)
        if random.random() < self.hflip_prob:
            tensors = [t if t.dim() != 3 else TF.hflip(t) for t in tensors]

        # 随机垂直翻转 (所有张量)
        if random.random() < self.vflip_prob:
            tensors = [t if t.dim() != 3 else TF.vflip(t) for t in tensors]

        # 对图像张量添加高斯噪声 (只对图像)
        for i, t in enumerate(tensors):
            if i < 2:  # 前两个是图像 (根据实际数量调整)
                # if self.bright > 0:
                #     if t.min()<0 or t.max()>1:
                #         t = torch.clamp(t, 0, 1)
                #     t = self.bright_jitter(t)
                if self.noise_level > 0:
                    noise = torch.randn_like(t) * self.noise_level
                    tensors[i] = t + noise

        return tuple(tensors)


# utility functions used in the dataloaders of SEN12MS-CR and SEN12MS-CR-TS

# different bands contain different information, maybe triple turple was needed.
def image_read(path_IMG):  # [0,255]
    # 8,7,6,5 small
    # 4,3,2,1 big
    # 9 location
    tif = rasterio.open(path_IMG)
    small_img = tif.read([8, 7, 6, 5]).astype(np.float32)
    big_img = tif.read([4, 3, 2, 1]).astype(np.float32)
    location_label = tif.read([9]).astype(np.float32)  # numpy
    tif.close()

    small_img = torch.from_numpy(small_img)
    big_img = torch.from_numpy(big_img)
    location_label = torch.from_numpy(location_label)

    return {'image_s': small_img,
            'image_b': big_img,
            'll': location_label}


def label_read(path_label):
    tif = rasterio.open(path_label)
    small_label = tif.read([2]).astype(np.float32)
    big_label = tif.read([2]).astype(np.float32)
    small_label[(small_label == 3) | (small_label == 4)] = 0
    big_label[(big_label == 3) | (big_label == 4)] = 0
    small_label = torch.from_numpy(small_label)
    big_label = torch.from_numpy(big_label)

    return {'label_s': small_label,
            'label_b': big_label}


# ==========================dataset load==========================
class ToTensorNorm(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self):
        pass

    def __call__(self, sample):

        image, label = sample['image'], sample['mask']

        if (np.max(label) < 1e-6):
            label = label
        else:
            label = label / np.max(label)

        if np.max(image) != 0:
            image = image / np.max(image)
        else:
            image = image

        return {'image': torch.from_numpy(image),
                'mask': torch.from_numpy(label)}


class CloudDataset(Dataset):
    def __init__(self, root, file_name_list, test_mode=1):
        self.root = root
        self.file_name_list = file_name_list

        # self.root = './data'#'/media/xk/新加卷/code/DAFormer-master/data'r'E:/code/CDnetV2-pytorch-master-main/CDnetV2-pytorch-master-main/data/cloud/
        # self.root = '/media/user/新加卷1/wwxdata/cloud_detection/CD_data_0601/'  # path_replace
        self.test_mode = test_mode

        self.aug = MultiInputAugmentation(
            hflip_prob=0.5,
            vflip_prob=0.3,
            noise_level=0.03,
            bright=0.5
        )

    def __len__(self):
        # return 8        # skip training
        return len(self.file_name_list)

    def __getitem__(self, idx):
        if self.test_mode == 0:  # train
            img_name = self.file_name_list[idx]
            name = img_name.split('.')[0]
            label_name = 'labels/' + img_name
            img_name = 'images/' + img_name

        if self.test_mode == 1:
            img_name = self.file_name_list[idx]  # ES_LC80350192014190LGN00_ori_sample_768_line_768.tif
            name = img_name.split('.')[0]  # ES_LC80350192014190LGN00_ori_sample_768_line_768
            label_name = 'labels/' + img_name
            img_name = 'test_images/' + img_name

        if self.test_mode == 2:
            img_name = self.file_name_list[idx]
            name = img_name.split('.')[0]
            label_name = 'labels/' + img_name
            img_name = 'val_images/' + img_name

        if self.test_mode == 3:
            img_name = self.file_name_list[idx]
            name = img_name.split('.')[0]
            label_name = 'cs_label/' + img_name
            img_name = 'cs_image/' + img_name

        image = image_read(self.root + '/' + img_name)
        label = label_read(self.root + '/' + label_name)

        if self.test_mode == 0:  # train
            aug_img1, aug_img2, aug_label1, aug_label2, aug_label3 = (
                self.aug(image['image_s'], image['image_b'], label['label_s'], label['label_b'], image['ll']))

            sample = {'image_s': aug_img1, 'image_b': aug_img2,
                    'mask_s': aug_label1, 'mask_b': aug_label2,
                    'name': name, 'location': aug_label3}
        else:
            sample = {'image_s': image['image_s'], 'image_b': image['image_b'],
                      'mask_s': label['label_s'], 'mask_b': label['label_b'],
                      'name': name, 'location': image['ll']}
        # print(sample)
        return sample


class CloudDataset_v530(Dataset):
    def __init__(self, root, file_name_list, test_mode=1):
        self.root = root
        self.file_name_list = file_name_list

        self.test_mode = test_mode

        self.data_list = []
        self.__load_file__()

        self.aug = MultiInputAugmentation(
            hflip_prob=0.5,
            vflip_prob=0.3,
            noise_level=0.03,
            bright=0.5
        )

    def __len__(self):
        # return 8        # skip training
        return len(self.file_name_list)

    def __load_file__(self, ):
        for img_name in tqdm(self.file_name_list):
            name = img_name.split('.')[0]
            if self.test_mode == 0:  # train
                label_name = 'labels/' + img_name
                img_name = 'images/' + img_name
            elif self.test_mode == 1:
                label_name = 'labels/' + img_name
                img_name = 'test_images/' + img_name
            elif self.test_mode == 2:
                label_name = 'labels/' + img_name
                img_name = 'val_images/' + img_name
            else:
                raise ValueError

            image = image_read(self.root + '/' + img_name)
            label = label_read(self.root + '/' + label_name)

            sample = {'image_s': image['image_s'], 'image_b': image['image_b'],
                      'mask_s': label['label_s'], 'mask_b': label['label_b'],
                      'name': name, 'location': image['ll']}

            self.data_list.append(sample)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        if self.test_mode == 0:  # train
            aug_img1, aug_img2, aug_label1, aug_label2, aug_label3 = (
                self.aug(data['image_s'], data['image_b'], data['mask_s'], data['mask_b'], data['location']))

            data = {'image_s': aug_img1, 'image_b': aug_img2,
                    'mask_s': aug_label1, 'mask_b': aug_label2,
                    'name': data['name'], 'location': aug_label3}
            # print(data)

        return data
