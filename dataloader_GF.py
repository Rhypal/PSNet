# data loader
from __future__ import print_function, division
import os
import torch
from PIL import Image
# from skimage import io
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize


# from osgeo import gdal
import rasterio
from rasterio.merge import merge
# utility functions used in the dataloaders of SEN12MS-CR and SEN12MS-CR-TS

# different bands contain different information, maybe triple turple was needed.
def image_read(path_IMG):  # [0,255]
    # 8,7,6,5 small
    # 4,3,2,1 big
    # 9 location
    tif = rasterio.open(path_IMG)
    small_img = tif.read([8,7,6,5]).astype(np.float32)
    big_img = tif.read([4,3,2,1]).astype(np.float32)
    location_label = tif.read([9]).astype(np.float32)
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
        # self.label_name_list = lbl_nam
        # e_list
        # self.root = './data'#'/media/xk/新加卷/code/DAFormer-master/data'r'E:/code/CDnetV2-pytorch-master-main/CDnetV2-pytorch-master-main/data/cloud/
        # self.root = '/media/user/新加卷1/wwxdata/cloud_detection/CD_data_0601/'  # path_replace
        self.test_mode = test_mode

    def __len__(self):
        # return 8        # skip training
        return len(self.file_name_list)

    def __getitem__(self, idx):

        if self.test_mode == 0:

            img_name = self.file_name_list[idx]
            name = img_name.split('.')[0]

            # label_name = 'train/label/' + img_name
            # img_name = 'train/image/' + img_name
            label_name = 'labels/' + img_name
            img_name = 'images/' + img_name
        if self.test_mode == 1:

            img_name = self.file_name_list[idx]     # ES_LC80350192014190LGN00_ori_sample_768_line_768.tif
            name = img_name.split('.')[0]       # ES_LC80350192014190LGN00_ori_sample_768_line_768

            label_name = 'labels/' + img_name
            img_name = 'test_images/' + img_name
            # label_name = 'pretrain_labels/' + img_name
            # img_name = 'pretrain_images/' + img_name

        if self.test_mode == 2:

            img_name = self.file_name_list[idx]
            name = img_name.split('.')[0]

            label_name = 'labels/' + img_name
            img_name = 'val_images/' + img_name


        image = image_read(self.root + '/' + img_name)

        label = label_read(self.root + '/' + label_name)

        sample = {'image_s': image['image_s'], 'image_b': image['image_b'],
                  'mask_s': label['label_s'], 'mask_b': label['label_b'],
                  'name': name,'location':image['ll']}  #
        # print(sample)
        return sample
    