# data loader
# from __future__ import print_function, division
import os
import cv2
import torch
# from skimage import io
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize
# from osgeo import gdal
import rasterio
from PIL import Image
import matplotlib.pyplot as plt
from rasterio.merge import merge
# utility functions used in the dataloaders of SEN12MS-CR and SEN12MS-CR-TS

# different bands contain different information, maybe triple turple was needed.
'''
def image_read(path_IMG):  # [0,255]
    # 8,7,6,5 small
    # 4,3,2,1 big
    # 9 location
    tif = rasterio.open(path_IMG)
    small_img = tif.read([8,7,6,5]).astype(np.float32)
    big_img = tif.read([4,3,2,1]).astype(np.float32)
    location_label = tif.read([9]).astype(np.float32)
    return {'image_s': small_img/255.0,
            'image_b': big_img/255.0,
            'll': location_label}


def label_read(path_label):
    tif = rasterio.open(path_label)
    small_label = tif.read([2]).astype(np.float32)
    big_label = tif.read([1]).astype(np.float32)

    return {'label_s': small_label,
            'label_b': big_label}
'''

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


class Cityscapes(Dataset):
    def __init__(self, root, split, transforms, img_transform, label_transform, debug=False):
        """:param root(string) – Root directory of dataset where directory leftImg8bit and gtFine or gtCoarse
         are located.
         :param split (string, optional) – The image split to use, train, test or val
         :param transforms (callable, optional) – A function/transform that takes input sample and its target
         as entry and returns a transformed version.
         :param img_transform (callable, optional) – A function/transform that takes in an image and returns a
         transformed version. E.g, transforms.RandomCrop
         :param label_transform (callable, optional) – A function/transform that takes in the target and transforms it.
         """
        # self.root = root
        self.split = split
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.transforms = transforms

        self.img_list, self.label_list, self.name_list = self.__list_dirs(root) # 需要一个路径
        self.label_id_list = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]

        if debug:
            self.img_list = self.img_list[:80]
            self.label_list = self.label_list[:80]
            self.name_list = self.label_list[:80]

    def __list_dirs(self, root_dir):
        img_list = list()
        label_list = list()
        name_list = list()

        image_dir = os.path.join(root_dir, 'leftImg8bit', self.split)
        label_dir = os.path.join(root_dir, 'gtFine', self.split)

        # files = sorted(os.listdir(image_dir))
        for city in sorted(os.listdir(image_dir)):
            current_image_dir = os.path.join(image_dir, city)
            current_label_dir = os.path.join(label_dir, city)
            for file_name in os.listdir(current_image_dir):
                image_name = '.'.join(file_name.split('.')[:-1])

                img_path = os.path.join(current_image_dir, file_name)

                label_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0], 'gtFine_labelIds.png')
                label_path = os.path.join(current_label_dir, label_name)
                if not os.path.exists(label_path) or not os.path.exists(img_path):
                    continue
                img_list.append(img_path)
                label_list.append(label_path)
                name_list.append(image_name)

        return img_list, label_list, name_list

    # @staticmethod
    # def tonp(img):
    #     if isinstance(img, Image.Image):
    #         img = np.array(img)
    #
    #     return img.astype(np.uint8)

    @staticmethod
    def cv2_read_image(image_path, mode='RGB'):
        img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if mode == 'RGB':
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        elif mode == 'BGR':
            return img_bgr

        elif mode == 'P':
            return np.array(Image.open(image_path).convert('P'))

        else:
            print('Not support mode {}'.format(mode))
            exit(1)

    def _encode_label(self, labelmap):
        labelmap = np.array(labelmap)

        shape = labelmap.shape
        encoded_labelmap = np.ones(shape=(shape[0], shape[1]), dtype=np.float32) * 255
        for i, class_id in enumerate(self.label_id_list):
            encoded_labelmap[labelmap == class_id] = i

        return encoded_labelmap

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img = self.cv2_read_image(self.img_list[index], mode='BGR')
        labelmap = self.cv2_read_image(self.label_list[index], mode='P')
        imag_name = self.name_list[index]

        if self.label_id_list is not None:
            labelmap = self._encode_label(labelmap)

        if self.transforms is not None:
            img, labelmap = self.transforms(img, labelmap=labelmap)

        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.label_transform is not None:
            labelmap = self.label_transform(labelmap)

        return dict(
            img=img,
            labelmap=labelmap,
            index=index,
            imag_name=imag_name
        )


class CloudDataset_contrastive(Dataset):
    def __init__(self, file_name_list, test_mode=1):
        self.file_name_list = file_name_list
        # self.label_name_list = lbl_nam
        # e_list
        # self.root = './data'  #'/media/xk/新加卷/code/DAFormer-master/data'r'E:/code/CDnetV2-pytorch-master-main/CDnetV2-pytorch-master-main/data/cloud/
        # self.root = '/media/user/新加卷1/wwxdata/cloud_detection/CD_data_0601/'  # path_replace
        # self.root = '/media/user/数据盘/new_2024/GF-2'
        self.root = '../../dataset/GF2'  # /media/user/新加卷1/new_2024/GF-2
        # self.root = '/media/xk/新加卷1/sentinel_99'
        self.test_mode = test_mode

    def __len__(self):
        # return 4
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

            img_name = self.file_name_list[idx]
            name = img_name.split('.')[0]


            # label_name = 'test/label/' + img_name
            # img_name = 'test/image/' + img_name
            label_name = 'labels/' + img_name
            img_name = 'test_images/' + img_name

        if self.test_mode == 2:

            img_name = self.file_name_list[idx]
            name = img_name.split('.')[0]

            label_name = 'val/label/' + img_name
            img_name = 'val/image/' + img_name
        image_path = self.root + '/' + img_name
        label_path = self.root + '/' +label_name
        image_dataset = rasterio.Open(image_path)
        image = image_dataset.ReadAsArray()
        image = np.transpose(image,[1,2,0])
        image = image / 255.0

        label_dataset = rasterio.Open(label_path)
        label = label_dataset.ReadAsArray()
        label = label.resize(384,384,1)
        label = np.transpose(label, [2, 0, 1])

        img_10ch = image.astype(np.float32)
        label_1ch = label.astype(np.float32)
        # img_10ch = Image.fromarray((image).astype(np.uint8))
        # label_1ch = Image.fromarray(np.uint8(label))

        sample = {'image': img_10ch, 'mask': label_1ch, 'name': name}
        print(sample)

        return sample


class CSWVDataset(Dataset):
    def __init__(self, img_cloud_dir, img_snow_dir, lab_cloud_dir, lab_snow_dir, transform=None):
        self.img_cloud_dir = img_cloud_dir
        self.img_snow_dir = img_snow_dir
        self.lab_cloud_dir = lab_cloud_dir
        self.lab_snow_dir = lab_snow_dir

        self.img_cloud_filenames = self._load_filenames(img_cloud_dir)
        self.img_snow_filenames = self._load_filenames(img_snow_dir)

        self.transform = transform

    def _load_filenames(self,base_dir):
        filenames = []
        for percent_dir in os.listdir(base_dir):
            percent_path = os.path.join(base_dir,percent_dir)
            if os.path.isdir(percent_path):
                for filename in os.listdir(percent_path):
                    if filename.endswith('.tif'):
                        filenames.append(os.path.join(percent_path,filename))
        return filenames

    def __len__(self):
        return len(self.img_cloud_filenames) + len(self.img_snow_filenames)

    def __getitem__(self, idx):
        channel = 0
        if idx < len(self.img_cloud_filenames):
            img_path = self.img_cloud_filenames[idx]
            lab_path = img_path.replace('ImgCloud','LabCloud').replace('.tif','.tif')
            # print(lab_path)
            with rasterio.open(lab_path) as src:
                # 读取指定通道的像素值
                label = src.read(channel + 1)       # rasterio的通道索引从1开始
            # label = 1  # Cloud label
        else:
            img_path = self.img_snow_filenames[idx-len(self.img_cloud_filenames)]
            lab_path = img_path.replace('ImgSnow','LabSnow').replace('.tif','.tif')
            # print(lab_path)
            with rasterio.open(lab_path) as src:
                # 读取指定通道的像素值
                label = src.read(channel + 1)
            # label = 2  # Snow label

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(lab_path) #.convert("L")  # 转为灰度图

        # # 创建一个图形窗口
        # plt.figure(figsize=(12, 6))
        #
        # # 显示RGB图像
        # plt.subplot(1, 2, 1)  # 1行2列的第1个位置
        # plt.imshow(image)
        # plt.title('RGB Image')
        # plt.axis('off')  # 关闭坐标轴
        #
        # # 显示灰度图像
        # plt.subplot(1, 2, 2)  # 1行2列的第2个位置
        # plt.imshow(mask, cmap='gray')
        # plt.title('Grayscale Mask')
        # plt.axis('off')  # 关闭坐标轴
        #
        # # 显示图像
        # plt.tight_layout()
        # plt.show()

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask, label


class SnowDataset(Dataset):
    def __init__(self, img_cloud_dir, img_snow_dir, lab_cloud_dir, lab_snow_dir, transform=None):
        self.img_cloud_dir = img_cloud_dir
        self.img_snow_dir = img_snow_dir
        self.lab_cloud_dir = lab_cloud_dir
        self.lab_snow_dir = lab_snow_dir

        self.img_cloud_filenames = self._get_cloud_paths(img_cloud_dir)
        self.img_snow_filenames, self.lab_snow_filenames = self._get_snow_paths(img_snow_dir, lab_snow_dir)
        self.transform = transform

    def _get_cloud_paths(self, directory):
        return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.tif')]

    def _get_snow_paths(self, img_snow_dir, lab_snow_dir):
        """获取雪数据和标签"""
        snow_images = []
        snow_labels = []
        for percent_folder in os.listdir(img_snow_dir):
            snow_folder = os.path.join(img_snow_dir, percent_folder)
            label_folder = os.path.join(lab_snow_dir, percent_folder)

            if os.path.isdir(snow_folder) and os.path.isdir(label_folder):
                snow_image_paths = [os.path.join(snow_folder, f) for f in os.listdir(snow_folder) if f.endswith('.tif')]
                snow_label_paths = [os.path.join(label_folder, f) for f in os.listdir(label_folder) if f.endswith('.png')]

                snow_images.extend(snow_image_paths)
                snow_labels.extend(snow_label_paths)

        return snow_images, snow_labels

    def __len__(self):
        return len(self.img_cloud_filenames) + len(self.img_snow_filenames)

    def __getitem__(self, idx):
        if idx < len(self.img_cloud_filenames):
            img_path = self.img_cloud_filenames[idx]
            lab_path = img_path.replace('pretrain_images','labels')
            # print(lab_path)
            # 读取cloud图像,标签
            with rasterio.open(img_path) as src:
                image = src.read([4,3,2,1])
            with rasterio.open(lab_path) as src:
                mask = src.read(1)       # rasterio的通道索引从1开始
                mask[mask == 3] = 0
                mask[mask == 4] = 2
        else:
            img_path = self.img_snow_filenames[idx-len(self.img_cloud_filenames)]
            lab_path = self.lab_snow_filenames[idx-len(self.img_cloud_filenames)]
            # print(lab_path)
            with rasterio.open(img_path) as src:
                image = src.read()
            with rasterio.open(lab_path) as src:
                mask = src.read(1)      # 读取指定通道的像素值
                mask[mask == 1] = 2

        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).long()

        return image, mask

class CloudS26(Dataset):
    def __init__(self, image_dir, segmask_dir,transform=None):
        self.images = self.read_muitiband_images(image_dir)  # 调用下面的方法加载遥感影像
        self.labels = self.read_water_labels(segmask_dir)  # 调用下面的方法加载加载数据集
        self.transform = transform
        #self.image_filenames = self._load_filenames(image_dir)


    # 加载images 方法
    def read_muitiband_images(self, image_dir):
        images = []  # 为images 创建一个空数组
        imgs = os.listdir(image_dir)
        for img in imgs:  # 遍历整个包含image的文文件夹
            filetype = os.path.splitext(img)[-1]  # filetype 返回文件的格式
            if filetype == '.tif':  # 判断是否为tiff格式的图像
                img_path = os.path.join(image_dir, img)  # 如果是tiff，将文件路径拼接
                with rasterio.open(img_path) as src:
                    images = src.read()  # 返回numpy数组
        return images

    # 加载labels的方法，与读取images类似
    def read_water_labels(self, segmask_dir):
        labels = []
        labs = os.listdir(segmask_dir)
        for lab in labs:
            filetype = os.path.splitext(lab)[-1]
            if filetype == '.tif':
                lab_path = os.path.join(segmask_dir, lab)
                with rasterio.open(lab_path) as src:
                    # 读取指定通道的像素值
                    labels = src.read(1)
        return labels

    # 返回数据集长度
    def __len__(self):
        return len(self.images)

    # 实现定义数据的索引操作，并将图像返回给tensor张亮
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()
        return image, label

class SenDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_filenames = self._get_paths(img_dir)
        self.transform = transform

    def _get_paths(self, directory):
        return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.tif')]

    def _read_tif(self, file_path):
        with rasterio.open(file_path) as src:
            image = src.read([8,7,6,5])
            # image = np.moveaxis(image,0,-1)  # 转换为(H, W, C)形式
            image = image.astype(np.float32)  # 转换为float32
        return image

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_filenames[idx])
        lab_path = img_path.replace('pretrain_images','pretrain_labels')
        # print(lab_path)
        # 读取cloud图像,标签
        image = self._read_tif(img_path)
        with rasterio.open(lab_path) as src:
            # 读取指定通道的像素值
            mask = src.read(2)       # rasterio的通道索引从1开始
            mask[mask == 4] = 2
        label = 1

        # image = Image.fromarray(image)
        # mask = Image.fromarray(mask)
        image = torch.tensor(image)         # [4,384,384]
        mask = mask.astype(np.float32)      # [384,384]
        mask = torch.tensor(mask)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask, label

