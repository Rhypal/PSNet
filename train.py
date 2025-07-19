###
# main train
###
import datetime
import os
import time
import random
import numpy as np
import argparse
from torchvision.transforms import transforms
from dataloader_GF import CloudDataset, ToTensorNorm
from model.all_seg_model import ModelTwo
from generic_train import Generic_Train_1, Generic_Train_2
from torch.utils.data import DataLoader, DistributedSampler
from model.all_model_base import print_options, seed_torch
import ssl
import urllib.request
import torch, gc
from torch import distributed, multiprocessing
from torch.nn import DataParallel

gc.collect()
torch.cuda.empty_cache()    #  free empty cache

ssl._create_default_https_context = ssl._create_unverified_context
response = urllib.request.urlopen('https://example.com')
def _init_fn(worker_id):
    random.seed(10 + worker_id)
    np.random.seed(10 + worker_id)
    torch.manual_seed(10 + worker_id)
    torch.cuda.manual_seed(10 + worker_id)
    torch.cuda.manual_seed_all(10 + worker_id)
##===================================================##
##********** Configure training settings ************##
##===================================================##
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='TransGA', help='')
parser.add_argument('--batch_size', type=int, default=3, help='batch size used for training')   # sentinel, landsat8
parser.add_argument('--root', type=str, default='/mnt/894a6e1e-2772-4477-b417-173963f8ccec/dataset/new_2024/landsat8')
parser.add_argument('--train_path', type=str, default='/mnt/894a6e1e-2772-4477-b417-173963f8ccec/dataset/new_2024/landsat8/images')
parser.add_argument('--val_path', type=str, default='/mnt/894a6e1e-2772-4477-b417-173963f8ccec/dataset/new_2024/landsat8/val_images')

parser.add_argument('--optimizer', type=str, default='Adam', help='Adam')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate of optimizer')    # 1e-4
parser.add_argument('--lr_step', type=int, default=5, help='lr decay rate')
parser.add_argument('--lr_start_epoch_decay', type=int, default=10, help='epoch to start lr decay')
parser.add_argument('--max_epochs', type=int, default=30)       # 50
parser.add_argument('--save_freq', type=int, default=10)
parser.add_argument('--val_freq', type=int, default=5)  # Why not try?

parser.add_argument('--save_model_dir', type=str, default='./models/', help='directory used to store trained networks')
# parser.add_argument('--local_rank', type=int, help='local_rank for distributed training')
parser.add_argument('--resume_epoch', type=int, default=0)
parser.add_argument('--checkpath', type=str, default='save/MambaOne_sen_2025_04_11-17_35/pth/last_0.6348.pth')

opts = parser.parse_args()
current_time = datetime.datetime.now()
time_str = current_time.strftime('%Y_%m_%d-%H_%M')
opts.save_model_dir = opts.save_model_dir + f"{opts.model}/"+f"{time_str}_land/"
print_options(opts)


# #***************** choose gpu *********************##
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda:0")
# #*************** Create dataloader *****************##
seed_torch()
all_start = time.time()

train_list = os.listdir(opts.train_path)
val_list = os.listdir(opts.val_path)
train_dataset = CloudDataset(root=opts.root, file_name_list=train_list, test_mode=0)    # 训练集，只负责数据的抽象，一次调用getitem只返回一个样本
val_dataset = CloudDataset(root=opts.root, file_name_list=val_list, test_mode=2)        # test_mode=1表测试集，test_mode=2表验证集
# return image_s,image_b,label_s,label_b,location
train_dataset_size = len(train_dataset)
val_dataset_size = len(val_dataset)

print("---")
print('train size: ', train_dataset_size)
print('val size: ', val_dataset_size)
print("---")

# 训练网络时，要对一个batch的数据进行操作，同时还要shuffle，pytorch提供了DataLoader——是一个可迭代的对象
# train_dataloader = DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=4,
#                               pin_memory=True,drop_last=True, sampler=train_sampler)
# test_dataloader = DataLoader(test_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=4,
#                             pin_memory=True, worker_init_fn=_init_fn, sampler=val_sampler)

train_dataloader = DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=4,
                              pin_memory=True, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=4,
                            pin_memory=True, worker_init_fn=_init_fn)

# 训练大图加小图的网络模型
model = ModelTwo(opts).to(device)

# 训练过程
Generic_Train_1(model, opts, train_dataloader, val_dataloader).train()

file_name = os.path.join(opts.save_model_dir, 'opt.txt')    # 保存训练好的模型/checkpoint
with open(file_name, 'a') as opt_file:
    opt_file.write('[Network %s] Total Time Cost : %.3f s' % (opts.model, time.time() - all_start))
    opt_file.write('\n')
