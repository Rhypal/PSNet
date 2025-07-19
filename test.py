import os
import time
import torch
import torch.nn as nn
import argparse
import numpy as np
import cv2
from torch.autograd import Variable
from sklearn import metrics
from datetime import datetime
from dataloader_GF import CloudDataset, ToTensorNorm
from torch.utils.data import DataLoader
import rasterio
from PIL import Image

from model.other_models.danet_single import DANet
from model.other_models.BABFNet.BABFNet import BABFNet
from model.TransGA.TransGA import TransGANets
from model.TransGA.TransGA_fusion7 import MambaCNN, MambaOne
from model.unet_model import UNet
from model.BoundaryNets_ori import BoundaryNets
from model.other_models.CDnetV2 import CDnetV2
from model.other_models.mcdnet import MCDNet


torch.cuda.empty_cache()

def ensure_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d - mi) / (ma - mi)

    return dn

def save_output(image_name, pred, d_dir):
    predict = pred
    predict = predict.squeeze()
    cloud_level = np.sum(pred)*10 / pred.size
    predict_np = (predict * 255).astype(np.uint8)
    # print("SAVE", predict_np)
    # print(predict_np.shape)
    # color_file = Image.fromarray(colorize(predict_np).transpose(1, 2, 0), 'RGB')
    im = Image.fromarray(predict_np).convert('RGB')
    # im.show()
    # img = Image.fromarray(colorize(color_file).transpose(1, 2, 0), 'RGB')
    im.save(d_dir + image_name + "_%3f" % (cloud_level) + '.png')

def score_calculation(gt, pred):
    pred_label = pred.data.cpu().numpy()
    true_label = gt.data.cpu().numpy()
    shape = pred_label.shape
    # pred_label = pred.detach().cpu()
    #
    # true_label = gt.detach().cpu().numpy()
    targetlabel = true_label

    pred_label = pred_label.flatten()
    true_label = true_label.flatten()
    true_label = true_label.astype("int")
    # print(pred_label, true_label)
    for a in range(len(pred_label)):
        if pred_label[a] > 0.5:
            pred_label[a] = 1
        else:
            pred_label[a] = 0

    recall_score = metrics.recall_score(true_label, pred_label, average='macro')  # 召回率
    pre_score = metrics.precision_score(true_label, pred_label, average='macro')
    F1 = metrics.f1_score(true_label, pred_label, average='macro')

    ACC = metrics.accuracy_score(true_label, pred_label)  # 准确度ACC

    return {'recall':recall_score, 'pre':pre_score,  'acc':ACC, 'f1':F1 }, pred_label.reshape(shape)

def image_save(origin_image, gt, pre_label, filename,acc):
    # for result visualization, three image concatenate on row,based on spanet
    # setting the shape we needed
    gap = 10
    gap_color = (0, 255, 0)
    # allim = np.zeros((2, 2, 3, 384, 384))
    allim = np.zeros((1, 3, 3, 384, 384))
    # print(origin_image.shape, origin_image)
    o = origin_image[0, [1,2,3], :, :].data.cpu().numpy()   # * 255
    # print(o.min(), o.max())
    l = gt[0, :, :, :].data.cpu().numpy() * 255
    # prob = pre_prob[0, :, :, :].data.cpu().numpy() * 255
    ore_l = pre_label[0, :, :, :] * 255
    allim[0, 0, :] = o
    allim[0, 1, :] = l
    # allim[1, 0, :] = prob
    allim[0, 2, :] = ore_l

    allim = allim.transpose(0, 3, 1, 4, 2)
    allim = allim.reshape((384, 3*384, 3))
    new_width = 3 * 384 + 2 * gap
    new_image = np.zeros((384, new_width, 3), dtype=allim.dtype)
    new_image[:, :384, :] = allim[:, :384, :]
    new_image[:, 384:384 + gap, :] = gap_color
    new_image[:, 384+gap:384*2+gap, :] = allim[:, 384:384*2, :]
    new_image[:, 384*2+gap:384*2+2*gap, :] = gap_color
    new_image[:, 384*2+2*gap:, :] = allim[:, 384*2:, :]
    cv2.imwrite(filename + '_' +str(round(acc,3)) + '.png', new_image)
    # im.save()


# later complication
def test(model, opts):
    all_start = time.time()

    # dataloader
    test_list = os.listdir(opts.test_path)
    # test_list = os.listdir(opts.train_path)
    test_dataset = CloudDataset(root=opts.root, file_name_list=test_list, test_mode=1)
    test_dataset_size = len(test_dataset)
    print("---")
    print('test size: ', test_dataset_size)

    test_dataloader = DataLoader(test_dataset, batch_size=opts.batch_sz, shuffle=False, pin_memory=True)

    # average metric calculation
    iters = 0
    ACC = 0
    Precision = 0
    Recall_score = 0
    F1 = 0
    write_log_path = opts.result_save + 'result_log.txt'
    fp =open(write_log_path, "a+")
    title = str(datetime.now())
    fp.write(title + '\n')

    with torch.no_grad():
        for inputs in test_dataloader:
            image_s, image_b = inputs['image_s'], inputs['image_b']
            label_s, label_b = inputs['mask_s'], inputs['mask_b']
            location_map = inputs['location'] # How to use it ?
            if torch.cuda.is_available():
                image_s = Variable(image_s).to(device)
                image_b = Variable(image_b).to(device)
                label_s = Variable(label_s).to(device)
                label_b = Variable(label_b).to(device)
                location_map = Variable(location_map).to(device)

            file_name = inputs['name'][0]
            file_path = opts.result_save + file_name

            # pre_prob = model(image_s, image_b)  # danet
            # pre_prob,_ = model(image_s, image_b, location_map)
            # pre_prob = pre_prob["output"]   # two branch, three input["x_cat_loss"]
            pre_prob = model(image_s, image_b, location_map)

            all_metrics,pre_lab = score_calculation(label_s,pre_prob)
            recall_score= all_metrics['recall']
            precision = all_metrics['pre']
            acc= all_metrics['acc']
            f1 = all_metrics['f1']
            # save
            image_save(image_s, label_s, pre_lab, file_path, acc)

            print(
                f'{iters}, {file_name}, Recall:{recall_score:.4f}, Precision:{precision:.4f}, Acc:{acc:.4f}, F1:{f1:.4f}')
            fp.write(
                f'{iters}, {file_name}, Recall:{recall_score:.4f}, Precision:{precision:.4f}, Acc:{acc:.4f}, F1:{f1:.4f}' + '\n')
            ACC += acc
            Precision += precision
            Recall_score += recall_score
            F1 += f1

            iters += 1

    print('Testing done. ')
    print(
        f'Acc:{ACC / iters:.4f}, Precision:{Precision / iters:.4f}, Recall:{Recall_score / iters:.4f}, F1:{F1 / iters:.4f}' + '\n' + f'Time cost: {time.time() - all_start:.4f}')
    fp.write(
        'Testing done. ' + '\n' + f'Acc:{ACC / iters:.4f}, Precision:{Precision / iters:.4f}, Recall{Recall_score / iters:.4f}, F1:{F1 / iters:.4f}')


def test_small(model, opts):
    all_start = time.time()

    # dataloader
    test_list = os.listdir(opts.test_path)
    # test_list = os.listdir(opts.train_path)
    test_dataset = CloudDataset(root=opts.root, file_name_list=test_list, test_mode=1)
    test_dataset_size = len(test_dataset)
    print("---")
    print('test size: ', test_dataset_size)

    test_dataloader = DataLoader(test_dataset, batch_size=opts.batch_sz, shuffle=False, pin_memory=True)

    # average metric calculation
    iters = 0
    ACC = 0
    Precision = 0
    Recall_score = 0
    F1 = 0
    write_log_path = opts.result_save + 'result_log.txt'
    fp =open(write_log_path, "a+")
    title = str(datetime.now())
    fp.write(title + '\n')

    with torch.no_grad():
        for inputs in test_dataloader:
            image_s, label_s = inputs['image_s'], inputs['mask_s']
            if torch.cuda.is_available():
                image_s = Variable(image_s).to(device)
                label_s = Variable(label_s).to(device)

            file_name = inputs['name'][0]
            file_path = opts.result_save + file_name

            # pre_prob = model(image_s)  # danet
            pre_prob = model(image_s)["output"]  # TransGA
            # pre_prob = model(image_s, image_s)    # MCDNet
            all_metrics, pre_lab = score_calculation(label_s, pre_prob)
            recall_score= all_metrics['recall']
            precision = all_metrics['pre']
            acc= all_metrics['acc']
            f1 = all_metrics['f1']
            # save
            image_save(image_s, label_s, pre_lab, file_path, acc)

            print(
                f'{iters}, {file_name}, Recall:{recall_score:.4f}, Precision:{precision:.4f}, Acc:{acc:.4f}, F1:{f1:.4f}')
            fp.write(
                f'{iters}, {file_name}, Recall:{recall_score:.4f}, Precision:{precision:.4f}, Acc:{acc:.4f}, F1:{f1:.4f}' + '\n')
            ACC += acc
            Precision += precision
            Recall_score += recall_score
            F1 += f1

            iters += 1

    print('Testing done. ')
    print(f'Acc:{ACC / iters:.4f}, Precision:{Precision / iters:.4f}, Recall:{Recall_score / iters:.4f}, F1:{F1 / iters:.4f}' + '\n'+ f'Time cost: {time.time()-all_start:.4f}')
    fp.write(
        'Testing done. ' + '\n' + f'Acc:{ACC / iters:.4f}, Precision:{Precision / iters:.4f}, Recall{Recall_score / iters:.4f}, F1:{F1 / iters:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_sz', type=int, default=16, help='batch size used for testing')
    parser.add_argument('--model_name', type=str, default='sen_TransGA')  #  danet, danet_singel, fcn_basic

    # parser.add_argument('--test_path', type=str, default='/media/user/新加卷1/wwxdata/cloud_detection/CD_data_0601/test/image')
    parser.add_argument('--test_path', type=str,
                        default='/mnt/894a6e1e-2772-4477-b417-173963f8ccec/dataset/new_2024/sentinel-2/test_images')  # sentinel, landsat8
    parser.add_argument('--root', type=str, default='/mnt/894a6e1e-2772-4477-b417-173963f8ccec/dataset/new_2024/sentinel-2')

    parser.add_argument('--result_save', type=str, default='./result/')

    opts = parser.parse_args()
    opts.result_save = opts.result_save + opts.model_name + '/'
    if not os.path.exists(opts.result_save):
        os.makedirs(opts.result_save)

    device = torch.device("cuda:0")
    # Seg_Net = MambaCNN(4, 1).to(device)
    Seg_Net = TransGANets(4,1).to(device)       # ['output']
    # Seg_Net = nn.DataParallel(Seg_Net)

    checkpoint = torch.load('models/TransGA/2024_12_28-09_44_sen/model_pth/best_25_net_Seg.pth', map_location=device)
    Seg_Net.load_state_dict(checkpoint['network'], strict=False)
    Seg_Net.eval()

    # test(Seg_Net, opts)
    test_small(Seg_Net, opts)

