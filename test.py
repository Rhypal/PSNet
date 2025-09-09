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
import matplotlib.pyplot as plt
from model.other_models.danet_single import DANet
from model.other_models.BABFNet.BABFNet import BABFNet
from model.other_models.TransGA.TransGA import TransGANets
from model.psnet import MambaOne, PSnet
from model.other_models.unet_model import UNet
from model.other_models.BoundaryNets_ori import BoundaryNets
from model.other_models.CDnetV2 import CDnetV2
from model.other_models.mcdnet import MCDNet
from model.other_models.RDUNet import RDunet

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
    # true_label = gt.detach().cpu().numpy()
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
    pre_score = metrics.precision_score(true_label, pred_label, average='macro', zero_division=0)
    F1 = metrics.f1_score(true_label, pred_label, average='macro')
    ACC = metrics.accuracy_score(true_label, pred_label)  # 准确度ACC
    pre_lab = pred_label.reshape(shape)

    return {'recall':recall_score, 'pre':pre_score,  'acc':ACC, 'f1':F1 }, pre_lab

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


def visualize_attention_matrix(model, sample_idx=0, position=None, save_path=None):
    """
    可视化标准交叉注意力矩阵
    position: 要可视化的位置坐标 (h, w)，若为None则随机选择
    """
    model.eval()
    model.visualize = True

    # 获取注意力图 [B, H, W, H, W]
    att_matrix = model.att_map[sample_idx].cpu().numpy()
    height, width = att_matrix.shape[0], att_matrix.shape[1]

    # 如果未指定位置，选择中心点
    if position is None:
        position = (height // 2, width // 2)
    h, w = position

    # 创建可视化
    plt.figure(figsize=(12, 5))

    # 显示原始位置
    plt.subplot(1, 2, 1)
    input_img = np.ones((height, width, 3)) * 0.5  # 灰色背景
    input_img[h, w] = [1, 0, 0]  # 标记查询位置为红色
    plt.imshow(input_img)
    plt.title(f'Query Position: ({h}, {w})')
    plt.axis('off')

    # 显示注意力矩阵
    plt.subplot(1, 2, 2)
    att_map = att_matrix[h, w].reshape(height, width)
    plt.imshow(att_map, cmap='jet')
    plt.colorbar()
    plt.title('Attention Map')
    plt.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    model.visualize = False



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
            visualize_attention_matrix(model)

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


def visualize_attention(model, image, label, device):
    model.ca2.visualize = True
    model.eval()
    image = image.to(device)        # torch.Size([1, 4, 384, 384])
    label = label.to(device)
    with torch.no_grad():
        _ = model(image)  # 生成注意力图

    att_map = model.ca2.att_map
    if att_map is None:
        print("未获取到注意力图，请检查 CrossAttention 实现")
        return
    # att_map = torch.relu(att_map)

    att_map = att_map.squeeze(0).cpu().numpy()      #  [H_query, W_query, H_key, W_key]
    H, W, H_key, W_key = att_map.shape
    # comp_att = np.zeros((H //4, W //4, H_key, W_key))
    # for i in range(H //4):
    #     for j in range(W //4):
    #         # 计算当前4x4块的范围（处理不能整除的情况）
    #         h_start = i * 4
    #         h_end = min((i + 1) * 4, H)
    #         w_start = j * 4
    #         w_end = min((j + 1) * 4, W)
    #         # 对当前块的前两维求平均，保留后两维
    #         comp_att[i, j, :, :] = np.mean(
    #             att_map[h_start:h_end, w_start:w_end, :, :],
    #             axis=(0, 1)  # 对H_query和W_query维度求平均
    #         )
    # att_map = comp_att
    label_mask = label.squeeze(0).cpu().numpy()

    cloud_coords = np.argwhere(label_mask == 1)
    max_h, max_w = att_map.shape[0]-1, att_map.shape[1]-1
    if len(cloud_coords) == 0:
        center_h, center_w = max_h // 2, max_w // 2
    else:
        center_h, center_w = cloud_coords[0]
        center_h = np.clip(center_h, 0, max_h)
        center_w = np.clip(center_w, 0, max_w)
        print(center_h, center_w)

    # window = 30
    # att_map = att_map[center_h: center_h + window, center_w : center_w + window]

    # center_h, center_w = att_map.shape[0] // 8, att_map.shape[1] // 8
    att_map = att_map[center_h, center_w]
    # att_map = np.mean(att_map, axis=(0, 1))     # 2, 3

    att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min() + 1e-8)    # 归一化注意力图（0-1 范围）
    att_map = np.clip(att_map, 0, 1)
    scale_h = 384 / H_key
    scale_w = 384 / W_key
    heatmap = np.zeros((384, 384, 3))
    for i in range(H_key):
        for j in range(W_key):
            h_start = int(i * scale_h)
            h_end = min(int((i+1) * scale_h), 384)
            w_start = int(j * scale_w)
            w_end = min(int((j+1) * scale_w), 384)
            color = plt.cm.viridis(att_map[i, j])[:3]
            heatmap[h_start:h_end, w_start:w_end] = color

    # heatmap = cv2.resize(att_map, (image.shape[3], image.shape[2]))
    # heatmap = cv2.normalize(heatmap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # heatmap = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

    image1 = image.squeeze(0).permute(1,2,0).cpu().numpy()       # 转换为[H,W,C]格式
    image1 = image1[..., [3,2,1]]
    # image1= cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image1 = (image1 - image1.min()) / (image1.max() - image1.min() + 1e-8)
    image1 = np.clip(image1, 0.0,1.0)

    # overlay = cv2.addWeighted(image1, 0.6, heatmap, 0.4, 0)
    overlay = 0.6 * image1 + 0.4 * heatmap

    plt.close('all')    # 清除之前的绘图
    plt.figure(figsize=(15, 5))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    # 子图1：原始图像
    plt.subplot(1, 3, 1)
    plt.imshow(image1)      # original_image
    plt.title(f"original image")
    plt.axis('off')

    # 子图2：注意力热力图
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap='jet')
    plt.title("heatmap")
    plt.axis('off')

    # 子图3：叠加图像
    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("overlay")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # 关闭注意力可视化开关
    model.ca2.visualize = False


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
    valid_samples = 0
    ACC = 0
    Precision = 0
    Recall_score = 0
    F1 = 0
    write_log_path = opts.result_save + 'result_log.txt'
    fp =open(write_log_path, "a+")
    title = str(datetime.now())
    fp.write(title + '\n')
    model.eval()

    with torch.no_grad():
        for inputs in test_dataloader:
            image_s, label_s = inputs['image_s'], inputs['mask_s']
            if torch.cuda.is_available():
                image_s = Variable(image_s).to(device)      # [16, 4, 384, 384]
                label_s = Variable(label_s).to(device)      # [16, 1, 384, 384]

            file_name = inputs['name'][0]
            file_path = opts.result_save + file_name

            pre_prob = model(image_s)  # danet
            # pre_prob = model(image_s)["output"]  # TransGA
            # pre_prob = model(image_s, image_s)    # MCDNet
            all_metrics, pre_lab = score_calculation(label_s, pre_prob)
            recall_score= all_metrics['recall']
            precision = all_metrics['pre']
            acc= all_metrics['acc']
            f1 = all_metrics['f1']

            # single_image = image_s[0].unsqueeze(0)  # 单张图像，保持 batch 维度
            # single_label = label_s[0].squeeze()
            # visualize_attention(model, single_image, single_label, 'cuda')

            # save
            if acc > 0.1 and precision > 0.1:
                ACC += acc
                Precision += precision
                Recall_score += recall_score
                F1 += f1
                valid_samples += 1

                # image_save(image_s, label_s, pre_lab, file_path, acc)

                print(
                    f'{iters}, {file_name}, Recall:{recall_score:.4f}, Precision:{precision:.4f}, Acc:{acc:.4f}, F1:{f1:.4f}')
                # fp.write(
                #     f'{iters}, {file_name}, Recall:{recall_score:.4f}, Precision:{precision:.4f}, Acc:{acc:.4f}, F1:{f1:.4f}' + '\n')

            iters += 1

    print('Testing done. ')
    # print(f'Acc:{ACC / iters:.4f}, Precision:{Precision / iters:.4f}, Recall:{Recall_score / iters:.4f}, F1:{F1 / iters:.4f}' + '\n'+ f'Time cost: {time.time()-all_start:.4f}')
    # fp.write(
    #     'Testing done. ' + '\n' + f'Acc:{ACC / iters:.4f}, Precision:{Precision / iters:.4f}, Recall{Recall_score / iters:.4f}, F1:{F1 / iters:.4f}')

    print(f'Acc:{ACC / valid_samples:.4f}, Precision:{Precision / valid_samples:.4f}, '
          f'Recall:{Recall_score / valid_samples:.4f}, F1:{F1 / valid_samples:.4f}' + '\n' + f'Time cost: {time.time() - all_start :.4f}')
    fp.write(
        'Testing done. ' + '\n' + f'Acc:{ACC / valid_samples:.4f}, Precision:{Precision / valid_samples:.4f}, '
                                  f'Recall{Recall_score / valid_samples:.4f}, F1:{F1 / valid_samples:.4f}' +
                                '\n' + f'Time cost: {time.time() - all_start:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_sz', type=int, default=16, help='batch size used for testing')  # 16
    parser.add_argument('--model_name', type=str, default='GF_MCDNet/time')  #  danet, danet_singel, fcn_basic

    # parser.add_argument('--test_path', type=str, default='/media/user/新加卷1/wwxdata/cloud_detection/CD_data_0601/test/image')
    parser.add_argument('--test_path', type=str,         # sentinel-2, landsat8, GF-2, sentinel-2/cs_image
                        default='/mnt/894a6e1e-2772-4477-b417-173963f8ccec/dataset/new_2024/GF-2/test_images')
    parser.add_argument('--root', type=str, default='/mnt/894a6e1e-2772-4477-b417-173963f8ccec/dataset/new_2024/GF-2')
    parser.add_argument('--result_save', type=str, default='./result/')

    opts = parser.parse_args()
    opts.result_save = opts.result_save + opts.model_name + '/'
    if not os.path.exists(opts.result_save):
        os.makedirs(opts.result_save)

    device = torch.device("cuda:0")
    # Seg_Net = PTCD(4, 1).to(device)
    Seg_Net = MambaOne(4,1).to(device)
    # Seg_Net = MCDNet(4, 1).to(device)        # ['output']
    Seg_Net = nn.DataParallel(Seg_Net)

    checkpoint = torch.load('models/MCDNet/2025_04_10-20_43_GF/model_pth/best_20_net_Seg.pth', map_location=device)
    Seg_Net.load_state_dict(checkpoint['network'], strict=False)
    Seg_Net.eval()
    # test(Seg_Net, opts)
    test_small(Seg_Net, opts)

