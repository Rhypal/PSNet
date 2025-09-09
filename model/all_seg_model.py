import os
from torch.nn import functional as F
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn import metrics
from torch.autograd import Variable
from utils.iou_loss import IOU
from model.all_model_base import ModelBase
from model.sync_batchnorm.replicate import patch_replication_callback
from torch.optim import lr_scheduler
from model.psnet import MambaOne, PSnet
from model.other_models.BABFNet.BABFNet import BABFNet
from model.other_models.danet_single import DANet
from model.other_models.unet_model import UNet
from model.other_models.BoundaryNets_ori import BoundaryNets
from model.other_models.CDnetV2 import CDnetV2
from model.other_models.RDUNet import RDunet
from model.other_models.mcdnet import MCDNet

from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

def score_calculation(gt, pred):
    # pred_label = pred.data.cpu().numpy()
    # true_label = gt.data.cpu().numpy()
    pred_label = pred.detach().cpu().numpy()
    true_label = gt.detach().cpu().numpy()
    shape = pred_label.shape
    # targetlabel = true_label

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

    ACC = metrics.accuracy_score(true_label, pred_label)  # 准确度ACC

    return {'recall':recall_score, 'pre':pre_score,  'acc':ACC}, pred_label.reshape(shape)


class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, device='cuda', temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device)) # 超参数 温度
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float()) # 主对角线为0，其余位置全为1的mask矩阵

    def forward(self, query, pos, neg): # emb_i, emb_j 是来自同一图像的两种不同的预处理方法得到
        z_query = F.normalize(query, dim=1) # (bs, dim) ---> (bs, dim)
        z_pos = F.normalize(pos, dim=1) # (bs, dim) ---> (bs, dim)
        z_neg = F.normalize(neg, dim=1) # (bs, dim) ---> (bs, dim)
        representations_pos = torch.cat([z_query, z_pos], dim=0) # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations_pos.unsqueeze(1), representations_pos.unsqueeze(0), dim=2) # simi_mat: (2*bs, 2*bs)

        sim_ij = torch.diag(similarity_matrix, self.batch_size) # bs
        sim_ji = torch.diag(similarity_matrix, -self.batch_size) # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0) # 2*bs

        nominator = torch.exp(positives / self.temperature) # 2*bs
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature) # 2*bs, 2*bs

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1)) # 2*bs
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss


class ModelTwo(ModelBase):
    def __init__(self, opts, device):
        super(ModelTwo, self).__init__()
        self.opts = opts
        self.device = device
        # create network
        # self.seg_net = MambaCNN(4,1).to('cuda:0')  # model
        self.seg_net = PSnet(4, 1).to(device)
        # self.seg_net = trans_single(4,1)

        # if torch.cuda.is_available():
        #     if len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) > 1:
        #         model = nn.DataParallel(self.seg_net)
        #         patch_replication_callback(model)  # ?
        #     self.seg_net = self.seg_net.cuda()

        self.print_networks(self.seg_net, opts)

        # initialize optimizers
        if self.opts.optimizer == 'Adam':
            self.optimizer_G = torch.optim.Adam(self.seg_net.parameters(), lr=opts.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-04)
        elif self.opts.optimizer == 'SGD':
            self.optimizer_G = torch.optim.SGD(self.seg_net.parameters(), lr=opts.lr, momentum=0.9, weight_decay=1e-06)
        else:
            self.optimizer_G = torch.optim.AdamW(self.seg_net.parameters(), lr=opts.lr, weight_decay=1e-04)

        self.lr_scheduler = lr_scheduler.StepLR(self.optimizer_G, step_size=self.opts.lr_step, gamma=0.5)

        self.bce_loss = nn.BCELoss(reduction='mean')      # BCEWithLogitsLoss
        self.iou_loss = IOU(size_average=True)
        # self.contarstive_loss = PixelContrastLoss()

    def set_input(self, inputs):
        if torch.cuda.is_available():
            image_s = Variable(inputs['image_s']).to(self.device)
            image_b = Variable(inputs['image_b']).to(self.device)
            label_s = Variable(inputs['mask_s']).to(self.device)
            label_b = Variable(inputs['mask_b']).to(self.device)
            location_map = Variable(inputs['location']).to(self.device)
        else:
            image_s, image_b = inputs['image_s'], inputs['image_b']
            label_s, label_b = inputs['mask_s'], inputs['mask_b']
            location_map = inputs['location']  # Does it need to transfer to variable?
        return image_s, image_b, label_s, label_b, location_map

    def forward(self, image_s, image_b, location_map):
        p_label, f_loss = self.seg_net(image_s, image_b,  location_map)  # two branch
        return p_label, f_loss

    def loss_fn(self, p_label, gt):     # gt.squeeze(1).long()
        loss = self.bce_loss(p_label, gt) + self.iou_loss(p_label, gt)
        return loss

    def optimize_parameters(self, image_s, image_b, label_s, label_b, location_map):       # self.pred_label_s.argmax(dim=1).unsqueeze(1)
        self.optimizer_G.zero_grad()

        # with autocast():
        pred_label_s, f_loss = self.forward(image_s, image_b, location_map)
        loss_all = self.loss_fn(pred_label_s, label_s) + f_loss

        scores, pre_binary = score_calculation(label_s, pred_label_s)
        scaler.scale(loss_all).backward()
        torch.nn.utils.clip_grad_norm_(self.seg_net.parameters(), 1.0)
        scaler.step(self.optimizer_G)
        scaler.update()

        return loss_all.item(), pred_label_s, scores, pre_binary

    def val_score(self, image_s, image_b, label_s, label_b, location_map):
        pred_label_s, _ = self.forward(image_s, image_b, location_map)

        all, pre_binary = score_calculation(label_s, pred_label_s)

        return pred_label_s, all, pre_binary

    def save_checkpoint(self, epoch):
        self.save_network(self.seg_net, self.optimizer_G, epoch, self.lr_scheduler, self.opts.save_model_dir)

    def save_best_checkpoint(self, epoch):
        self.save_best_network(self.seg_net, self.optimizer_G, epoch, self.lr_scheduler, self.opts.save_model_dir)

    def load_resume_epoch(self):
        self.load_dict_param(self.seg_net, self.optimizer_G, self.opts.checkpath)


class ModelOne(ModelBase):
    def __init__(self, opts, device):
        super(ModelOne, self).__init__()
        self.opts = opts
        self.device = device
        # create network
        # self.seg_net = DANet(4,1).to(device='cuda:1')  # da_single
        # self.seg_net = FCN_basic(4,1)  # FCN_basic
        self.seg_net = MambaOne(4, 1).to(device)
        # self.seg_net = RDunet(4,1).to('cuda:1')

        self.print_networks(self.seg_net, opts)

        # initialize optimizers
        if self.opts.optimizer == 'Adam':
            self.optimizer_G = torch.optim.Adam(self.seg_net.parameters(), lr=opts.lr, betas=(0.9, 0.999), eps=1e-08,
                                                weight_decay=0)
        else:
            self.optimizer_G = torch.optim.SGD(self.seg_net.parameters(), lr=opts.lr, momentum=0.9, weight_decay=1e-06)

        self.bce_loss = nn.BCELoss(reduction='mean')    # BCEWithLogitsLoss
        # self.bce_loss = nn.CrossEntropyLoss()
        self.iou_loss = IOU(size_average=True)
        self.lr_scheduler = lr_scheduler.StepLR(self.optimizer_G, step_size=self.opts.lr_step, gamma=0.5)
        # self. lr_lambda = lambda iter: (1-iter /opts.max_epochs) ** 0.5
        # self.lr_scheduler = lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=self.lr_lambda)

    def set_input(self, inputs):

        self.image = inputs['image_s']
        self.label = inputs['mask_s']
        # self.location_map = inputs['location']  # Does it need to transfer to variable?
        if torch.cuda.is_available():
            self.image = Variable(self.image).to(self.device)
            self.label = Variable(self.label).to(self.device)

        return self.image, self.label

    def forward(self):
        pre_label = self.seg_net(self.image)  # single branch
        # pre_label = nn.Sigmoid()(pre_label)
        # print('p_label', pre_label.shape, pre_label)
        return pre_label

    def loss_fn(self, p_label, gt):
        loss = 1.0 * self.bce_loss(p_label, gt) + 1.0 * self.iou_loss(p_label, gt)  #0.1 0.5 1.0
        return loss

    def optimize_parameters(self):
        p_label = self.forward()
        loss_all = self.loss_fn(p_label, self.label)
        scores, pre_binary = score_calculation(self.label, p_label)
        # self.optimizer_G.zero_grad()
        # loss_all.backward()     # retain_graph=True
        # self.optimizer_G.step()
        # self.lr_scheduler.step()

        self.optimizer_G.zero_grad()
        scaler.scale(loss_all).backward()
        torch.nn.utils.clip_grad_norm_(self.seg_net.parameters(), 1.0)
        scaler.step(self.optimizer_G)
        scaler.update()

        return loss_all.item(), p_label, scores, pre_binary

    def val_score(self):
        p_label = self.forward()
        all, pre_binary = score_calculation(self.label, p_label)

        return p_label, all, pre_binary

    def save_checkpoint(self, epoch):
        self.save_network(self.seg_net, self.optimizer_G, epoch, self.lr_scheduler, self.opts.save_model_dir)

    def save_best_checkpoint(self, epoch):
        self.save_best_network(self.seg_net, self.optimizer_G, epoch, self.lr_scheduler, self.opts.save_model_dir)

    def load_resume_epoch(self):
        self.load_dict_param(self.seg_net, self.optimizer_G, self.opts.checkpath)


