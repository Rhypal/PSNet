import os
from torch.nn import functional as F
import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from torch.autograd import Variable
from loss import CrossEntropy2d
from iou_loss import IOU

from model.danet import DANet
from model.danet_single import DANet as DA_single
# from model.FCN.FCN_basic import FCN_basic
# from model.TransGA.TransGA import TransGANets
# from model.TransGA.TransGA_fusion import TransGANets as transFU
# from model.TransGA.TransGA_fusion2 import TransGANets as transFU
from model.TransGA.TransGA_fusion5 import TransGANets as transFU
# from model.TransGA.TransGA_single import TransGANets as trans_single
# from model.TransGA.TransGA_fusion_URIM import TransGANets as transURIM
# from model.TransGA.TransGA_fusion_contrastive import TransGANets as trans_contras
from model.all_model_base import ModelBase
from model.sync_batchnorm.replicate import patch_replication_callback
from torch.optim import lr_scheduler

# contrastive loss calculation
from model.ConstrastSampling import PixelContrastLoss

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

    ACC = metrics.accuracy_score(true_label, pred_label)  # 准确度ACC

    return {'recall':recall_score, 'pa':pre_score,  'acc':ACC}, pred_label.reshape(shape)


class ModelDANet(ModelBase):
    def __init__(self, opts):
        super(ModelDANet, self).__init__()
        self.opts = opts

        # create network

        self.seg_net = DANet(4,1)  #
        if torch.cuda.is_available():
            if len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) > 1:
                model = nn.DataParallel(self.seg_net)
                patch_replication_callback(model)  # ?
            self.seg_net = self.seg_net.cuda()

        self.print_networks(self.seg_net, opts)

        # initialize optimizers
        if self.opts.optimizer == 'Adam':
            self.optimizer_G = torch.optim.Adam(self.seg_net.parameters(), lr=opts.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

        self.bce_loss = nn.BCELoss(size_average=True)
        self.iou_loss = IOU(size_average=True)
        self.lr_scheduler = lr_scheduler.StepLR(self.optimizer_G, step_size=self.opts.lr_step, gamma=0.5)

    def set_input(self, inputs):

        self.image_s, self.image_b = inputs['image_s'],  inputs['image_b']
        self.label_s, self.label_b = inputs['mask_s'], inputs['mask_b']
        self.location_map = inputs['location']  # Does it need to transfer to variable?
        if torch.cuda.is_available():
            self.image_s = Variable(self.image_s).cuda()
            self.image_b = Variable(self.image_b).cuda()
            self.label_s = Variable(self.label_s).cuda()
            self.label_b = Variable(self.label_b).cuda()

        return self.image_s, self.image_b , self.label_s, self.label_b, self.location_map

    def forward(self):
        p_label = self.seg_net(self.image_s,self.image_b)  # dual branch

        return p_label

    def loss_fn(self, p_label, gt):
        loss = self.bce_loss(p_label,gt) + self.iou_loss(p_label, gt)
        return loss

    def optimize_parameters(self):
        self.pred_label_s = self.forward()

        self.loss_all = self.loss_fn(self.pred_label_s, self.label_s)
        scores = score_calculation(self.label_s, self.pred_label_s)
        self.optimizer_G.zero_grad()
        self.loss_all.backward()
        self.optimizer_G.step()

        return self.loss_all.item(), self.pred_label_s, scores

    def val_score(self):
        self.pred_label_s = self.forward()
        all = score_calculation(self.label_s, self.pred_label_s)

        return self.pred_label_s, all

    def save_checkpoint(self, epoch):
        self.save_network(self.seg_net, self.optimizer_G, epoch, self.lr_scheduler, self.opts.save_model_dir)

    def load_resume_epoch(self, epoch):
        self.load_dict_param(epoch)

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

class ModelSingle_(ModelBase):
    def __init__(self, opts):
        super(ModelSingle_, self).__init__()
        self.opts = opts

        # create network

        # self.seg_net = DA_single(4,1)  # da_single
        # self.seg_net = FCN_basic(4,1)  # FCN_basic
        # self.seg_net = TransGANets(4,1)  # TransGA
        self.seg_net = transFU(4,1)  # TransGANets
        # self.seg_net = trans_single(4,1)
        # self.seg_net = transURIM(4, 1)  # Transfusion_urim
        # self.seg_net = trans_contras(4,1)  # Transfusion_contrastive
        if torch.cuda.is_available():
            if len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) > 1:
                model = nn.DataParallel(self.seg_net)
                patch_replication_callback(model)  # ?
            self.seg_net = self.seg_net.cuda()

        self.print_networks(self.seg_net, opts)

        # initialize optimizers
        if self.opts.optimizer == 'Adam':
            self.optimizer_G = torch.optim.Adam(self.seg_net.parameters(), lr=opts.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

        self.bce_loss = nn.BCELoss(reduction='mean')
        self.iou_loss = IOU(size_average=True)
        self.lr_scheduler = lr_scheduler.StepLR(self.optimizer_G, step_size=self.opts.lr_step, gamma=0.5)
        self.contarstive_loss = PixelContrastLoss()

    def set_input(self, inputs):

        self.image_s, self.image_b = inputs['image_s'],  inputs['image_b']
        self.label_s, self.label_b = inputs['mask_s'], inputs['mask_b']
        self.location_map = inputs['location']  # Does it need to transfer to variable?
        if torch.cuda.is_available():
            self.image_s = Variable(self.image_s).cuda()
            self.image_b = Variable(self.image_b).cuda()
            self.label_s = Variable(self.label_s).cuda()
            # self.label_b = Variable(self.label_b).cuda()
            self.location_map = Variable(self.location_map).cuda()
        return self.image_s, self.image_b, self.label_s, self.label_b, self.location_map

    def forward(self):
        # p_label = self.seg_net(self.image_s)  # single branch
        p_label = self.seg_net(self.image_s, self.image_b,  self.location_map)  # two branch
        return p_label

# 只用了这个损失
    def loss_transGA(self,p_label,contrassive_label, gt, gt2):
        x_loss, x_cas1_loss, x_cas2_loss, x_cas3_loss, x_cas4_loss = p_label["output"], p_label["x_cas1_loss"], p_label["x_cas2_loss"], \
                                                              p_label["x_cas3_loss"],p_label["x_cas4_loss"]
        loss0 = self.loss_fn(x_loss, gt)
        # loss1 = self.loss_fn(x_fuse_loss, gt)
        loss2 = self.loss_fn(x_cas1_loss, gt)
        loss3 = self.loss_fn(x_cas2_loss, gt)
        loss4 = self.loss_fn(x_cas3_loss, gt)
        loss5 = self.loss_fn(x_cas4_loss.cuda(), gt2.cuda())

        pos_sample = contrassive_label[2]  # 从大图中抠出来的小图
        neg_sample = contrassive_label[1]  # 小图
        anchor = contrassive_label[0]   # 大图和小图融合后的
        # print(pos_sample.shape,neg_sample.shape,anchor.shape)
        pos_sample = pos_sample.view(pos_sample.size(0),-1)
        neg_sample = neg_sample.view(neg_sample.size(0), -1)
        anchor = anchor.view(anchor.size(0), -1)
        sim_pos = F.cosine_similarity(anchor,pos_sample, dim=1)
        neg_pos = F.cosine_similarity(anchor,neg_sample, dim=1)
        loss_c = torch.maximum((- torch.mean(sim_pos).cuda() + torch.mean(neg_pos).cuda()),torch.tensor(0).cuda())
        torch.mean(-sim_pos)
        # print(loss_c.item())
        loss = loss0 + loss2 + loss3 + loss4 + loss5

        return loss, loss_c

    def loss_trans_contarstive(self, p_label, gt):
        # basic loss -- from loss_fn
        basic_loss = self.loss_fn(p_label["output"],gt)
        contrastive_loss = self.contarstive_loss(p_label["embed"],gt.clone(),p_label["output"].clone())

        loss = basic_loss + contrastive_loss

        return loss


    def loss_fn(self, p_label, gt):
        loss = self.bce_loss(p_label,gt) + self.iou_loss(p_label, gt)
        return loss

    def optimize_parameters(self):
        self.pred_label_s,self.contrastive_add = self.forward()
        # self.contrastive_Loss = ContrastiveLoss(batch_size=4)
        # self.loss_all = self.loss_fn(self.pred_label_s, self.label_s)
        self.loss_all, self.loss_c = self.loss_transGA(self.pred_label_s,self.contrastive_add, self.label_s, self.label_b)
        # self.contrastive = self.contrastive_Loss()
        # self.loss_all = self.loss_trans_contarstive(self.pred_label_s, self.label_s)
        scores, pre_binary = score_calculation(self.label_s, self.pred_label_s["output"])
        self.optimizer_G.zero_grad()
        self.loss_all.backward(retain_graph=True)
        self.loss_c.backward()
        self.optimizer_G.step()

        return self.loss_all.item(),self.loss_c.item(), self.pred_label_s["output"], scores, pre_binary

    def val_score(self):
        self.pred_label_s = self.forward()[0]["output"]
        all, pre_binary = score_calculation(self.label_s, self.pred_label_s)

        return self.pred_label_s, all, pre_binary

    def save_checkpoint(self, epoch):
        self.save_network(self.seg_net, self.optimizer_G, epoch, self.lr_scheduler, self.opts.save_model_dir)

    def save_best_checkpoint(self, epoch):
        self.save_best_network(self.seg_net, self.optimizer_G, epoch, self.lr_scheduler, self.opts.save_model_dir)

    def load_resume_epoch(self):
        self.load_dict_param(self.seg_net, self.optimizer_G, self.opts.checkpath)

class ModelSingle(ModelBase):
    def __init__(self, opts):
        super(ModelSingle, self).__init__()
        self.opts = opts

        # create network

        # self.seg_net = DA_single(4,1)  # da_single
        # self.seg_net = FCN_basic(4,1)  # FCN_basic
        # self.seg_net = TransGANets(4,1)  # TransGA
        self.seg_net = transFU(4,1)  # TransGANets
        # self.seg_net = trans_single(4,1)
        # self.seg_net = transURIM(4, 1)  # Transfusion_urim
        # self.seg_net = trans_contras(4,1)  # Transfusion_contrastive
        if torch.cuda.is_available():
            if len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) > 1:
                model = nn.DataParallel(self.seg_net)
                patch_replication_callback(model)  # ?
            self.seg_net = self.seg_net.cuda()

        self.print_networks(self.seg_net, opts)

        # initialize optimizers
        if self.opts.optimizer == 'Adam':
            self.optimizer_G = torch.optim.Adam(self.seg_net.parameters(), lr=opts.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

        self.bce_loss = nn.BCELoss(reduction='mean')
        self.iou_loss = IOU(size_average=True)
        self.lr_scheduler = lr_scheduler.StepLR(self.optimizer_G, step_size=self.opts.lr_step, gamma=0.5)
        self.contarstive_loss = PixelContrastLoss()

    def set_input(self, inputs):

        self.image_s, self.image_b = inputs['image_s'],  inputs['image_b']
        self.label_s, self.label_b = inputs['mask_s'], inputs['mask_b']
        self.location_map = inputs['location']  # Does it need to transfer to variable?
        if torch.cuda.is_available():
            self.image_s = Variable(self.image_s).cuda()
            self.image_b = Variable(self.image_b).cuda()
            self.label_s = Variable(self.label_s).cuda()
            # self.label_b = Variable(self.label_b).cuda()
            self.location_map = Variable(self.location_map).cuda()
        return self.image_s, self.image_b, self.label_s, self.label_b, self.location_map

    def forward(self):
        # p_label = self.seg_net(self.image_s)  # single branch
        p_label = self.seg_net(self.image_s, self.image_b,  self.location_map)  # two branch
        return p_label

# 只用了这个损失
    def loss_transGA(self,p_label,contrassive_label, gt, gt2):
        x_loss, x_cas1_loss, x_cas2_loss, x_cas3_loss, x_cas4_loss = p_label["output"], p_label["x_cas1_loss"], p_label["x_cas2_loss"], \
                                                              p_label["x_cas3_loss"],p_label["x_cas4_loss"]
        loss0 = self.loss_fn(x_loss, gt)
        # loss1 = self.loss_fn(x_fuse_loss, gt)
        # loss2 = self.loss_fn(x_cas1_loss, gt)
        # loss3 = self.loss_fn(x_cas2_loss, gt)
        # loss4 = self.loss_fn(x_cas3_loss, gt)
        # loss5 = self.loss_fn(x_cas4_loss.cuda(), gt2.cuda())

        pos_sample = contrassive_label[2]  # 从大图中抠出来的小图
        neg_sample = contrassive_label[1]  # 小图
        anchor = contrassive_label[0]   # 大图和小图融合后的
        # print(pos_sample.shape,neg_sample.shape,anchor.shape)
        pos_sample = pos_sample.view(pos_sample.size(0),-1)
        neg_sample = neg_sample.view(neg_sample.size(0), -1)
        anchor = anchor.view(anchor.size(0), -1)
        sim_pos = F.cosine_similarity(anchor,pos_sample, dim=1)
        neg_pos = F.cosine_similarity(anchor,neg_sample, dim=1)
        loss_c = torch.maximum((- torch.mean(sim_pos).cuda() + torch.mean(neg_pos).cuda()),torch.tensor(0).cuda())
        torch.mean(-sim_pos)
        # print(loss_c.item())
        # loss = loss0 + loss2 + loss3 + loss4 + loss5
        loss = loss0

        return loss, loss_c

    def loss_trans_contarstive(self, p_label, gt):
        # basic loss -- from loss_fn
        basic_loss = self.loss_fn(p_label["output"],gt)
        contrastive_loss = self.contarstive_loss(p_label["embed"],gt.clone(),p_label["output"].clone())

        loss = basic_loss + contrastive_loss

        return loss


    def loss_fn(self, p_label, gt):
        loss = self.bce_loss(p_label,gt) + self.iou_loss(p_label, gt)
        return loss

    def optimize_parameters(self):
        self.pred_label_s,self.contrastive_add = self.forward()
        # self.contrastive_Loss = ContrastiveLoss(batch_size=4)
        # self.loss_all = self.loss_fn(self.pred_label_s, self.label_s)
        self.loss_all, self.loss_c = self.loss_transGA(self.pred_label_s,self.contrastive_add, self.label_s, self.label_b)
        # self.contrastive = self.contrastive_Loss()
        # self.loss_all = self.loss_trans_contarstive(self.pred_label_s, self.label_s)
        scores, pre_binary = score_calculation(self.label_s, self.pred_label_s["output"])
        self.optimizer_G.zero_grad()
        self.loss_all.backward(retain_graph=True)
        self.loss_c.backward()

        self.optimizer_G.step()

        return self.loss_all.item(),self.loss_c.item(), self.pred_label_s["output"], scores, pre_binary

    def val_score(self):
        self.pred_label_s = self.forward()[0]["output"]
        all, pre_binary = score_calculation(self.label_s, self.pred_label_s)

        return self.pred_label_s, all, pre_binary

    def save_checkpoint(self, epoch):
        self.save_network(self.seg_net, self.optimizer_G, epoch, self.lr_scheduler, self.opts.save_model_dir)

    def save_best_checkpoint(self, epoch):
        self.save_best_network(self.seg_net, self.optimizer_G, epoch, self.lr_scheduler, self.opts.save_model_dir)

    def load_resume_epoch(self):
        self.load_dict_param(self.seg_net, self.optimizer_G, self.opts.checkpath)

class ModelContrastive(ModelBase):
    def __init__(self, opts):
        super(ModelContrastive, self).__init__()
        self.opts = opts

        # create network

        # self.seg_net = DA_single(4,1)  # da_single
        # self.seg_net = FCN_basic(4,1)  # FCN_basic
        # self.seg_net = TransGANets(4,1)  # TransGA
        self.seg_net = transFU(4,1,0)  # TransGANets
        # self.seg_net = trans_single(4,1)
        # self.seg_net = transURIM(4, 1)  # Transfusion_urim
        # self.seg_net = trans_contras(4,1)  # Transfusion_contrastive
        if torch.cuda.is_available():
            if len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) > 1:
                model = nn.DataParallel(self.seg_net)
                patch_replication_callback(model)  # ?
            self.seg_net = self.seg_net.cuda()

        self.print_networks(self.seg_net, opts)

        # initialize optimizers
        if self.opts.optimizer == 'Adam':
            self.optimizer_G = torch.optim.Adam(self.seg_net.parameters(), lr=opts.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

        self.bce_loss = nn.BCELoss(reduction='mean')
        self.iou_loss = IOU(size_average=True)
        self.lr_scheduler = lr_scheduler.StepLR(self.optimizer_G, step_size=self.opts.lr_step, gamma=0.5)
        self.contarstive_loss = PixelContrastLoss()

    def set_input(self, inputs):

        self.image = inputs['image_s']
        self.label= inputs['mask_s']
        # self.location_map = inputs['location']  # Does it need to transfer to variable?
        if torch.cuda.is_available():
            self.image = Variable(self.image).cuda()
           # self.image_b = Variable(self.image_b).cuda()
            self.label = Variable(self.label).cuda()
            # self.label_b = Variable(self.label_b).cuda()
            # self.location_map = Variable(self.location_map).cuda()
        return self.image,self.label

    def forward(self):
        # p_label = self.seg_net(self.image_s)  # single branch
        logits = self.seg_net(self.image)  # two branch
        return logits



    def loss_trans_contarstive(self, p_label, gt):
        # basic loss -- from loss_fn
        basic_loss = self.loss_fn(p_label["output"],gt)
        contrastive_loss = self.contarstive_loss(p_label["embed"],gt.clone(),p_label["output"].clone())

        loss = basic_loss + contrastive_loss

        return loss


    def loss_fn(self, p_label, gt):
        loss = self.bce_loss(p_label,gt) + self.iou_loss(p_label, gt)
        return loss

    def optimize_parameters(self):
        self.pred_label_s,self.contrastive_add = self.forward()
        # self.contrastive_Loss = ContrastiveLoss(batch_size=4)
        # self.loss_all = self.loss_fn(self.pred_label_s, self.label_s)
        self.loss_all, self.loss_c = self.loss_transGA(self.pred_label_s,self.contrastive_add, self.label_s, self.label_b)
        # self.contrastive = self.contrastive_Loss()
        # self.loss_all = self.loss_trans_contarstive(self.pred_label_s, self.label_s)
        scores, pre_binary = score_calculation(self.label_s, self.pred_label_s["output"])
        self.optimizer_G.zero_grad()
        self.loss_all.backward(retain_graph=True)
        self.loss_c.backward()

        self.optimizer_G.step()

        return self.loss_all.item(),self.loss_c.item(), self.pred_label_s["output"], scores, pre_binary

    def val_score(self):
        self.pred_label_s = self.forward()[0]["output"]
        all, pre_binary = score_calculation(self.label_s, self.pred_label_s)

        return self.pred_label_s, all, pre_binary

    def save_checkpoint(self, epoch):
        self.save_network(self.seg_net, self.optimizer_G, epoch, self.lr_scheduler, self.opts.save_model_dir)

    def save_best_checkpoint(self, epoch):
        self.save_best_network(self.seg_net, self.optimizer_G, epoch, self.lr_scheduler, self.opts.save_model_dir)

    def load_resume_epoch(self):
        self.load_dict_param(self.seg_net, self.optimizer_G, self.opts.checkpath)

class ModelTwo(ModelBase):
    def __init__(self, opts):
        super(ModelTwo, self).__init__()
        self.opts = opts

        # create network

        # self.seg_net = DA_single(4,1)  # da_single
        # self.seg_net = FCN_basic(4,2)  # FCN_basic
        # self.seg_net = TransGANets(4,1)  # TransGA
        self.seg_net = transFU(4,1) # TransGANets
        # self.seg_net = trans_single(4,1)
        # self.seg_net = transURIM(4, 1)  # Transfusion_urim
        if torch.cuda.is_available():
            if len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) > 1:
                model = nn.DataParallel(self.seg_net)
                patch_replication_callback(model)  # 同步批处理标准化
            self.seg_net = self.seg_net.cuda()

        self.print_networks(self.seg_net, opts)

        # initialize optimizers
        if self.opts.optimizer == 'Adam':
            self.optimizer_G = torch.optim.Adam(self.seg_net.parameters(), lr=opts.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

        # self.bce_loss = nn.BCELoss(reduction='mean')
        # self.iou_loss = IOU(size_average=True)
        self.ce_loss = nn.CrossEntropyLoss()
        self.lr_scheduler = lr_scheduler.StepLR(self.optimizer_G, step_size=self.opts.lr_step, gamma=0.5)

    def set_input(self, inputs):

        self.image_s, self.image_b = inputs['image_s'],  inputs['image_b']
        self.label_s, self.label_b = inputs['mask_s'], inputs['mask_b']
        self.location_map = inputs['location']  # Does it need to transfer to variable?
        if torch.cuda.is_available():
            self.image_s = Variable(self.image_s).cuda()
            self.image_b = Variable(self.image_b).cuda()
            self.label_s = Variable(self.label_s).cuda()
            # self.label_b = Variable(self.label_b).cuda()
            self.location_map = Variable(self.location_map).cuda()
        return self.image_s, self.image_b, self.label_s, self.label_b, self.location_map

    def forward(self):
        p_label = self.seg_net(self.image_s)  # single branch
        # p_label = self.seg_net(self.image_s, self.image_b,  self.location_map)  # two branch
        return p_label

    def loss_transGA(self,p_label, gt):
        x_loss, x_fuse_loss, x_cas1_loss, x_cas2_loss, x_cat_loss = p_label["output"], p_label["x_fuse_loss"], p_label["x_cas1_loss"], \
                                                              p_label["x_cas2_loss"], p_label["x_cat_loss"]
        loss0 = self.loss_fn(x_loss, gt)
        loss1 = self.loss_fn(x_fuse_loss, gt)
        loss2 = self.loss_fn(x_cas1_loss, gt)
        loss3 = self.loss_fn(x_cas2_loss, gt)
        loss4 = self.loss_fn(x_cat_loss, gt)

        loss = loss0 + loss1 + loss2 + loss3 + loss4

        return loss


    def loss_fn(self, p_label, gt):
        # loss = self.bce_loss(p_label,gt) + self.iou_loss(p_label, gt)
        loss = self.ce_loss(p_label,gt.squeeze(1).long())
        return loss

    def optimize_parameters(self):
        self.pred_label_s = self.forward()

        self.loss_all = self.loss_fn(self.pred_label_s, self.label_s)
        # self.loss_all = self.loss_transGA(self.pred_label_s, self.label_s)
        scores, pre_binary = score_calculation(self.label_s, self.pred_label_s.argmax(dim=1).unsqueeze(1))
        self.optimizer_G.zero_grad()
        self.loss_all.backward()
        self.optimizer_G.step()

        return self.loss_all.item(), self.pred_label_s.argmax(dim=1).unsqueeze(1), scores, pre_binary

    def val_score(self):
        self.pred_label_s = self.forward().argmax(dim=1)
        all, pre_binary = score_calculation(self.label_s, self.pred_label_s.unsqueeze(1))

        return self.pred_label_s.unsqueeze(1), all, pre_binary

    def save_checkpoint(self, epoch):
        self.save_network(self.seg_net, self.optimizer_G, epoch, self.lr_scheduler, self.opts.save_model_dir)

    def save_best_checkpoint(self, epoch):
        self.save_best_network(self.seg_net, self.optimizer_G, epoch, self.lr_scheduler, self.opts.save_model_dir)



