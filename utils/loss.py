import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

device = torch.device("cuda:0")
# Classes
class BCEWithLogitsLoss2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(BCEWithLogitsLoss2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, 1, h, w)
                target:(n, 1, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 4
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(2), "{0} vs {1} ".format(predict.size(2), target.size(2))
        assert predict.size(3) == target.size(3), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict[target_mask]
        loss = F.binary_cross_entropy_with_logits(predict, target, weight=weight, size_average=self.size_average)
        return loss


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.MSELoss()


    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        target_tensor = self.get_target_tensor(prediction, target_is_real) # Create label tensors with the same size as the input.
        loss = self.loss(prediction, target_tensor)  # Calculate the loss value, where 'self.loss = nn.MSELoss()'

        return loss


class CrossEntropy2d(nn.Module):

    def __init__(self, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        #loss = F.cross_entropy(predict, target, weight=weight, reduction='elementwise_mean')
        loss = F.cross_entropy(predict, target, weight=weight, reduction='mean')	
        return loss



# Cross-entropy Loss
class FSCELoss(nn.Module):
    def __init__(self, class_weights, ignore_index):
        super(FSCELoss, self).__init__()
        # weight = torch.FloatTensor(class_weights).cuda()
        weight =None
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction='mean')

    def forward(self, inputs, *targets, weights=None, **kwargs):
        loss = 0.0
        # print('Input:', inputs)
        
        if isinstance(inputs, tuple) or isinstance(inputs, list):   # 检查输入inputs是否为元组或列表
            if weights is None:
                weights = [1.0] * len(inputs)       # 权重初始化为全1

            for i in range(len(inputs)):
                if len(targets) > 1:        # 如果有多个目标，则针对每个输入使用相应的目标进行缩放
                    target = self._scale_target(targets[i], (384,384))
                    loss += weights[i] * self.ce_loss(inputs[i], target)
                else:                       # 如果只有一个目标，则使用相同的目标计算所有输入的损失
                    target = self._scale_target(targets[0], (384,384))
                    loss += weights[i] * self.ce_loss(inputs[i], target)

        else:
            target = self._scale_target(targets[0], (384,384))
            loss = self.ce_loss(inputs, target)
        # print('target:',target.shape)   # [2, 384, 384]
        return loss

    @staticmethod
    def _scale_target(targets_, scaled_size):
        # targets = targets_.clone().float()
        targets = targets_.unsqueeze(1)
        # print('before interpolation:', targets.shape)   #[2,1,384,384]
        targets = F.interpolate(targets, size=scaled_size, mode='bilinear')
        return targets.squeeze(1).long()    #[2,384,384]
        # return targets


class ContrastCELoss(nn.Module):
    def __init__(self, class_weights, ignore_index, loss_weight,
                 max_samples, max_views, temperature, base_temperature):
        super(ContrastCELoss, self).__init__()

        self.loss_weight = loss_weight
        self.seg_criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        self.contrast_criterion = PixelContrastLoss(max_samples=max_samples, max_views=max_views,
                                                    temperature=temperature, base_temperature=base_temperature,
                                                    ignore_index=ignore_index)
    def forward(self, seg, embedding, target, with_embed=False):
        # h, w = target.size(1), target.size(2)
        # assert "seg" in preds
        # assert "embed" in preds
        # seg = preds['seg']
        # embedding = preds['embed']

        # pred = F.interpolate(seg, size=(384, 384), mode='bilinear', align_corners=True)
        # print(pred.shape, target.shape)     # torch.Size([2, 3, 384, 384]) torch.Size([2, 384, 384])
        seg = torch.clamp(seg, min=1e-8, max=1 - 1e-8)
        loss = self.seg_criterion(seg, target)
        # print('ce loss:', loss)       # 0.554

        pred = torch.argmax(seg, 1)
        loss_contrast = self.contrast_criterion(embedding, target, pred)     # embedding=[b,256,384,384],target=label

        # 0.554+0.1*4.147=0.969
        return loss + self.loss_weight * loss_contrast, {'ce': loss, 'contrast': loss_contrast}

        # just a trick to avoid errors in distributed training
        # return loss + 0.0 * loss_contrast, {'ce': loss, 'contrast': 0.0}


class PixelContrastLoss(nn.Module):
    def __init__(self, max_samples, max_views, temperature, base_temperature, ignore_index):
        super(PixelContrastLoss, self).__init__()

        self.temperature = temperature
        self.base_temperature = base_temperature
        self.ignore_label = ignore_index
        self.max_samples = max_samples
        self.max_views = max_views

    def _hard_anchor_sampling(self, X, y_hat, y):
        batch_size, feat_dim = X.shape[0], X.shape[-1]

        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii]
            this_classes = torch.unique(this_y)
            this_classes = [x for x in this_classes if x != self.ignore_label]
            this_classes = [x for x in this_classes if torch.count_nonzero((this_y == x)) > self.max_views]

            classes.append(this_classes)
            total_classes += len(this_classes)

        if total_classes == 0:
            return None, None

        n_view = self.max_samples // total_classes
        n_view = min(n_view, self.max_views)

        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).to(device)
        y_ = torch.zeros(total_classes, dtype=torch.float).to(device)

        X_ptr = 0
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_classes = classes[ii]

            for cls_id in this_classes:
                hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero(as_tuple=False)
                easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero(as_tuple=False)

                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]

                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard >= n_view / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                elif num_easy >= n_view / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                else:
                    # Log.info('this shoud be never touched! {} {} {}'.format(num_hard, num_easy, n_view))
                    raise Exception

                perm = torch.randperm(num_hard)
                hard_indices = hard_indices[perm[:num_hard_keep]]
                perm = torch.randperm(num_easy)
                easy_indices = easy_indices[perm[:num_easy_keep]]
                indices = torch.cat((hard_indices, easy_indices), dim=0)

                X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                y_[X_ptr] = cls_id
                X_ptr += 1

        return X_, y_

    def _contrastive(self, feats_, labels_):
        anchor_num, n_view = feats_.shape[0], feats_.shape[1]   # 4,147456

        labels_ = labels_.contiguous().view(-1, 1)      # (589824,1)

        # Batch size for splitting the large similarity matrix
        batch_size = 1024

        # Initialize an empty list to store computed mask chunks
        mask_chunks = []

        # Compute the similarity matrix in batches
        for i in range(0, labels_.shape[0], batch_size):
            # Compute a batch of the similarity matrix
            batch_labels = labels_[i:i + batch_size]  # (batch_size, 1)
            batch_mask = torch.eq(batch_labels, labels_.transpose(0, 1)).float().to(device)
            mask_chunks.append(batch_mask)

        # Concatenate the chunks to form the final mask
        mask = torch.cat(mask_chunks, dim=0)  # (589824, 589824)

        # mask = torch.eq(labels_, torch.transpose(labels_, 0, 1)).float().cuda()

        contrast_count = n_view
        contrast_feature = torch.cat(torch.unbind(feats_, dim=1), dim=0)    # 在dim=1上拆分再拼接，把不同视图的特征合并成一个大的特征向量

        anchor_feature = contrast_feature
        anchor_count = contrast_count
        # 计算特征相似度
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)),
                                        self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask

        # logits_mask 生成一个掩码，防止anchor与自己进行对比，在对角线填充0
        logits_mask = torch.ones_like(mask).scatter_(1, torch.arange(anchor_num * anchor_count).view(-1, 1).to(device), 0)
        mask = mask * logits_mask

        # 计算对比损失
        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def forward(self, feats, labels=None, predict=None):
        labels = labels.unsqueeze(1).float().clone()        # target[b,384*384], feats=embed
        labels = labels.float().clone()
        labels = torch.nn.functional.interpolate(labels, (feats.shape[2], feats.shape[3]), mode='bilinear')
        labels = labels.squeeze(1).long()
        assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)

        batch_size = feats.shape[0]
        labels = labels.contiguous().view(batch_size, -1)       # (4,147456)
        predict = predict.contiguous().view(batch_size, -1)     # (4,147456),[[0,1,1,...0,1,0],...,[0,1,0,...1,1,0]]
        feats = feats.permute(0, 2, 3, 1)                   # [b,384,384,256]
        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])    # [b,147456,256]

        feats, labels = self._hard_anchor_sampling(feats, labels, predict)    # 选择困难样本

        loss = self._contrastive(feats, labels)
        return loss
