import datetime
import os
import time
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
from torchvision.utils import make_grid

def score_split(input):
    recall = input['recall']
    pre = input['pre']
    acc = input['acc']
    return recall, pre, acc


class Generic_Train_1():
    def __init__(self, model, opts, train_dataloader, val_dataloader):
        self.model=model
        self.opts=opts      # train_single定义的一些参数
        self.train_dataloader=train_dataloader
        self.val_dataloader=val_dataloader

    def train(self):
        prev_time = time.time()
        # instantiate tensorboard logger
        writer = SummaryWriter(self.opts.save_model_dir, comment='test_your_comment')   # 创建日志记录器，用于记录训练中的指标，以便可视化
        # total_steps = 0
        # log_loss = 0
        best_score = 0
        if self.opts.resume_epoch > 0:
            start_epoch = self.opts.resume_epoch
            self.model.load_resume_epoch()
        else:
            start_epoch = 0

        for epoch in range(start_epoch, self.opts.max_epochs):
            # self.train_dataloader.sampler.set_epoch(epoch)
            log_loss = 0
            log_loss2 = 0
            acc_average = 0
            recall_average = 0
            pre_average = 0

            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.model.optimizer_G, 'min',
                                                                      factor=0.5, patience=2, min_lr=1e-6)

            for i, data in enumerate(self.train_dataloader):
                start_time = time.time()        # 训练开始时间

                # 调用模型的set_input方法，将输入数据data分解成多个变量，输入特征、目标标签、位置信息
                x_in_s, y_s = self.model.set_input(data)
                # print('y_s:', y_s.shape, y_s)   # [8, 1, 384, 384]


                # 调用 optimize_parameters 方法进行模型的前向传播和反向传播，计算损失并更新模型参数
                batch_loss, pre_lab_s, scores, pre_binary = self.model.optimize_parameters()
                log_loss = log_loss + batch_loss    # 将当前批次的损失值累加到log_loss中
                recall, pre, acc = score_split(scores)   # 调用score_split函数，基于模型输出的scores计算评价指标，召回率、总体准确率、准确率
                acc_average = acc_average + acc
                recall_average = recall_average + recall
                pre_average = pre_average + pre

                batches_done = epoch * len(self.train_dataloader) + i   # 计算当前已经处理的批次数，len(self.train_dataloader)是每个轮次的批次数，i是当前批次
                batches_left = self.opts.max_epochs * len(self.train_dataloader) - batches_done     # 计算剩余的批次数
                time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))    # 估算剩余训练时间

                print('epoch', epoch+1, 'iteration', i, '  ', 'loss', round(batch_loss, 3), '  ', 'RECALL', round(recall,3), '  ', 'Pre',round(pre,3),
                      '  ', 'ACC',round(acc,3), '  ','use time',round(time.time() - start_time,3), 'ETA', time_left)

                prev_time = time.time()
            lr_scheduler.step(log_loss)

            writer.add_scalar("train_loss", log_loss/len(self.train_dataloader), epoch+1)        # 将标量添加到summary
            writer.add_scalar("train_acc", acc_average/len(self.train_dataloader), epoch+1)
            writer.add_image(f'Img_train/img', x_in_s[0, [1,2,3], ...], epoch+1, dataformats='CHW')
            writer.add_image(f'Img_train/pre_label', pre_lab_s[0, [0], ...], epoch+1, dataformats='CHW')  #pre_lab_s[0, [0], ...]
            writer.add_image(f'Img_train/label_binary', pre_binary[0, [0], ...], epoch + 1, dataformats='CHW')
            writer.add_image(f'Img_train/label', y_s[0, [0], ...], epoch+1, dataformats='CHW')

            if (epoch + 1) % self.opts.val_freq == 0:
                print("validation...")
                self.model.seg_net.eval()
                with torch.no_grad():
                    _iter = 0
                    recall_avg = 0
                    pre_avg = 0
                    acc_avg = 0
                    for data in self.val_dataloader:
                        x_in_s, y_s = self.model.set_input(data)
                        out_v, out_scores, pre_binary = self.model.val_score()
                        recall, pre, acc = score_split(out_scores)
                        recall_avg = recall_avg + recall
                        pre_avg = pre_avg + pre
                        acc_avg = acc_avg + acc
                        _iter += 1
                    print(f'召回率： {round(recall_avg / _iter,3)}')
                    print(f'precision： {round(pre_avg / _iter,3)}')
                    print(f'ACC: {round(acc_avg / _iter,3)}')
                    writer.add_scalars("acc", {'Valid': acc_avg / _iter, "Train": acc_average/len(self.train_dataloader)}, epoch+1)
                    writer.add_scalars("recall", {'Valid': recall_avg / _iter, "Train": recall_average / len(self.train_dataloader)}, epoch + 1)
                    writer.add_scalars("pre", {'Valid': pre_avg / _iter, "Train": pre_average / len(self.train_dataloader)}, epoch + 1)

                    writer.add_image(f'Img_val/img', x_in_s[0, [2, 3, 1], ...], epoch+1, dataformats='CHW')
                    writer.add_image(f'Img_val/pre_label', out_v[0, [0], ...], epoch+1, dataformats='CHW')
                    writer.add_image(f'Img_val/label_binary',pre_binary[0, [0], ...], epoch + 1, dataformats='CHW') # pre_binary[0, [0], ...]
                    writer.add_image(f'Img_val/label', y_s[0, [0], ...], epoch+1, dataformats='CHW')


                    if acc_avg / _iter >= best_score:  # save best model
                        best_score = acc_avg / _iter  # Just use PSNR score
                        # if PSNR_4 > best_score:  # save best model
                        # 	best_score = PSNR_4   # Just use PSNR score v3_ssim use
                        # self.model._best_checkpoint(epoch+1)
                        self.model.save_best_checkpoint(epoch+1)
                    self.model.seg_net.train()


            if (epoch + 1) % self.opts.save_freq == 0:
                print('save model ...')
                # checkpoint_dict = {
                #     'epoch': epoch,
                #     'model_state_dict': self.model.state_dict(),
                #     'optim_state_dict': self.optimizer.state_dict()
                # }
                # torch.save(checkpoint_dict,
                #            os.path.join(args.checkpoint_dir, f'{args.dataset}_' + f'epoch{epoch + 1}' + '.pth'))
                self.model.save_checkpoint(epoch+1)

            # if epoch == self.opts.max_epochs - 1:
            #     self.model.save_checkpoint(epoch)

class Generic_Train_2():
    def __init__(self, model, opts, train_dataloader, val_dataloader):
        self.model=model
        self.opts=opts      # train_single定义的一些参数
        self.train_dataloader=train_dataloader
        self.val_dataloader=val_dataloader

    def train(self):
        prev_time = time.time()
        # instantiate tensorboard logger
        writer = SummaryWriter(self.opts.save_model_dir, comment='test_your_comment')   # 创建日志记录器，用于记录训练中的指标，以便可视化
        best_score = 0
        if self.opts.resume_epoch > 0:
            start_epoch = self.opts.resume_epoch
            self.model.load_resume_epoch()
        else:
            start_epoch = 0

        for epoch in range(start_epoch, self.opts.max_epochs):
            # self.train_dataloader.sampler.set_epoch(epoch)
            log_loss = 0
            # log_loss2 = 0
            acc_average = 0
            recall_average = 0
            pre_average = 0

            # lr_scheduler = torch.optim.lr_scheduler.StepLR(self.model.optimizer_G,
            #                                                       step_size=self.opts.lr_step, gamma=0.5)
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.model.optimizer_G, 'min',
                                                                      factor=0.5, patience=2, min_lr=1e-6)

            for i, data in enumerate(self.train_dataloader):
                start_time = time.time()        # 训练开始时间

                # 调用模型的set_input方法，将输入数据data分解成多个变量，输入特征、目标标签、位置信息
                x_in_s, x_in_b, y_s, y_b, location_map = self.model.set_input(data)
                # print('x_in_s:', type(x_in_s))
                # grid_img = make_grid(x_in_s, nrow=2)
                # plt.imshow(grid_img.permute(1,2,0).cpu().detach().numpy())
                # plt.axis('off')
                # plt.show()

                # batch_loss, out, y_g, out_g = self.model.optimize_parameters()
                # 调用 optimize_parameters 方法进行模型的前向传播和反向传播，计算损失并更新模型参数
                batch_loss, pre_lab_s, scores, pre_binary = self.model.optimize_parameters(x_in_s, x_in_b, y_s, y_b, location_map)

                log_loss = log_loss + batch_loss    # 将当前批次的损失值累加到log_loss中

                recall, pre, acc = score_split(scores)   # 调用score_split函数，基于模型输出的scores计算评价指标，召回率、总体准确率、准确率
                acc_average = acc_average + acc
                recall_average = recall_average + recall
                pre_average = pre_average + pre

                batches_done = epoch * len(self.train_dataloader) + i   # 计算当前已经处理的批次数，len(self.train_dataloader)是每个轮次的批次数，i是当前批次
                batches_left = self.opts.max_epochs * len(self.train_dataloader) - batches_done     # 计算剩余的批次数
                time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))    # 估算剩余训练时间

                print('epoch', epoch+1, 'iteration', i, '  ', 'loss', round(batch_loss, 3), '  ', 'RECALL', round(recall,3), '  ', 'Pre',round(pre,3),
                      '  ', 'ACC',round(acc,3), '  ','use time',round(time.time() - start_time,3), 'ETA', time_left)

                prev_time = time.time()

            lr_scheduler.step(log_loss)

            writer.add_scalar("train_loss", log_loss/len(self.train_dataloader), epoch+1)
            writer.add_scalar("train_acc", acc_average/len(self.train_dataloader), epoch+1)
            writer.add_image(f'Img_train/img', x_in_s[0, [1,2,3], ...], epoch+1, dataformats='CHW')
            writer.add_image(f'Img_train/pre_label', pre_lab_s[0, [0], ...], epoch+1, dataformats='CHW')  #pre_lab_s[0, [0], ...]
            writer.add_image(f'Img_train/label_binary', pre_binary[0, [0], ...], epoch + 1, dataformats='CHW')
            writer.add_image(f'Img_train/label', y_s[0, [0], ...], epoch+1, dataformats='CHW')

            if (epoch + 1) % self.opts.val_freq == 0 or epoch == 1:
                print("validation...")
                self.model.seg_net.eval()
                with torch.no_grad():
                    _iter = 0
                    recall_avg = 0
                    pre_avg = 0
                    acc_avg = 0
                    for data in self.val_dataloader:
                        x_in_s, x_in_b, y_s, y_b, location_map = self.model.set_input(data)
                        out_v, out_scores, pre_binary = self.model.val_score(x_in_s, x_in_b, y_s, y_b, location_map)
                        recall, pre, acc = score_split(out_scores)
                        recall_avg = recall_avg + recall
                        pre_avg = pre_avg + pre
                        acc_avg = acc_avg + acc
                        _iter += 1
                    print(f'召回率： {round(recall_avg / _iter,3)}')
                    print(f'precision： {round(pre_avg / _iter,3)}')
                    print(f'ACC: {round(acc_avg / _iter,3)}')
                    writer.add_scalars("acc", {'Valid': acc_avg / _iter, "Train": acc_average/len(self.train_dataloader)}, epoch+1)
                    writer.add_scalars("recall", {'Valid': recall_avg / _iter, "Train": recall_average / len(self.train_dataloader)}, epoch + 1)
                    writer.add_scalars("pre", {'Valid': pre_avg / _iter, "Train": pre_average / len(self.train_dataloader)}, epoch + 1)

                    writer.add_image(f'Img_val/img', x_in_s[0, [2, 3, 1], ...], epoch+1, dataformats='CHW')
                    writer.add_image(f'Img_val/pre_label', out_v[0, [0], ...], epoch+1, dataformats='CHW')
                    writer.add_image(f'Img_val/label_binary',pre_binary[0, [0], ...], epoch + 1, dataformats='CHW') # pre_binary[0, [0], ...]
                    writer.add_image(f'Img_val/label', y_s[0, [0], ...], epoch+1, dataformats='CHW')


                    if acc_avg / _iter >= best_score:  # save best model
                        best_score = acc_avg / _iter  # Just use PSNR score
                        # if PSNR_4 > best_score:  # save best model
                        # 	best_score = PSNR_4   # Just use PSNR score v3_ssim use
                        # self.model._best_checkpoint(epoch+1)
                        self.model.save_best_checkpoint(epoch+1)
                    self.model.seg_net.train()


            if (epoch + 1) % self.opts.save_freq == 0:
                print('save model ...')
                # checkpoint_dict = {
                #     'epoch': epoch,
                #     'model_state_dict': self.model.state_dict(),
                #     'optim_state_dict': self.optimizer.state_dict()
                # }
                # torch.save(checkpoint_dict,
                #            os.path.join(args.checkpoint_dir, f'{args.dataset}_' + f'epoch{epoch + 1}' + '.pth'))
                self.model.save_checkpoint(epoch+1)

            # if epoch == self.opts.max_epochs - 1:
            #     self.model.save_checkpoint(epoch)


class Train():
    def __init__(self, model, opts, train_dataloader, val_dataloader):
        self.model=model
        self.opts=opts      # train_single定义的一些参数
        self.train_dataloader=train_dataloader
        self.val_dataloader=val_dataloader

    def train(self):
        prev_time = time.time()
        # instantiate tensorboard logger
        writer = SummaryWriter(self.opts.save_model_dir, comment='test_your_comment')   # 创建日志记录器，用于记录训练中的指标，以便可视化
        # total_steps = 0
        # log_loss = 0
        best_score = 0
        if self.opts.resume_epoch > 0:
            start_epoch = self.opts.resume_epoch
            self.model.load_resume_epoch()
        else:
            start_epoch = 0

        for epoch in range(start_epoch, self.opts.max_epochs):
            # self.train_dataloader.sampler.set_epoch(epoch)
            log_loss = 0
            log_loss2 = 0
            acc_average = 0
            recall_average = 0
            pre_average = 0
            # linear_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.model.optimizer_G,
            #                                                       step_size=self.opts.lr_step, gamma=0.5)
            linear_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.model.optimizer_G,
                                                                  mode='min', patience=2, factor=0.5, min_lr=1e-6)

            for i, data in enumerate(self.train_dataloader):
                start_time = time.time()        # 训练开始时间
                # 调用模型的set_input方法，将输入数据data分解成多个变量，输入特征、目标标签、位置信息
                x_in_s, y_s = self.model.set_input(data)
                # print('y_s:', y_s.shape, y_s)   # [8, 1, 384, 384]

                # 调用 optimize_parameters 方法进行模型的前向传播和反向传播，计算损失并更新模型参数
                batch_loss, pre_lab_s, scores, pre_binary = self.model.optimize_parameters()
                log_loss = log_loss + batch_loss    # 将当前批次的损失值累加到log_loss中
                recall, pre, acc = score_split(scores)   # 调用score_split函数，基于模型输出的scores计算评价指标，召回率、总体准确率、准确率
                acc_average = acc_average + acc
                recall_average = recall_average + recall
                pre_average = pre_average + pre

                batches_done = epoch * len(self.train_dataloader) + i   # 计算当前已经处理的批次数，len(self.train_dataloader)是每个轮次的批次数，i是当前批次
                batches_left = self.opts.max_epochs * len(self.train_dataloader) - batches_done     # 计算剩余的批次数
                time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))    # 估算剩余训练时间

                print('epoch', epoch+1, 'iteration', i, '  ', 'loss', round(batch_loss, 3), '  ', 'RECALL', round(recall,3), '  ', 'Pre',round(pre,3),
                      '  ', 'ACC',round(acc,3), '  ','use time',round(time.time() - start_time,3), 'ETA', time_left)

                prev_time = time.time()

            linear_lr_scheduler.step(log_loss)

            writer.add_scalar("train_loss", log_loss/len(self.train_dataloader), epoch+1)        # 将标量添加到summary
            writer.add_scalar("train_acc", acc_average/len(self.train_dataloader), epoch+1)
            writer.add_image(f'Img_train/img', x_in_s[0, [1,2,3], ...], epoch+1, dataformats='CHW')
            # writer.add_image(f'Img_train/pre_label', pre_lab_s[0, [0], ...], epoch+1, dataformats='CHW')  #pre_lab_s[0, [0], ...]
            writer.add_image(f'Img_train/label_binary', pre_binary[0, [0], ...], epoch + 1, dataformats='CHW')
            writer.add_image(f'Img_train/label', y_s[0, [0], ...], epoch+1, dataformats='CHW')

            if (epoch + 1) % self.opts.val_freq == 0:
                print("validation...")
                self.model.seg_net.eval()
                with torch.no_grad():
                    _iter = 0
                    recall_avg = 0
                    pre_avg = 0
                    acc_avg = 0
                    for data in self.val_dataloader:
                        x_in_s, y_s = self.model.set_input(data)
                        out_v, out_scores, pre_binary = self.model.val_score()
                        recall, pre, acc = score_split(out_scores)
                        recall_avg = recall_avg + recall
                        pre_avg = pre_avg + pre
                        acc_avg = acc_avg + acc
                        _iter += 1
                    print(f'召回率： {round(recall_avg / _iter,3)}')
                    print(f'precision： {round(pre_avg / _iter,3)}')
                    print(f'ACC: {round(acc_avg / _iter,3)}')
                    writer.add_scalars("acc", {'Valid': acc_avg / _iter, "Train": acc_average/len(self.train_dataloader)}, epoch+1)
                    writer.add_scalars("recall", {'Valid': recall_avg / _iter, "Train": recall_average / len(self.train_dataloader)}, epoch + 1)
                    writer.add_scalars("pre", {'Valid': pre_avg / _iter, "Train": pre_average / len(self.train_dataloader)}, epoch + 1)

                    writer.add_image(f'Img_val/img', x_in_s[0, [2, 3, 1], ...], epoch+1, dataformats='CHW')
                    writer.add_image(f'Img_val/pre_label', out_v[0, [0], ...], epoch+1, dataformats='CHW')
                    writer.add_image(f'Img_val/label_binary',pre_binary[0, [0], ...], epoch + 1, dataformats='CHW') # pre_binary[0, [0], ...]
                    writer.add_image(f'Img_val/label', y_s[0, [0], ...], epoch+1, dataformats='CHW')


                    if acc_avg / _iter >= best_score:  # save best model
                        best_score = acc_avg / _iter  # Just use PSNR score

                        self.model.save_best_checkpoint(epoch+1)
                    self.model.seg_net.train()


            if (epoch + 1) % self.opts.save_freq == 0:
                print('save model ...')
                self.model.save_checkpoint(epoch+1)

