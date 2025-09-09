import torch
import torch.nn as nn
from torch.nn import init
import os
import numpy as np
# from ptflops import get_model_complexity_info
from torch.optim import lr_scheduler
import random

class ModelBase(nn.Module):
    def save_network(self, network, optimizer, epoch, lr_scheduler, save_dir):
        checkpoint = {
            "network": network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch
        }
        save_dir = save_dir + '/model_pth/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_filename = '%s_net_Seg.pth' % (str(epoch))
        save_path = os.path.join(save_dir, save_filename)
        torch.save(checkpoint, save_path)

    def save_best_network(self, network, optimizer, epoch, lr_scheduler, save_dir):
        checkpoint = {
            "network": network.state_dict(),
            "network": network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch
        }
        save_dir = save_dir + '/model_pth/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_filename = 'best_%s_net_Seg.pth' % (str(epoch))
        save_path = os.path.join(save_dir, save_filename)
        torch.save(checkpoint, save_path)

    def load_dict_param(self, network, optimizer, path):
        checkpoint = torch.load(path)
        print('checkpoint loaded', checkpoint.keys())
        model_dict = network.state_dict()
        pre_dict = checkpoint['network']     # model network
        pre_dict = {k: v for k, v in pre_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pre_dict)
        network.load_state_dict(model_dict)

        # optimizer.load_state_dict(checkpoint['optimizer'])

        # network.load_state_dict(checkpoint['network'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    def print_networks(self, network, opt):
        print('---------- Networks initialized -------------')
        num_params = 0
        for param in network.parameters():
            num_params += param.numel()
        print('[Network] Total number of parameters : %.3f M' % (num_params / 1e6))
        file_name = os.path.join(opt.save_model_dir, 'opt.txt')
        with open(file_name, 'a') as opt_file:
            opt_file.write('[Network %s] Total number of parameters : %.3f M' % (opt.model, num_params / 1e6))
            opt_file.write('\n')
        print('-----------------------------------------------')

    def update_lr(self):
        self.lr_scheduler.step()
        for param_group in self.optimizer_G.param_groups:
            print('optimizer_G_lr', param_group['lr'])

    def init_weights(self, net, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                init.normal_(m.weight.data, 1.0, gain)
                init.constant_(m.bias.data, 0.0)

        print('initialize network with %s' % init_type)
        return net.apply(init_func)


def print_options(opts):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opts).items()):
        message += '{:>25}: {:<30}\n'.format(str(k), str(v))
    message += '----------------- End -------------------'
    print(message)

    # save to the disk
    if not os.path.exists(opts.save_model_dir):
        os.makedirs(opts.save_model_dir)
    file_name = os.path.join(opts.save_model_dir, 'opt.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

