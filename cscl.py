import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import transforms
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
import rasterio
from sklearn.metrics import accuracy_score
from model.psnet import Encoder
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import seaborn as sns
import warnings
from rasterio.errors import NotGeoreferencedWarning

# 忽略NotGeoreferencedWarning
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
from matplotlib.font_manager import FontProperties

font = FontProperties(fname="/usr/share/fonts/truetype/msttcorefonts/times.ttf")
# plt.rcParams["font.family"] = ["Times New Roman", "Times"]
# plt.plot([1,2,3])
# plt.xlabel("x label")
# plt.show()

def set_seed(seed=42):      # 设置随机种子确保可复现性
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# 数据集类 - 云（正样本）和雪（负样本）区分
class SnowDataset(Dataset):
    def __init__(self, cloud_dir, snow_dir, transform=None):
        self.cloud_dir = cloud_dir
        self.snow_dir = snow_dir
        self.transform = transforms.Compose([transforms.Lambda(lambda x: x / 255.0),
            transforms.Normalize((0.485, 0.456, 0.406, 0.5), (0.229, 0.224, 0.225, 0.25)),
            transforms.RandomHorizontalFlip(p=0.5), transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))])

        # 加载云图像（正样本）和雪图像（负样本）路径
        self.cloud_paths = [os.path.join(cloud_dir, f) for f in os.listdir(cloud_dir) if f.endswith('.tif')]
        self.snow_paths = self._get_snow_paths(snow_dir)

        # 为保证样本平衡，按数量少的类别确定总样本数，保证云和雪样本数量1:1
        self.min_num = min(len(self.cloud_paths), len(self.snow_paths))
        self.num_samples = self.min_num * 2

        print(f"Loaded {len(self.cloud_paths)} cloud images and {len(self.snow_paths)} snow images, "
              f"using {self.min_num} samples per class for balanced training")

    def _get_snow_paths(self, img_snow_dir):
        snow_images = []
        for percent_folder in os.listdir(img_snow_dir):
            snow_folder = os.path.join(img_snow_dir, percent_folder)
            if os.path.isdir(snow_folder):
                snow_image_paths = [os.path.join(snow_folder, f) for f in os.listdir(snow_folder) if f.endswith('.tif')]
                snow_images.extend(snow_image_paths)
        return snow_images

    def __len__(self):
        # 每个云图像作为锚点，需覆盖所有对比组合，此处返回总样本数（云+雪）
        return self.num_samples

    def __getitem__(self, idx):
        # 前半部分是云（锚点/正样本），后半部分是雪（负样本）
        # 交替返回云（偶数索引）和雪（奇数索引）图像，保证1:1比例
        if idx % 2 == 0:
            img_path = self.cloud_paths[idx // 2 % self.min_num]
            label = 0
        else:
            img_path = self.snow_paths[idx // 2 % self.min_num]
            label = 1

        # 读取遥感图像（多光谱处理）
        with rasterio.open(img_path) as src:
            image = src.read([4, 3, 2, 1])  # 取 RGB 通道（按需调整）
        image = torch.from_numpy(image).float()     # numpy -> tensor [0,255]
        # if self.transform:
        image = self.transform(image)

        return {'image': image, 'label': label, 'path': img_path}


# 模型定义（PSNet + MoCo 逻辑调整）
class PSNet(nn.Module):
    def __init__(self, dim=512, mlp_dim=1024, num_classes=2):
        super(PSNet, self).__init__()
        self.encoder = Encoder(n_channels=4, n_classes=num_classes)  # 基础编码器
        # 投影头（对比学习用）
        self.projection_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(dim, mlp_dim),
            nn.BatchNorm1d(mlp_dim),
            nn.GELU(), nn.Dropout(0.5),
            nn.Linear(mlp_dim, dim)
        )
        # 分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.encoder(x)          # 提取特征
        projection = self.projection_head(features)  # 对比学习投影
        cls_output = self.classifier(features)       # 分类输出
        return projection, cls_output


# 对比学习框架（云为正样本，雪为负样本）
class MoCoCloudSnow(nn.Module):
    def __init__(self, dim=512, mlp_dim=1024, temperature=0.2, num_classes=2, momentum=0.99):
        super(MoCoCloudSnow, self).__init__()
        self.temperature = temperature
        self.momentum = momentum
        self.base_encoder = PSNet(dim, mlp_dim, num_classes)  # 共享编码器
        self.momentum_encoder = PSNet(dim, mlp_dim, num_classes)

        # 初始化动量编码器权重与基础编码器一致
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # 权重初始化
            param_m.requires_grad = False  # 不更新动量编码器梯度

    @torch.no_grad()
    def _momentum_update_encoder(self):
        """动量更新编码器"""
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * self.momentum + param_b.data * (1.0 - self.momentum)

    def forward(self, images, labels):
        """
        images: 批次内所有图像
        labels: 批次内所有标签（0=云，1=雪）
        """
        # 提取基础编码器特征（带梯度）
        all_proj_base, all_cls_base = self.base_encoder(images)
        all_proj_base = F.normalize(all_proj_base, dim=1)  # 归一化特征

        # 提取动量编码器特征（无梯度）
        with torch.no_grad():
            self._momentum_update_encoder()  # 更新动量编码器
            all_proj_momentum, _ = self.momentum_encoder(images)
            all_proj_momentum = F.normalize(all_proj_momentum, dim=1)  # 归一化特征

        # 划分云和雪样本
        cloud_mask = (labels == 0)
        snow_mask = (labels == 1)
        clouds_base = all_proj_base[cloud_mask]  # 基础编码器的云特征（用于预测）
        clouds_momentum = all_proj_momentum[cloud_mask]  # 动量编码器的云特征（作为正样本目标）
        snows_momentum = all_proj_momentum[snow_mask]  # 动量编码器的雪特征（作为负样本目标）
        cloud_labels = labels[cloud_mask]

        # 确保同时有云和雪样本，否则返回分类损失
        if clouds_base.size(0) == 0 or snows_momentum.size(0) == 0:
            classification_loss = F.cross_entropy(all_cls_base, labels)
            return classification_loss, all_cls_base

        # 对比学习：云作为锚点，其他云为正样本，雪为负样本
        pos_sim = torch.matmul(clouds_base, clouds_momentum.transpose(0, 1)) / self.temperature
        neg_sim = torch.matmul(clouds_base, snows_momentum.transpose(0, 1)) / self.temperature

        # 排除自身对比（对角线元素）
        mask = ~torch.eye(clouds_base.size(0), dtype=torch.bool).to(clouds_base.device)
        pos_sim = pos_sim[mask].view(clouds_base.size(0), -1)

        # 合并正、负样本相似度
        logits = torch.cat([pos_sim, neg_sim], dim=1)
        contrastive_labels = torch.zeros(clouds_base.size(0), dtype=torch.long).to(clouds_base.device)

        # 计算对比损失和分类损失
        contrastive_loss = F.cross_entropy(logits, contrastive_labels)
        classification_loss = F.cross_entropy(all_cls_base, labels)
        loss = classification_loss + 0.5 * contrastive_loss

        return loss, all_cls_base



# 分类精度计算（可选，若需评估分类任务）
def calculate_accuracy(output, target):
    """计算分类准确率"""
    pred = torch.argmax(output, dim=1)
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    acc = accuracy_score(pred, target)
    # correct = pred.eq(target).sum().item() / target.size(0)
    return acc


def visualize_samples(dataset, num_samples=4):
    fig, axes = plt.subplots(1, num_samples, figsize=(12, 3))
    for i in range(num_samples):
        idx = np.random.randint(0, len(dataset))
        sample = dataset[idx]
        img = sample['image'].permute(1, 2, 0).numpy()  # 调整为(H,W,C)
        axes[i].imshow(img)
        axes[i].set_title(f"Label: {sample['label']}")
        axes[i].axis('off')
    plt.show()


def extract_features(model, data_loader, max_samples=200):      # 提取对比学习的投影特征
    model.eval()
    features = []
    labels = []
    total_samples = 0
    with torch.no_grad():
        for batch in data_loader:
            images = batch['image'].cuda(non_blocking=True)
            batch_labels = batch['label'].numpy()  # 真实标签（0=云，1=雪）
            # 提取对比学习的投影特征
            projections, _ = model.base_encoder(images)
            projections = F.normalize(projections, dim=1)  # 归一化特征
            # 计算本批次要添加的样本数
            samples_to_add = min(len(batch_labels), max_samples - total_samples)
            # 添加到特征和标签列表
            features.append(projections[:samples_to_add].cpu().numpy())
            labels.append(batch_labels[:samples_to_add])
            total_samples += samples_to_add
            # 如果已达到最大样本数，停止处理
            if total_samples >= max_samples:
                break
    # 拼接所有批次的特征和标签
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    print(f"Extracted features for {len(features)} samples")
    return features, labels


def visualize_tsne(features, labels, title="T-SNE Feature Space", save_dir=None):
    # t-SNE 降维到 2D
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings = tsne.fit_transform(features)
    # 绘图
    plt.figure(figsize=(8, 6))
    plt.scatter(embeddings[labels == 0, 0], embeddings[labels == 0, 1], c='mediumseagreen',
                label='Cloud (0)', alpha=0.6, s=50)
    plt.scatter(embeddings[labels == 1, 0], embeddings[labels == 1, 1], c='orange', label='Snow (1)', alpha=0.6, s=50)
    plt.title(title)
    plt.legend()
    plt.grid(False)
    save_path = os.path.join(save_dir, f"t_SNE.png")
    try:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print('t-SNE save')
    except Exception as e:
        print('t-SNE not save')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix', save_dir=None):
    """
    绘制混淆矩阵
    :param y_true: 真实标签，形状为 (n_samples,)
    :param y_pred: 预测标签，形状为 (n_samples,)
    :param classes: 类别名称列表，如 ['Cloud', 'Snow']
    :param title: 混淆矩阵标题
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm / cm.sum(axis=1, keepdims=True) * 100
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'{title} (%)')
    save_path = os.path.join(save_dir, f"confusion.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


# 平均指标计算类
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# 训练逻辑（核心：构建锚点、正样本、负样本的 batch）
def train_moco(args):
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # 初始化模型、优化器
    model = MoCoCloudSnow(dim=args.dim, mlp_dim=args.mlp_dim, temperature=args.temp, num_classes=2)
    model = model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=2, min_lr=1e-5)

    # 构建数据集：云（正）、雪（负）
    full_dataset = SnowDataset(args.cloud_dir, args.snow_dir, transform=None)
    # 划分训练集和验证集 (8:2)
    # train_size = int(0.8 * len(full_dataset))
    # val_size = len(full_dataset) - train_size
    # train_dataset, val_dataset = random_split(
    #     full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed)
    # )

    # 数据加载器
    train_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)

    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
    #                         num_workers=args.workers, pin_memory=True)

    # TensorBoard（可选）
    # writer = SummaryWriter(log_dir=args.output_dir) if args.rank == 0 else None

    best_acc = 0.6
    val_acc, val_loss = 0.0, 0.0
    start_time = time.time()

    for epoch in range(args.epochs):
        # 训练阶段
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, epoch, args)
        # 验证阶段
        # if (epoch + 1) % args.print_freq == 0:
        #     val_loss, val_acc = validate(model, val_loader, epoch, args)

        lr_scheduler.step(train_loss)

        # 保存最佳模型（基于验证集损失）
        # if (epoch + 1) % args.print_freq == 0 and val_acc > best_acc:
        #     best_acc = val_acc
        #     save_network(model, optimizer, epoch, args.output_dir)
        #     print('save model')

        # 打印日志
        # if (epoch + 1) % args.print_freq == 0:
        #     print(f"Epoch [{epoch + 1}/{args.epochs}] - "
        #           f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc * 100:.2f}% - "
        #           f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc * 100:.2f}% - "
        #           f"Best Val Acc: {best_acc * 100:.2f}%")

    print(f"Training completed in {time.time() - start_time:.2f} seconds")
    print(f"Best validation accuracy: {best_acc * 100:.2f}%")

def train_epoch(model, data_loader, optimizer, epoch, args):
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    true_list = []
    pred_list = []

    for i, batch in enumerate(data_loader):
        # 分离云（正）、雪（负）样本（需保证 batch 中有正/负样本）
        images = batch['image'].to(args.device)
        labels = batch['label'].to(args.device)

        # 前向传播
        loss, cls_output = model(images, labels)
        # print('分类输出：', cls_output)        # tensor([[-0.2033,  0.2662],
        # print('真实标签：', labels)        # tensor([0, 0, 1, 1, 0, 0, 0, 0], device='cuda:0')
        pred = torch.argmax(cls_output, dim=1)
        pred_list.extend(pred.cpu().numpy())
        true_list.extend(labels.cpu().numpy())

        # 反向传播， 在反向传播后添加梯度监控，梯度正常
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 限制梯度范数≤1.0
        optimizer.step()

        # 计算分类精度
        acc = calculate_accuracy(cls_output, labels)

        # 更新指标，这里用anchors的数量作为batch_size
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc, labels.size(0))

    # 打印训练信息
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Train Epoch [{epoch + 1}/{args.epochs}] - "
          f"Loss: {loss_meter.avg:.4f}, LR: {current_lr:.6f}, Acc: {acc_meter.avg * 100:.2f}%")
    if acc_meter.avg > 0.95:
        print("Visualizing training set t-SNE and confusion")
        train_features, train_labels = extract_features(model, data_loader, max_samples=400)
        visualize_tsne(train_features, train_labels, title="T-SNE Feature Space", save_dir=args.output_dir)
        plot_confusion_matrix(true_list, pred_list, ['cloud', 'snow'], title=f'Confusion matrix', save_dir=args.output_dir)

    return loss_meter.avg, acc_meter.avg


def validate(model, data_loader, epoch, args):
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    # true_list = []
    # pred_list = []

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            # images = batch['image'].cuda(non_blocking=True)
            # labels = batch['label'].cuda(non_blocking=True)
            images = batch['image'].to(args.device)
            labels = batch['label'].to(args.device)

            loss, cls_output = model(images, labels)

            # pred = torch.argmax(cls_output, 1)
            # pred_list.extend(pred.cpu().numpy())  # 收集预测标签
            # true_list.extend(labels.cpu().numpy())  # 收集真实标签

            acc = calculate_accuracy(cls_output, labels)
            loss_meter.update(loss.item(), images.size(0))
            acc_meter.update(acc, labels.size(0))
    # 打印验证信息
    print(f"Val Epoch [{epoch + 1}/{args.epochs}] - "
          f"Loss: {loss_meter.avg:.4f}, Acc: {acc_meter.avg * 100:.2f}%")

    # print("\nVisualizing feature space")
    # # 提取验证集特征（也可提取训练集特征）
    # features, labels = extract_features(model, data_loader, max_samples=500)
    # visualize_tsne(features, labels, title='t-SNE: Feature Space')
    # plot_confusion_matrix(true_list, pred_list, ['cloud', 'snow'], title=f'Confusion matrix')

    return loss_meter.avg, acc_meter.avg


def save_network(network, optimizer, epoch, save_dir):
    save_path = os.path.join(save_dir, f'{epoch+1}_net_Seg.pth')
    torch.save({
        "network": network.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch
    }, save_path)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Cloud-Snow Contrastive Learning')

    # 数据路径
    parser.add_argument('--cloud_dir', type=str,
                        default='/mnt/894a6e1e-2772-4477-b417-173963f8ccec/dataset/new_2024/GF-2/images',
                        help='Path to cloud images directory')
    parser.add_argument('--snow_dir', type=str,
                        default='/mnt/894a6e1e-2772-4477-b417-173963f8ccec/dataset/SnowData/SnowImageSubset',
                        help='Path to snow images directory')
    parser.add_argument('--output_dir', type=str, default='./models/MambaOne/cscl_GF',
                        help='Output directory for checkpoints and logs')

    # 模型参数
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--dim', type=int, default=512,
                        help='Projection head output dimension')
    parser.add_argument('--mlp_dim', type=int, default=1024,
                        help='MLP hidden dimension')
    parser.add_argument('--moco-m', type=float, default=0.99,
                        help='moco momentum of updating momentum encoder (default: 0.99)')
    parser.add_argument('--moco-m-cos', action='store_true',
                        help='gradually increase moco momentum to 1 with a '
                             'half-cycle cosine schedule')
    parser.add_argument('--temp', type=float, default=0.2, help='softmax temperature (default: 0.2)')
    parser.add_argument('--weight', type=float, default=0.1, help='default: 0.2')


    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,      # Adam=0.001
                        help='Initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--warmup-epochs', type=int, default=10,
                        help='Number of warmup epochs')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of data loading workers')
    parser.add_argument('--print-freq', type=int, default=5,
                        help='Print frequency')
    parser.add_argument('--rank', type=int, default=0,
                        help='Rank for distributed training')
    parser.add_argument('--seed', type=int, default=256,
                        help='Random seed')
    parser.add_argument('--img-size', type=int, default=384,
                        help='Input image size')

    args = parser.parse_args()

    # 训练
    train_moco(args)


if __name__ == '__main__':
    main()

