import torch
import torch.nn as nn
import torch.optim as optim
from model import Search_Model
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
import json
import os
import argparse

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cos_sim(a, b):
    if len(a.shape) == 1:
        a = a.unsqueeze(0)
    if len(b.shape) == 1:
        b = b.unsqueeze(0)
    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))

parser = argparse.ArgumentParser(description='Train the model')
parser.add_argument('-dataset', default='dataset', help='Dataset directory')
parser.add_argument("-info_path", default="info.json", help="Path to the info file (default: info.json)")
parser.add_argument('-train_log', default='train_log.csv', help='Path to the training log file')
parser.add_argument('-pre_train', default=None, help='Path to pre-trained weights')
parser.add_argument('-model_path', default='models', help='Path to save models')
parser.add_argument('-train_batch_size', type=int, default=256, help='Batch size for training')
parser.add_argument('-val_batch_size', type=int, default=256, help='Batch size for validation')
parser.add_argument('-lr', type=float, default=0.001, help='Initial learning rate')
parser.add_argument('-epochs', type=int, default=50, help='Number of epochs for training')
parser.add_argument('-t_max', type=int, default=10, help='Number of iterations per learning rate cycle')
parser.add_argument('-eta_min', type=float, default=0.00005, help='Minimum learning rate')
args = parser.parse_args()

info = json.load(open(args.info_path, 'r', encoding="utf-8"))
dataset_path = args.dataset
train_log_path = args.train_log
pre_train = args.pre_train
model_path = args.model_path
lr = args.lr
epochs = args.epochs
t_max = args.t_max
eta_min = args.eta_min

# 定义数据转换
transform_source = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=info['mean'], std=info['std']),
])

# 定义数据转换
transform_augmented = transforms.Compose([
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(p=0.2),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=info['mean'], std=info['std']),
])

class AnimeDataset(Dataset):
    def __init__(self, root_dir, transform_source, transform_augmented, is_val=False):
        self.root_dir = root_dir
        self.transform_source = transform_source
        self.transform_augmented = transform_augmented
        self.is_val = is_val
        if is_val:
            img_list = os.listdir(root_dir)
            self.image_list = [img for img in img_list if '_val' not in img]
        else:
            self.image_list = os.listdir(root_dir)
        

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_list[idx])
        image = Image.open(img_name).convert('RGB')
        source_image = self.transform_source(image)
        if self.is_val:
            val_img = os.path.join(self.root_dir, f"{os.path.splitext(self.image_list[idx])[0]}_val.jpg")
            augmented_image = self.transform_source(Image.open(val_img).convert('RGB'))
        else:
            augmented_image = self.transform_augmented(image)
        images = torch.stack([source_image, augmented_image], dim=0)
        return images

# 从data目录下加载训练和验证数据集
train_dataset = AnimeDataset(f'{dataset_path}/train', transform_source, transform_augmented)
val_dataset = AnimeDataset(f'{dataset_path}/val', transform_source, transform_augmented, is_val=True)

# 定义数据加载器
train_batch_size = args.train_batch_size
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_batch_size = args.val_batch_size
val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=4, pin_memory=True)

model = Search_Model().to(device)

# 定义损失函数
def custom_loss(cosine_sim_matrix, scale=20):
    scores = cosine_sim_matrix * scale
    labels = torch.tensor(
        range(len(scores)), dtype=torch.long, device=scores.device
    )
    return nn.CrossEntropyLoss()(scores, labels)

def evaluate_model(model, test_loader, k_list=[1, 5, 10]):
    model.eval()
    correct_matches = {k: 0 for k in k_list}

    with torch.no_grad():
        outputs1_all = []
        outputs2_all = []
        for images in test_loader:
            images = images.to(device)
            source_images = images[:, 0, :, :, :]
            augmented_images = images[:, 1, :, :, :]

            outputs1 = model(source_images)
            outputs2 = model(augmented_images)

            outputs1_all.append(outputs1)
            outputs2_all.append(outputs2)

        cosine_sim_matrix = cos_sim(torch.cat(outputs1_all), torch.cat(outputs2_all))

        _, topk_indices = torch.topk(cosine_sim_matrix, max(k_list), dim=1, largest=True, sorted=True)
        # 计算中位数倒数排名
        reciprocal_rank = 0
        for i in range(len(val_dataset)):
            if i in topk_indices[i].tolist():
                rank = (topk_indices[i] == i).nonzero().item()
                reciprocal_rank += 1 / (rank + 1)
                for k in k_list:
                    if rank <= k:
                        correct_matches[k] += 1

    topk_accuracies = {k: correct_matches[k] / len(val_dataset) for k in k_list}
    mrr = reciprocal_rank / len(val_dataset)

    return topk_accuracies, mrr

def train():
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)

    # 使用 torch.cuda.amp 进行混合精度训练
    scaler = torch.cuda.amp.GradScaler()

    # 训练模型
    num_epochs = epochs
    best_mrr = 0.0

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch, images in tqdm(enumerate(train_loader), total=len(train_loader)):
            images = images.to(device)
            source_images = images[:, 0, :, :, :]
            augmented_images = images[:, 1, :, :, :]

            optimizer.zero_grad()

            # 使用 autocast 来混合精度训练
            with torch.cuda.amp.autocast():
                # 前向传播，计算特征向量
                outputs1 = model(source_images)
                outputs2 = model(augmented_images)

                # 计算余弦相似度矩阵
                cosine_sim_matrix = cos_sim(outputs1, outputs2)

                # 计算损失
                loss = custom_loss(cosine_sim_matrix)
                total_loss += loss.item()

            # 反向传播和优化
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # 输出每个epoch的平均损失
        print(f'Training - Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader)}')
        scheduler.step()

        # 在验证集上评估模型
        model.eval()
        topk_accuracies, mrr = evaluate_model(model, tqdm(val_loader, total=len(val_loader)), k_list=[1, 5, 10, 20])
        print(f'Validation - Epoch [{epoch+1}/{num_epochs}], Top-1 Accuracy: {topk_accuracies[1]}, Top-5 Accuracy: {topk_accuracies[5]}, Top-10 Accuracy: {topk_accuracies[10]}, Top-20 Accuracy: {topk_accuracies[20]}, MRR: {mrr}')
        # 保存模型
        torch.save(model.state_dict(), f'{model_path}/epoch_{epoch + 1}.pth')
        if mrr > best_mrr:
            best_mrr = mrr
            torch.save(model.state_dict(), f'{model_path}/best.pth')
        open(train_log_path, 'a').write(f'{epoch+1},{total_loss/len(train_loader)},{topk_accuracies[1]},{topk_accuracies[5]},{topk_accuracies[10]},{topk_accuracies[20]},{mrr}\n')

if __name__ == '__main__':
    open(args.train_log, 'w').write('epoch,loss,acc1,acc5,acc10,acc20,mrr\n')
    if args.pre_train and os.path.exists(args.pre_train):
        model.load_state_dict(torch.load(args.pre_train))
        print(f'Loaded model from {args.pre_train}')
    topk_accuracies, mrr = evaluate_model(model, tqdm(val_loader, total=len(val_loader)), k_list=[1, 5, 10, 20])
    print(f'Validation - Init, Top-1 Accuracy: {topk_accuracies[1]}, Top-5 Accuracy: {topk_accuracies[5]}, Top-10 Accuracy: {topk_accuracies[10]}, Top-20 Accuracy: {topk_accuracies[20]}, MRR: {mrr}')
    open(args.train_log, 'a').write(f'0,0,{topk_accuracies[1]},{topk_accuracies[5]},{topk_accuracies[10]},{topk_accuracies[20]},{mrr}\n')
    train()
