import argparse
import torch
import json
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def main(dataset_path='dataset', output='info.json', batch_size=64, max_samples=1000):
    # 定义数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # 假设你有一个名为 dataset 的数据集对象
    dataset = datasets.ImageFolder(dataset_path, transform=transform)

    # 定义 DataLoader 分批次加载数据
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    mean_sum = 0.0
    std_sum = 0.0
    total_samples = 0
    for data, _ in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean_sum += data.mean(2).sum(0)
        std_sum += data.std(2).sum(0)
        total_samples += batch_samples
        if total_samples > max_samples:
            break

    # 计算全局均值和标准差
    mean = mean_sum / total_samples
    std = std_sum / total_samples

    data = {
        'mean':mean.numpy().tolist(),
        'std': std.numpy().tolist()
    }

    with open(output, 'w') as f:
        f.write(json.dumps(data))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate mean and standard deviation of a dataset.')
    parser.add_argument('-dataset', default='dataset', help='Path to the dataset (default: dataset)')
    parser.add_argument('-batch_size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument("-output", default="info.json", help="Output file path (default: info.json)")
    parser.add_argument('-max_samples', type=int, default=1000, help='Maximum number of samples to use for calculation (default: 1000)')
    args = parser.parse_args()

    main(args.dataset, args.output, args.batch_size, args.max_samples)
