import os
import argparse
from PIL import Image
from torchvision import transforms

def gen_val(val_path, img_size=224, crop_scale=0.25, flip=0.2, color=0.2):
    # 定义数据增强的转换
    data_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=(img_size, img_size), scale=(crop_scale, 1.0)),
        transforms.RandomHorizontalFlip(p=flip),
        transforms.ColorJitter(brightness=color, contrast=color, saturation=color, hue=color),
    ])

    # 遍历指定目录下的所有图片文件
    for root, _, files in os.walk(val_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')) and '_val' not in file.lower():
                # 构造文件路径
                file_path = os.path.join(root, file)

                # 打开图片
                img = Image.open(file_path)

                # 应用数据增强
                img_transformed = data_transform(img)

                # 构造保存路径
                save_path = os.path.join(root, f"{os.path.splitext(file)[0]}_val.jpg")

                # 保存增强后的图片
                img_transformed.save(save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate augmented images for validation.")
    parser.add_argument("-val_path", default='dataset/val', help="Path to the validation dataset(default: dataset/val).")
    parser.add_argument("-img_size", type=int, default=224, help="Size of the random crop (default: 224).")
    parser.add_argument("-crop_scale", type=float, default=0.25, help="Minimum random crop scale (default: 0.25).")
    parser.add_argument("-flip", type=float, default=0.2, help="Random horizontal flip probability (default: 0.2).")
    parser.add_argument("-color", type=float, default=0.2, help="Maximum random color brightness adjustment (default: 0.2).")
    
    args = parser.parse_args()

    # 调用生成函数
    gen_val(args.val_path, args.img_size, args.crop_scale, args.flip, args.color)
