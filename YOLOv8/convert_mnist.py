import os
import numpy as np
from PIL import Image

def read_idx(filename):
    with open(filename, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        count = int.from_bytes(f.read(4), 'big')
        if magic == 2049:
            return np.frombuffer(f.read(), np.uint8, count)
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')
        return np.frombuffer(f.read(), np.uint8, count*rows*cols).reshape(count, rows, cols)

# 路径配置
raw_path = "mnist/MNIST/raw"
save_root = "mnist_jpg"
os.makedirs(f"{save_root}/images/train", exist_ok=True)
os.makedirs(f"{save_root}/images/val", exist_ok=True)
os.makedirs(f"{save_root}/labels/train", exist_ok=True)
os.makedirs(f"{save_root}/labels/val", exist_ok=True)

# 读取MNIST二进制文件
train_imgs = read_idx(f"{raw_path}/train-images-idx3-ubyte")
train_labels = read_idx(f"{raw_path}/train-labels-idx1-ubyte")
val_imgs = read_idx(f"{raw_path}/t10k-images-idx3-ubyte")
val_labels = read_idx(f"{raw_path}/t10k-labels-idx1-ubyte")

# 导出训练集
for idx, img_arr in enumerate(train_imgs):
    label = train_labels[idx]
    img = Image.fromarray(img_arr)
    img.save(f"{save_root}/images/train/{idx}.jpg")
    # YOLO标签：数字类别 + 归一化中心坐标+宽高（28x28整张图数字）
    with open(f"{save_root}/labels/train/{idx}.txt", "w") as f:
        f.write(f"{label} 0.5 0.5 1.0 1.0\n")

# 导出验证集
for idx, img_arr in enumerate(val_imgs):
    label = val_labels[idx]
    img = Image.fromarray(img_arr)
    img.save(f"{save_root}/images/val/{idx}.jpg")
    with open(f"{save_root}/labels/val/{idx}.txt", "w") as f:
        f.write(f"{label} 0.5 0.5 1.0 1.0\n")

print("MNIST转换完成，数据集保存在 mnist_jpg")