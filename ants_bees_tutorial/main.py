import os

import torch
import torchvision
from PIL import Image
from torch import nn
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_root_dir = "dataset/train"
train_ants_dir = "ants"
train_bees_dir = "bees"

tran_pose = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(32, 32)),
    torchvision.transforms.RandomCrop(size=32, padding=4),  # 经过Resize后的图像不是已经达到32X32了吗，为什么还需要裁剪，且为什么裁剪会生效
    torchvision.transforms.ToTensor(),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class MyDataset(Dataset):
    def __init__(self, root_dir, label_dir):
        super(MyDataset, self).__init__()
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):  # 所以这个getitem的作用就是输出指定idx的图片和标签
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)  # 图片的路径
        img = Image.open(img_item_path)
        img = tran_pose(img)  # 将PIL图片文件经过变换转变为tensor类型
        label = self.label_dir
        if label == "ants":
            label = torch.tensor(0)
        else:
            label = torch.tensor(1)
        return img, label

    def __len__(self):
        return len(self.img_path)

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.hidden_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=32),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=64),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=128)
        )
        self.output_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=128 * 4 * 4, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(in_features=1024, out_features=256),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(in_features=256, out_features=32),
            nn.Linear(in_features=32, out_features=2)
        )


mydataset = MyDataset(train_root_dir, train_ants_dir)
print(mydataset.path)
print(mydataset.img_path)
print(mydataset.__getitem__(123)[0])
print(mydataset.__len__())
print(mydataset.__getitem__(123)[1])



