import os

import torch
import torchvision
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_root_dir = "dataset/train"
train_ants_dir = "ants"
train_bees_dir = "bees"
test_root_dir = "dataset/val"
test_ants_dir = "ants"
test_bees_dir = "bees"

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
        img = Image.open(img_item_path).convert("RGB")  # 如果不加convert就会报错RuntimeError
        img = tran_pose(img)  # 将PIL图片文件经过变换转变为tensor类型
        # img = torch.reshape(input=img, shape=(-1, 3, 32, 32))
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
    def forward(self, x):
        y = self.hidden_layer(x)
        y = self.output_layer(y)
        return y


my_net = MyNet()
my_net = my_net.to(device)

train_ants_dataset = MyDataset(train_root_dir, train_ants_dir)
train_bees_dataset = MyDataset(train_root_dir, train_bees_dir)
train_datasets = train_ants_dataset + train_bees_dataset
test_ants_dataset = MyDataset(test_root_dir, test_ants_dir)
test_bees_dataset = MyDataset(test_root_dir, test_bees_dir)
test_datasets = test_ants_dataset + test_bees_dataset

train_data_load = DataLoader(train_datasets, batch_size=8, shuffle=True, drop_last=True)
test_data_load = DataLoader(test_datasets, batch_size=8, shuffle=True, drop_last=True)
print(train_data_load.batch_sampler)

train_data_size = train_datasets.__len__()
test_data_size = test_datasets.__len__()

# print(my_net)
print(f"训练集长度为:{train_data_size}")
print(f"测试集长度为:{test_data_size}")

loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

learning_rate = 1e-3
optim = torch.optim.Adam(my_net.parameters(), lr=learning_rate)

train_steps = 0

epochs = 50

if __name__ == '__main__':
    for i in range(epochs):
        print(f"------第{i+1}次训练------")
        my_net.train()
        for data in train_data_load:
            images, targets = data
            images = images.to(device)
            targets = targets.to(device)

            outputs = my_net(images)
            loss = loss_fn(outputs, targets)
            optim.zero_grad()
            loss.backward()
            optim.step()

            train_steps += 1
            if train_steps % 10 == 0:
                print(f"训练次数:{train_steps}, loss:{loss}")

        my_net.eval()
        accuarcy = 0
        accuarcy_total = 0
        with torch.no_grad():
            for data in test_data_load:
                images, targets = data
                images = images.to(device)
                targets = targets.to(device)

                outputs = my_net(images)
                accuarcy = (outputs.argmax(axis=1) == targets).sum()
                accuarcy_total += accuarcy
            print(f"第{i+1}轮训练的准确率为: {accuarcy_total / test_data_size}")
            torch.save(my_net, f"ants_bees_{i + 1}_acc_{accuarcy_total/ test_data_size}")





