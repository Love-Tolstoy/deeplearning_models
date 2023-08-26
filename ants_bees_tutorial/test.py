import os

import torch
import torchvision
from PIL import Image
from torch import nn

root_dir = "test_dataset"
img_dir = "test3.png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_item_dir = os.path.join(root_dir, img_dir)
img = Image.open(img_item_dir)

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
            nn.Linear(in_features=32, out_features=2),
        )
    def forward(self, x):
        y = self.hidden_layer(x)
        y = self.output_layer(y)
        return y

tran_pose = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(32, 32)),
    torchvision.transforms.RandomCrop(size=32, padding=4),  # 经过Resize后的图像不是已经达到32X32了吗，为什么还需要裁剪，且为什么裁剪会生效
    torchvision.transforms.ToTensor(),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

ants_bees_dict = {
    0: "蚂蚁",
    1: "蜜蜂"
}

my_net = torch.load("ants_bees_10_acc_0.7189542651176453")
img = tran_pose(img)
img = torch.reshape(img, (1, 3, 32, 32))
img = img.to(device)
output = my_net(img)
print(output.shape)
print(output)
print(ants_bees_dict[output.argmax(axis=1).item()])

