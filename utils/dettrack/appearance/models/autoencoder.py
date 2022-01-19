"""
    自编码器外观特征提取模型
"""

import torch
import torch.nn as nn
from torch.tensor import Tensor

class AutoEncoder(nn.Module):
    """
        自编码器
    """
    def __init__(self, img_size: int, batch_size:int = 1, trainable: bool = False) -> None:
        super().__init__()

        self.img_size = img_size
        self.trainable = trainable
        self.batch_size = batch_size

        # encoder
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1)
        self.leaky1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.leaky2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2)
        self.leaky3 = nn.LeakyReLU()
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.liner1 = nn.Linear(in_features=128, out_features=128, bias=True)
        self.sigmoid1 = torch.nn.Sigmoid()

        # decoder
        self.liner2 = nn.Linear(in_features=128, out_features=3136, bias=True)
        self.leaky5 = nn.LeakyReLU()
        self.conv5 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.leaky6 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(3)
        self.conv6 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.sigmoid2 = torch.nn.Sigmoid()


    def forward(self, x: Tensor) -> Tensor:
        # encoder
        x = self.conv1(x)
        x = self.leaky1(x)

        x = self.conv2(x)
        x = self.leaky2(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = self.leaky3(x)

        x = self.conv4(x)
        x = self.bn2(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.liner1(x)
        # x = self.sigmoid1(x)
        
        # decoder
        if(self.trainable is True):
            x = self.liner2(x)
            x = self.leaky5(x)
            x = torch.reshape(x, (self.batch_size, 1, 56, 56))
            x = self.conv5(x)
            x = self.leaky6(x)
            x = self.bn3(x)
            x = self.conv6(x)
            x = self.sigmoid2(x)

        return x


if __name__ == "__main__":

    import torchvision.transforms as transforms

    transform = transforms.Compose({
        transforms.Resize((56, 56)),
        transforms.ToTensor()
    })

    model = AutoEncoder(True)

    from PIL import Image

    img = Image.open("/home/linsi/VisDrone-Experiences/Datasets/QR-Dataset/train-dataset/images-qrcode/0.jpg").convert("RGB")

    img = transform(img)
    
    img = torch.unsqueeze(img, 0)

    out = model(img)