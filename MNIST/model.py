import torch.nn as nn
import torch.nn.functional as F

# CNN 模型定义
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 卷积层1，输入1通道，输出32通道，卷积核3x3
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        # 批归一化1
        self.bn1 = nn.BatchNorm2d(32)
        # 最大池化层1
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 卷积层2，输入32通道，输出64通道，卷积核3x3
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # 批归一化2
        self.bn2 = nn.BatchNorm2d(64)
        # 最大池化层2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout
        self.dropout = nn.Dropout(0.5)

        # 全连接层1，输入64*7*7，输出128
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # 全连接层2，输入128，输出10（分类数）
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # 卷积1 -> 批归一化 -> 激活 -> 池化
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        # 卷积2 -> 批归一化 -> 激活 -> 池化
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # 展平
        x = x.view(-1, 64 * 7 * 7)
        # 全连接层1 -> 激活
        x = F.relu(self.fc1(x))
        # Dropout 应用在全连接层之间
        x = self.dropout(x)
        # 输出层
        x = self.fc2(x)
        return x
