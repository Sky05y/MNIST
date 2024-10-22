import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
from torchvision import datasets

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # 添加 Batch Normalization 层
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # 添加 Batch Normalization 层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 需要根据输入图像的大小调整
        self.fc2 = nn.Linear(128, 10)  # 假设有10个数字

    def forward(self, x):
        x = nn.functional.relu(self.bn1(self.conv1(x)))  # 添加 Batch Normalization
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = nn.functional.relu(self.bn2(self.conv2(x)))  # 添加 Batch Normalization
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 64 * 7 * 7)  # 展平
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 加载模型
def load_model(model_path):
    model = SimpleModel()  # 实例化模型
    model.load_state_dict(torch.load(model_path), strict = False)
    model.eval()  # 设置为评估模式
    return model

# 预处理图像
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # 调整大小
        transforms.ToTensor(),  # 转换为Tensor
        transforms.Normalize((0.5,), (0.5,))  # 归一化
    ])
    return transform(image).unsqueeze(0)  # 添加batch维度

# 进行预测
def predict(model, image_tensor):
    with torch.no_grad():  # 不需要计算梯度
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)  # 获取最大值的索引
        return predicted.item()  # 返回预测的数字

# 加载MNIST数据集
def load_mnist_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转为Tensor
        transforms.Normalize((0.5,), (0.5,))  # 归一化
    ])
    # 下载并加载测试集数据
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return mnist_test

# 选择MNIST数据集中的图像
def select_image_from_mnist(dataset, index):
    image, label = dataset[index]
    return image, label

# 主程序
if __name__ == "__main__":
    model_path = "mnist_model_best.pth"  # 模型路径
    mnist_dataset = load_mnist_dataset()  # 加载MNIST数据集
    for i in range(5):          
        index = int(input("请输入要识别的图像在MNIST测试集中的索引 (0-9999): "))  # 用户输入索引
        image, true_label = select_image_from_mnist(mnist_dataset, index)  # 选择图像

        model = load_model(model_path)  # 加载模型
        image_tensor = image.unsqueeze(0)  # 添加batch维度
        predicted_digit = predict(model, image_tensor)  # 进行预测

        print(f"预测结果为: {predicted_digit}, 实际标签为: {true_label}")  # 打印预测结果和实际标签
