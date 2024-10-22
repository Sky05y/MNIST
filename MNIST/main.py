import torch
from model import CNN
from data_loader import get_data_loaders
from train import train_model, evaluate_model

def main():
    # 检查是否可以使用 CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 获取数据加载器
    train_loader, test_loader = get_data_loaders(batch_size=64)

    # 创建模型并移动到 GPU
    model = CNN().to(device)
    
    # 训练模型
    train_model(model, train_loader, test_loader, num_epochs=10)

    # 评估模型
    evaluate_model(model, test_loader, device=device)

if __name__ == "__main__":
    main()
