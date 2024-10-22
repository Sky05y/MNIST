import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from visualization import plot_training_curves, plot_confusion_matrix
from torch.utils.tensorboard import SummaryWriter

def save_model(model, path='mnist_model_best.pth'):
    """保存模型的最佳权重"""
    torch.save(model.state_dict(), path)

def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.0001):
    """训练模型，并在验证集上性能最好的情况下保存模型权重"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # 将模型移动到指定设备

    # 创建一个 TensorBoard 日志目录
    writer = SummaryWriter(log_dir='runs/mnist_experiment')  
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 存储每个 epoch 的损失和准确性
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # 用于保存最佳模型的性能
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()  # 进入训练模式
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            # 将输入数据移动到指定设备
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()  # 清零梯度
            outputs = model(images)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = correct_train / total_train
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # 记录训练损失和准确率到 TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)

        # 计算验证集的损失和准确性
        val_loss, val_acc = evaluate_model(model, val_loader, mode='validation', device=device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # 记录验证损失和准确率到 TensorBoard
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('Accuracy/validation', val_acc, epoch)

        print(f'周期 [{epoch+1}/{num_epochs}], 训练损失: {train_loss:.4f}, 训练准确率: {train_acc * 100:.2f}%, 验证损失: {val_loss:.4f}, 验证准确率: {val_acc * 100:.2f}%')

        # 保存验证集上准确率最高的模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, f'mnist_model_best.pth')  # 保存最好的模型权重

    # 训练结束后可视化训练曲线
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)

    # 关闭 TensorBoard Writer
    writer.close()
    

def evaluate_model(model, data_loader, mode='test', device='cpu'):
    """评估模型性能并在测试模式下输出额外指标"""
    
    model.eval()  # 进入评估模式
    model.to(device)  # 将模型移动到指定设备
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, labels in data_loader:
            # 将数据移动到指定设备
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())  # 从 GPU 移回 CPU
            all_predictions.extend(predicted.cpu().numpy())  # 从 GPU 移回 CPU
    
    loss = running_loss / len(data_loader)
    accuracy = correct / total
    
    if mode == 'test':
        # 仅在测试模式下计算并打印额外的指标和混淆矩阵
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        cm = confusion_matrix(all_labels, all_predictions)

        print(f'Accuracy: {accuracy * 100:.2f}%')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print('Confusion Matrix:')
        print(cm)
        
        # 调用可视化混淆矩阵
        plot_confusion_matrix(all_labels, all_predictions)
    
    return loss, accuracy
