
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# 设置字体为 SimHei (黑体)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    # 训练曲线可视化代码
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='训练损失')
    plt.plot(epochs, val_losses, label='验证损失')
    plt.xlabel('周期')
    plt.ylabel('损失')
    plt.title('训练与验证损失')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label='训练准确率')
    plt.plot(epochs, val_accs, label='验证准确率')
    plt.xlabel('周期')
    plt.ylabel('准确率')
    plt.title('训练与验证准确率')
    plt.legend()

    plt.show()

def plot_confusion_matrix(true_labels, pred_labels):
    # 混淆矩阵可视化代码
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('预测值')
    plt.ylabel('真实值')
    plt.title('混淆矩阵')
    plt.show()
