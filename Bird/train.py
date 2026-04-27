import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import os
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# --- 系统级超参数与路径设置 ---
DATA_DIR = './train'
RESULT_DIR = './result'
BATCH_SIZE = 8        # EfficientNet-B4 参数量大且分辨率高，建议设为 8 防止 OOM
EPOCHS = 30           
NUM_CLASSES = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    """固定随机种子，消除玄学波动，保证实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def plot_learning_curve(train_losses, val_losses, val_accuracies):
    os.makedirs(RESULT_DIR, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='k')
    ax1.tick_params('y', colors='k')
    ax1.legend(loc='upper left')
    ax2 = ax1.twinx()
    ax2.plot(epochs, val_accuracies, 'g--', label='Validation Accuracy (%)')
    ax2.set_ylabel('Accuracy (%)', color='g')
    ax2.tick_params('y', colors='g')
    ax2.legend(loc='upper right')
    plt.title('Learning Curve (EfficientNet-B4 + Erasing + Label Smoothing)')
    save_path = os.path.join(RESULT_DIR, 'learning_curve.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存学习曲线 -> {save_path}")

def evaluate_and_plot_metrics(model, val_loader):
    print("开始生成实验报告所需图表...")
    os.makedirs(RESULT_DIR, exist_ok=True)
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)
            
    all_labels, all_preds, all_probs = np.array(all_labels), np.array(all_preds), np.array(all_probs)

    # 1. 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(40, 40))
    sns.heatmap(cm, annot=False, cmap='Blues', cbar=False)
    plt.title('Confusion Matrix (EfficientNet-B4)', fontsize=30)
    plt.xlabel('Predicted Label', fontsize=20)
    plt.ylabel('True Label', fontsize=20)
    save_path_cm = os.path.join(RESULT_DIR, 'confusion_matrix.png')
    plt.savefig(save_path_cm, dpi=300, bbox_inches='tight')
    plt.close()

    # 2. ROC曲线
    labels_bin = label_binarize(all_labels, classes=range(NUM_CLASSES))
    fpr, tpr, roc_auc = dict(), dict(), dict()
    for i in range(NUM_CLASSES):
        if len(np.unique(labels_bin[:, i])) > 1:
            fpr[i], tpr[i], _ = roc_curve(labels_bin[:, i], all_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(NUM_CLASSES) if i in fpr]))
    mean_tpr = np.zeros_like(all_fpr)
    valid_classes = 0
    for i in range(NUM_CLASSES):
        if i in fpr:
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            valid_classes += 1
    
    if valid_classes > 0:
        mean_tpr /= valid_classes
        macro_auc = auc(all_fpr, mean_tpr)
        plt.figure(figsize=(10, 8))
        plt.plot(all_fpr, mean_tpr, color='darkorange', lw=2, label=f'Macro-average ROC curve (AUC = {macro_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multi-class ROC Curve (EfficientNet-B4)')
        plt.legend(loc="lower right")
        save_path_roc = os.path.join(RESULT_DIR, 'roc_curve.png')
        plt.savefig(save_path_roc, dpi=300, bbox_inches='tight')
        plt.close()

def main():
    set_seed(42)

    # 【适配 B4】: 提升分辨率至 380x380 (B4的黄金分辨率)
    train_transform = transforms.Compose([
        transforms.Resize(400),
        transforms.RandomCrop(380),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)), # 随机擦除防过拟合
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(400),
        transforms.CenterCrop(380),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(DATA_DIR)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_dataset.dataset = copy.copy(full_dataset)
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 【核心更换】：加载 EfficientNet-B4
    model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
    
    # 替换分类头 (EfficientNet 的分类头结构与 ResNet 不同)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)
    model = model.to(DEVICE)

    # 标签平滑
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # 【适配 B4】：学习率设为 5e-5，比 ResNet 更加保守，保证初始训练稳定
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    print(f"总数据量: {len(full_dataset)} | 训练集: {train_size} | 验证集: {val_size}")
    print("架构: EfficientNet-B4 | 策略: 380x380高分辨率 + RandomErasing + Label Smoothing")
    
    best_acc = 0.0
    history_train_loss, history_val_loss, history_val_acc = [], [], []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        train_loss = running_loss / train_size
        epoch_val_loss = val_loss / val_size
        epoch_val_acc = correct / total * 100
        
        history_train_loss.append(train_loss)
        history_val_loss.append(epoch_val_loss)
        history_val_acc.append(epoch_val_acc)
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] LR: {current_lr:.6f} | Train Loss: {train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")

        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            torch.save(model.state_dict(), 'model.pth')
            print(f"  --> 🚀 已保存当前最高准确率模型: {best_acc:.2f}%")

    print("\n训练环节结束。最高验证集准确率:", round(best_acc, 2), "%")
    print("开始绘制图表...")
    plot_learning_curve(history_train_loss, history_val_loss, history_val_acc)
    
    model.load_state_dict(torch.load('model.pth'))
    evaluate_and_plot_metrics(model, val_loader)
    print("所有任务完成！")

if __name__ == '__main__':
    main()