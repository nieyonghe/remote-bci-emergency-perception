"""训练脚本 — SK-MMFMNet 端到端训练
流程：双模态 Dataset → SK-MMFMNet → 损失计算 → 反向传播 → 验证
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets.dual_modal_dataset import DualModalDataset
from models.sk_mmfmnet import SKMMFMNet


def train_one_epoch(model, loader, optimizer, criterion, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for img, eeg in loader:
        img, eeg = img.to(device), eeg.to(device)

        # 造模拟标签：每张图随机一个灾害类型（实际场景用真实标注数据替换）
        labels = torch.randint(0, 5, (img.size(0),)).to(device)

        optimizer.zero_grad()                    # 清空上一轮的梯度
        outputs = model(img, eeg)                # 前向传播
        loss = criterion(outputs, labels)        # 计算交叉熵损失
        loss.backward()                          # 反向传播——自动求所有参数的梯度
        optimizer.step()                         # 用梯度更新参数

        total_loss += loss.item() * img.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += img.size(0)

    return total_loss / total, correct / total


def main():
    # 配置
    TIF_DIR = r"C:\Users\Nie\Documents\New project\test_data"
    EEG_DIR = r"D:\PYTHON\untitled\remote_bci\preprocessing\data\eeg"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 数据
    ds = DualModalDataset(TIF_DIR, EEG_DIR)
    loader = DataLoader(ds, batch_size=2, shuffle=True)

    # 模型
    model = SKMMFMNet(img_channels=4, eeg_channels=32, num_classes=5)
    model.to(device)

    # 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        loss, acc = train_one_epoch(model, loader, optimizer, criterion, device)
        print(f"Epoch {epoch:2d} | Loss: {loss:.4f} | Acc: {acc:.2%}")

    # 保存模型
    save_path = Path(__file__).parent / "sk_mmfmnet_checkpoint.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n模型已保存: {save_path}")


if __name__ == "__main__":
    main()