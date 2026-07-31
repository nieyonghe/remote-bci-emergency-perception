"""消融实验 — 对比三种融合方式
文献 [2] SK-MMFMNet + [MBC-ATT] + [10] CardiacMamba

对比：
  MMFNet            — 纯遥感，无EEG
  SK-MMFMNet-cat    — 遥感+EEG 简单拼接
  SK-MMFMNet-attn   — 遥感+EEG 跨模态注意力（MBC-ATT）
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from datasets.dual_modal_dataset import DualModalDataset
from models.mmfnet import MMFNet
from models.sk_mmfmnet import SKMMFMNet, EEGEncoder
from models.cross_attention import CrossModalAttention


# ========== 简单拼接版 SK-MMFMNet ==========
class SKMMFMNetConcat(nn.Module):
    """遥感+EEG 简单拼接融合（消融对比基线）"""
    def __init__(self, img_channels=3, eeg_channels=32, num_classes=10):
        super().__init__()
        self.img_stage1 = nn.Sequential(
            nn.Conv2d(img_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
        )
        from models.mmfnet import SKAttention
        self.img_sk1 = SKAttention(32)
        self.img_sk2 = SKAttention(64)
        self.img_sk3 = SKAttention(128)
        self.img_stage2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.img_stage3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.img_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(32 + 64 + 128, 256), nn.ReLU(),
        )
        self.eeg_encoder = EEGEncoder(eeg_channels)
        self.classifier = nn.Sequential(
            nn.Linear(256 + 128, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, img, eeg):
        f1 = self.img_sk1(self.img_stage1(img))
        f2 = self.img_sk2(self.img_stage2(f1))
        f3 = self.img_sk3(self.img_stage3(f2))
        f2_up = nn.functional.interpolate(f2, size=f1.shape[2:], mode='bilinear', align_corners=False)
        f3_up = nn.functional.interpolate(f3, size=f1.shape[2:], mode='bilinear', align_corners=False)
        img_feat = self.img_head(torch.cat([f1, f2_up, f3_up], dim=1))
        eeg_feat = self.eeg_encoder(eeg)
        fused = torch.cat([img_feat, eeg_feat], dim=1)  # ← 简单拼接
        return self.classifier(fused)


# ========== 训练/评估函数 ==========
def train_one_epoch(model, loader, optimizer, criterion, device, use_eeg=True):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for img, eeg, labels in loader:
        img, labels = img.to(device), labels.to(device)
        optimizer.zero_grad()
        if use_eeg:
            eeg = eeg.to(device)
            outputs = model(img, eeg)
        else:
            outputs = model(img)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * img.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += img.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, device, use_eeg=True):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for img, eeg, labels in loader:
            img, labels = img.to(device), labels.to(device)
            if use_eeg:
                outputs = model(img, eeg.to(device))
            else:
                outputs = model(img)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += img.size(0)
    return correct / total


# ========== 主实验 ==========
def main():
    device = torch.device("cpu")
    torch.manual_seed(42)

    # 数据
    IMG_DIR = r"D:\PYTHON\untitled\remote_bci\data\remote_sensing\eurosat\2750"
    EEG_DIR = r"D:\PYTHON\untitled\remote_bci\preprocessing\data\eeg"
    ds = DualModalDataset(IMG_DIR, EEG_DIR, mode="folder")

    # 8:2 划分训练/测试集
    train_size = int(0.8 * len(ds))
    test_size = len(ds) - train_size
    train_ds, test_ds = random_split(ds, [train_size, test_size])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    print(f"训练集: {train_size}, 测试集: {test_size}")

    # 三个模型
    models = {
        "MMFNet (纯遥感)": (MMFNet(in_channels=3, num_classes=10).to(device), False),
        "SK-MMFMNet-cat (简单拼接)": (SKMMFMNetConcat().to(device), True),
        "SK-MMFMNet-attn (MBC-ATT)": (SKMMFMNet(img_channels=3, num_classes=10).to(device), True),
    }

    results = []
    for name, (model, use_eeg) in models.items():
        params = sum(p.numel() for p in model.parameters())
        print(f"\n{'='*50}")
        print(f"训练: {name}  ({params:,} 参数)")
        print(f"{'='*50}")

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        best_acc = 0.0

        for epoch in range(1, 6):
            loss, acc = train_one_epoch(model, train_loader, optimizer, criterion, device, use_eeg)
            test_acc = evaluate(model, test_loader, device, use_eeg)
            best_acc = max(best_acc, test_acc)
            print(f"  Epoch {epoch} | Train Loss: {loss:.4f} | Train Acc: {acc:.2%} | Test Acc: {test_acc:.2%}")

        # 时延
        img = torch.randn(1, 3, 64, 64).to(device)
        eeg = torch.randn(1, 32, 256).to(device)
        model.eval()
        with torch.no_grad():
            for _ in range(10):
                model(img, eeg) if use_eeg else model(img)  # 预热
            t0 = time.perf_counter()
            for _ in range(100):
                model(img, eeg) if use_eeg else model(img)
            latency = (time.perf_counter() - t0) / 100 * 1000

        results.append({
            "name": name,
            "params": params,
            "test_acc": best_acc,
            "latency_ms": latency,
        })
        torch.save(model.state_dict(), Path(__file__).parent / f"{name.replace(' ', '_').replace('(','').replace(')','')}.pth")

    # ===== 结果表 =====
    print(f"\n{'='*80}")
    print("消融实验结果汇总")
    print(f"{'='*80}")
    print(f"{'模型':<30} {'参数量':>8} {'测试精度':>8} {'时延(ms)':>8}")
    print(f"{'-'*30} {'-'*8} {'-'*8} {'-'*8}")
    for r in results:
        print(f"{r['name']:<30} {r['params']:>8,} {r['test_acc']:>7.2%} {r['latency_ms']:>8.1f}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()