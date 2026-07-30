"""MMFNet — 多模态特征融合网络 [文献 1]
遥感多光谱融合原版，SK-MMFMNet 的前身

架构：
  遥感图像 → CNN编码器 → 多尺度特征
                             ↓
                        SK选择性内核注意力 → 融合 → 分类
  用途（你的课题）：对空天地遥感数据做语义压缩，筛选关键灾害信息
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SKAttention(nn.Module):
    """
    Selective Kernel Attention — MMFNet 的核心模块
    原理：用两个不同大小的卷积核（3×3 和 5×5）并行处理同一特征，
          然后让网络自己学会"哪个核的响应更重要"
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        # 两个不同感受野的卷积分支
        self.conv3 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.conv5 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)

        # 压缩全连接：C → C/r → 2*C（为两个分支各生成一个权重向量）
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),         # 全局平均池化 → (B, C, 1, 1)
            nn.Flatten(),                     # → (B, C)
            nn.Linear(channels, mid),
            nn.ReLU(),
            nn.Linear(mid, channels * 2),    # → (B, 2C)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 两个卷积分支
        u3 = self.conv3(x)   # (B, C, H, W)
        u5 = self.conv5(x)   # (B, C, H, W)
        u = u3 + u5          # 逐元素和

        # 生成注意力权重
        w = self.fc(u)                     # (B, 2C)
        w = w.view(w.size(0), 2, -1)       # (B, 2, C) —— 2个分支各一组权重
        w = self.softmax(w)                # softmax 使两个权重和为1

        # 加权融合
        a3 = w[:, 0, :].view(-1, u3.size(1), 1, 1)  # (B, C, 1, 1)
        a5 = w[:, 1, :].view(-1, u5.size(1), 1, 1)
        return u3 * a3 + u5 * a5
class MMFNet(nn.Module):
    """
    MMFNet 主干网络 — 遥感图像语义压缩编码器
    文献 [1] 多模态特征融合的遥感图像语义分割网络

    在你的课题中担任：下行链路的遥感语义压缩模块
    输入空天地遥感影像 → 输出压缩后的灾害语义特征
    """

    def __init__(self, in_channels: int = 4, num_classes: int = 10):
        super().__init__()

        # === 遥感编码器（三阶段下采样） ===
        # Stage 1: 保持分辨率
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        # Stage 2: 2倍下采样
        self.stage2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        # Stage 3: 4倍下采样
        self.stage3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # === SK 注意力（每个阶段后各一个） ===
        self.sk1 = SKAttention(32)
        self.sk2 = SKAttention(64)
        self.sk3 = SKAttention(128)

        # === 多尺度特征融合 + 分类头 ===
        # 把三个尺度的特征压缩到统一大小后拼接
        self.fusion = nn.Sequential(
            nn.Conv2d(32 + 64 + 128, 256, 1),   # 1×1 卷积融合多尺度
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),              # → (B, 256, 1, 1)
            nn.Flatten(),                          # → (B, 256)
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        """
        x: (B, C, H, W) — 遥感波段矩阵
        返回: (B, num_classes) — 分类结果
        """
        # 提取多尺度特征
        f1 = self.sk1(self.stage1(x))    # (B, 32, H, W)
        f2 = self.sk2(self.stage2(f1))   # (B, 64, H/2, W/2)
        f3 = self.sk3(self.stage3(f2))   # (B, 128, H/4, W/4)

        # 上采样到统一尺寸
        f2_up = F.interpolate(f2, size=f1.shape[2:], mode='bilinear',
                              align_corners=False)
        f3_up = F.interpolate(f3, size=f1.shape[2:], mode='bilinear',
                              align_corners=False)

        # 拼接三个尺度 → 分类
        fused = torch.cat([f1, f2_up, f3_up], dim=1)  # (B, 224, H, W)
        return self.fusion(fused)
if __name__ == "__main__":
    # 用 dual_modal_dataset 输出的形状造假数据
    dummy_img = torch.randn(2, 4, 512, 512)   # 2张图, 4波段, 512×512

    model = MMFNet(in_channels=4, num_classes=5)  # 5类：正常/火灾/洪水/滑坡/倒塌
    out = model(dummy_img)
    print(f"输入形状: {dummy_img.shape}")
    print(f"输出形状: {out.shape}")             # (2, 5) —— 2张图各5个分类分数
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")