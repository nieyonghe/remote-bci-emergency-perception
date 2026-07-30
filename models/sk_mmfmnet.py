"""SK-MMFMNet — 遥感+EEG双模态选择性内核融合网络 [文献 2]
文献：Long J, Fang Z, Wang L. SK-MMFMNet: A Multi-dimensional Fusion Network
      of Remote Sensing Images and EEG signals. Information Fusion, 2024.

在你的课题中担任：
  遥感分支 → 下行链路语义压缩
  EEG分支  → 上行链路脑状态解码
  融合模块 → 根据脑状态动态调节遥感信息编码强度
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from .mmfnet import SKAttention    # 被其他模块导入时用这个
except ImportError:
    from mmfnet import SKAttention     # 直接运行脚本时用这个

try:
    from .cross_attention import CrossModalAttention
except ImportError:
    try:
        from models.cross_attention import CrossModalAttention
    except ImportError:
        from cross_attention import CrossModalAttention
class EEGEncoder(nn.Module):
    """
    EEG 1D 编码器 — 将脑电时序信号压缩为特征向量
    输入: (B, n_channels, n_times) 如 (B, 32, 256)
    输出: (B, 128) 特征向量
    """

    def __init__(self, n_channels: int = 32, n_times: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_channels, 32, 7, padding=3),   # 32通道 → 32特征
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),                             # 时间减半: 256→128

            nn.Conv1d(32, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),                             # 再减半: 128→64

            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),                     # 时间压缩为1: (B, 128, 1)
            nn.Flatten(start_dim=1),                     # → (B, 128)
        )

    def forward(self, eeg):
        return self.encoder(eeg)
class SKMMFMNet(nn.Module):
    """
    SK-MMFMNet 双模态融合网络
    文献 [2] Information Fusion, 2024 — 方志祥团队

    流程:
      遥感图像 ──→ CNN编码器(SK注意力) ──→ 遥感特征 (B, 256)
      EEG信号  ──→ 1D CNN编码器         ──→ EEG特征  (B, 128)
                                     ↓
                            特征拼接 → 融合全连接 → 分类
    """

    def __init__(self, img_channels: int = 4, eeg_channels: int = 32,
                 num_classes: int = 10):
        super().__init__()

        # === 遥感分支（复用 MMFNet 的三阶段结构） ===
        self.img_stage1 = nn.Sequential(
            nn.Conv2d(img_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
        )
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
        self.img_sk1 = SKAttention(32)
        self.img_sk2 = SKAttention(64)
        self.img_sk3 = SKAttention(128)

        # 遥感分支最终压缩为 256 维特征向量
        self.img_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32 + 64 + 128, 256),
            nn.ReLU(),
        )

        # === EEG 分支 ===
        self.eeg_encoder = EEGEncoder(eeg_channels)

        # === 跨模态注意力融合 ===
        self.cross_attn = CrossModalAttention(img_dim=256, eeg_dim=128)

        # === 分类头 ===
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),  # 输入从 384 变为 256——因为跨注意力输出保持遥感维度   # 遥感256 + EEG128 = 384 → 128
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, img, eeg):
        """
        img: (B, C_img, H, W)  遥感波段矩阵
        eeg: (B, C_eeg, T)     EEG时序信号
        返回: (B, num_classes)
        """
        # 遥感分支：提取多尺度特征
        f1 = self.img_sk1(self.img_stage1(img))         # (B, 32, H, W)
        f2 = self.img_sk2(self.img_stage2(f1))          # (B, 64, H/2, W/2)
        f3 = self.img_sk3(self.img_stage3(f2))          # (B, 128, H/4, W/4)

        # 多尺度特征融合 → 遥感特征向量
        f2_up = F.interpolate(f2, size=f1.shape[2:], mode='bilinear',
                              align_corners=False)
        f3_up = F.interpolate(f3, size=f1.shape[2:], mode='bilinear',
                              align_corners=False)
        img_feat = self.img_head(torch.cat([f1, f2_up, f3_up], dim=1))
        # img_feat: (B, 256)

        # EEG 分支
        eeg_feat = self.eeg_encoder(eeg)   # (B, 128)

        # 跨模态注意力融合：EEG 脑状态动态调节遥感特征
        fused = self.cross_attn(img_feat, eeg_feat)  # (B, 256)
        return self.classifier(fused)
if __name__ == "__main__":
    # 用 dual_modal_dataset 的真实输出形状造假数据
    dummy_img = torch.randn(2, 4, 512, 512)   # 2张遥感图, 4波段
    dummy_eeg = torch.randn(2, 32, 256)        # 2段脑电, 32通道, 256时间点

    model = SKMMFMNet(img_channels=4, eeg_channels=32, num_classes=5)
    out = model(dummy_img, dummy_eeg)

    print(f"遥感输入: {dummy_img.shape}")
    print(f"EEG输入:  {dummy_eeg.shape}")
    print(f"输出:     {out.shape}   (2个样本, 5类分数)")
    print(f"参数量:   {sum(p.numel() for p in model.parameters()):,}")