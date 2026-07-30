"""MBC-ATT 跨模态注意力融合模块 [文献补充]
文献：MBC-ATT — EEG+视觉跨模态注意力融合 (Frontiers)

在你的课题中担任：
  遥感特征(Query) + EEG脑状态(Key/Value) → Cross-Attention → 动态调节
  根据救援人员脑状态，决定遥感信息编码的重点和强度
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    """
    跨模态注意力融合
    原理：用遥感语义特征做 Query，用 EEG 脑状态特征做 Key 和 Value
          让网络学会「在当前脑状态下，遥感场景的哪些信息更重要」
    """

    def __init__(self, img_dim: int = 256, eeg_dim: int = 128,
                 hidden_dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads

        # 投影层：把两个模态映射到同一个空间
        self.q_proj = nn.Linear(img_dim, hidden_dim)   # 遥感 → Query
        self.k_proj = nn.Linear(eeg_dim, hidden_dim)   # EEG  → Key
        self.v_proj = nn.Linear(eeg_dim, hidden_dim)   # EEG  → Value

        self.out_proj = nn.Linear(hidden_dim, img_dim)  # 投影回遥感维度
        self.scale = self.head_dim ** -0.5               # 缩放因子，防止点积太大

    def forward(self, img_feat, eeg_feat):
        """
        img_feat: (B, img_dim)  遥感语义压缩向量
        eeg_feat: (B, eeg_dim)  EEG 脑状态向量
        返回:     (B, img_dim)  脑状态调节后的遥感特征
        """
        B = img_feat.size(0)

        # 投影
        Q = self.q_proj(img_feat).view(B, -1, self.num_heads, self.head_dim)
        # (B, num_heads, head_dim) 的等效变换
        Q = Q.permute(0, 2, 1, 3).reshape(B * self.num_heads, 1, self.head_dim)

        K = self.k_proj(eeg_feat).view(B, -1, self.num_heads, self.head_dim)
        K = K.permute(0, 2, 1, 3).reshape(B * self.num_heads, 1, self.head_dim)

        V = self.v_proj(eeg_feat).view(B, -1, self.num_heads, self.head_dim)
        V = V.permute(0, 2, 1, 3).reshape(B * self.num_heads, 1, self.head_dim)

        # 注意力计算：Q × K^T / sqrt(d)
        attn = (Q * K).sum(dim=-1, keepdim=True) * self.scale
        #           ↑ 点积：Query 和 Key 的相似度
        attn = F.softmax(attn, dim=0)    # 归一化为权重

        # 加权输出：attn × V
        out = (attn * V).reshape(B, self.num_heads, self.head_dim)
        out = out.reshape(B, self.hidden_dim)
        out = self.out_proj(out)

        # 残差连接：原始遥感特征 + 脑状态调节量
        return img_feat + out
if __name__ == "__main__":
    # 模拟 SK-MMFMNet 两个分支的输出
    dummy_img_feat = torch.randn(2, 256)   # 2个样本，遥感特征256维
    dummy_eeg_feat = torch.randn(2, 128)   # 2个样本，EEG特征128维

    attn = CrossModalAttention(img_dim=256, eeg_dim=128)
    fused = attn(dummy_img_feat, dummy_eeg_feat)

    print(f"遥感特征输入: {dummy_img_feat.shape}")
    print(f"EEG特征输入:   {dummy_eeg_feat.shape}")
    print(f"融合输出:      {fused.shape}   (遥感维度不变，但内部已被脑状态调节)")
    print(f"参数量:        {sum(p.numel() for p in attn.parameters()):,}")