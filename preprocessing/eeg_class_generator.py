"""基于类别的模拟 EEG 生成器 — 为 EuroSAT 每类生成不同脑电特征
用途：模拟不同灾害/地物场景下救援人员的脑状态差异
"""
import numpy as np
import torch


def generate_class_aware_eeg(n_classes: int = 10, n_channels: int = 32,
                              n_times: int = 256, sfreq: float = 256.0):
    """
    为每个类别生成独特的 EEG 模板
    每类的 alpha(8-13Hz)、beta(13-30Hz)、theta(4-8Hz) 功率不同
    返回: dict {class_idx: (n_channels, n_times) 张量}
    """
    t = np.arange(n_times) / sfreq
    templates = {}

    # 每类的频段功率分配（不同地物 → 不同脑状态）
    # 格式: (theta_power, alpha_power, beta_power, 总振幅)
    # 值越大该频段越突出
    class_profiles = {
        0: (0.2, 0.5, 0.3, 15.0),   # AnnualCrop     — 中 alpha（观察农田，放松）
        1: (0.1, 0.7, 0.2, 12.0),   # Forest         — 高 alpha（看森林，很放松）
        2: (0.2, 0.4, 0.4, 18.0),   # Herbaceous     — 中 beta（辨别植被，稍专注）
        3: (0.1, 0.2, 0.7, 22.0),   # Highway        — 高 beta（识别道路，高度专注）
        4: (0.1, 0.3, 0.6, 25.0),   # Industrial     — 高 beta（工业区，高度警觉）
        5: (0.3, 0.5, 0.2, 14.0),   # Pasture        — 中 alpha（牧场，放松）
        6: (0.2, 0.4, 0.4, 16.0),   # PermanentCrop  — 中 beta（经济作物，专注）
        7: (0.1, 0.3, 0.6, 20.0),   # Residential    — 中高 beta（居住区，关注）
        8: (0.4, 0.5, 0.1, 18.0),   # River          — 高 theta+alpha（水域，平静）
        9: (0.3, 0.6, 0.1, 16.0),   # SeaLake        — 高 alpha（湖泊，非常平静）
    }

    for cls_idx in range(n_classes):
        tp, ap, bp, amp = class_profiles.get(cls_idx, (0.3, 0.4, 0.3, 15.0))
        total = tp + ap + bp

        # 背景噪声
        noise = np.random.randn(n_channels, n_times) * 3e-6

        # 各频段信号
        theta = 10e-6 * np.sin(2 * np.pi * 6 * t)   # 6Hz theta
        alpha = 15e-6 * np.sin(2 * np.pi * 10 * t)   # 10Hz alpha
        beta  = 8e-6  * np.sin(2 * np.pi * 20 * t)   # 20Hz beta

        # 加权合成（每通道加不同的微小相位偏移）
        signal = np.zeros((n_channels, n_times))
        for ch in range(n_channels):
            phase_shift = np.random.uniform(0, 2 * np.pi)
            ch_theta = 10e-6 * np.sin(2 * np.pi * 6 * t + phase_shift * 0.3)
            ch_alpha = 15e-6 * np.sin(2 * np.pi * 10 * t + phase_shift * 0.5)
            ch_beta  = 8e-6  * np.sin(2 * np.pi * 20 * t + phase_shift * 0.7)
            signal[ch] = noise[ch] + (tp / total) * ch_theta + (ap / total) * ch_alpha + (bp / total) * ch_beta

        # 缩放振幅
        signal = signal * amp / np.std(signal)
        templates[cls_idx] = torch.from_numpy(signal.astype(np.float32))

    return templates


if __name__ == "__main__":
    templates = generate_class_aware_eeg()
    for cls_idx, data in templates.items():
        print(f"Class {cls_idx}: shape={data.shape}, "
              f"range=[{data.min():.1e}, {data.max():.1e}]")
    print(f"\nGenerated {len(templates)} class-aware EEG templates")