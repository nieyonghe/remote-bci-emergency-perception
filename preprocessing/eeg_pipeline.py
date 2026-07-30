"""EEG脑电预处理流水线 — 上行反馈支路
基于 [6] MNE-Python & EEG-Pype 开源预处理管线

流程：原始EEG(.edf) → 带通滤波 → ICA去眼电伪迹 → 分段Epoch → 转NumPy
用途：采集救援人员脑电，提取注意力/疲劳状态，反馈给下行BCI调节遥感信息强度
"""

import numpy as np
import mne
from pathlib import Path
from typing import Optional, Tuple


def preprocess_eeg(
    edf_path: str,
    low_freq: float = 1.0,
    high_freq: float = 40.0,
    tmin: float = -0.2,
    tmax: float = 0.8
) -> np.ndarray:
    """
    标准EEG预处理流水线
    Step 1: 读原始脑电文件
    Step 2: 带通滤波，去直流漂移和高频肌电
    Step 3: ICA 去除眼电/眨眼伪迹
    Step 4: 分段为 epoch
    Step 5: 转 NumPy (n_epochs, n_channels, n_times)
    """
    edf_path = Path(edf_path)
    if not edf_path.exists():
        raise FileNotFoundError(f"EEG文件不存在: {edf_path}")

    # Step 1: 读取
    raw = mne.io.read_raw(str(edf_path), preload=True)
    print(f"[Step 1] EEG读取: {len(raw.ch_names)}通道, {raw.n_times}采样点, {raw.info['sfreq']}Hz")

    # Step 2: 带通滤波 (保留 1-40Hz，去除直流漂移和高频噪声)
    raw.filter(low_freq, high_freq)
    print(f"[Step 2] 带通滤波: {low_freq}-{high_freq}Hz")

    # Step 3: ICA 去眼电伪迹
    # Step 3: ICA 去眼电伪迹（没有EOG通道时跳过）
    ica = mne.preprocessing.ICA(n_components=min(15, len(raw.ch_names) - 1),
                                random_state=42, max_iter='auto')
    ica.fit(raw)
    try:
        eog_indices, _ = ica.find_bads_eog(raw, ch_name=None)
        if len(eog_indices) > 0:
            ica.exclude = eog_indices
            raw = ica.apply(raw)
            print(f"[Step 3] ICA 去除 {len(eog_indices)} 个眼电成分")
        else:
            print(f"[Step 3] ICA 完成，未检测到明显眼电成分")
    except RuntimeError:
        print(f"[Step 3] ICA 完成，无EOG通道（模拟数据），跳过眼电去除")

    # Step 4: 分段为 epoch（以事件标记为基准）
    events = mne.make_fixed_length_events(raw, duration=1.0)
    epochs = mne.Epochs(raw, events, tmin=tmin, tmax=tmax,
                        baseline=(tmin, 0), preload=True)
    print(f"[Step 4] 分段: {len(epochs)} 个 epoch, 形状 (n_epochs, n_channels, n_times)")

    # Step 5: 转 NumPy
    data = epochs.get_data().astype(np.float32)
    print(f"[Step 5] 输出 NumPy 形状: {data.shape}")
    return data
def generate_synthetic_eeg(
    duration: float = 60.0,
    sfreq: float = 256.0,
    n_channels: int = 32,
    output_path: Optional[str] = None
) -> str:
    """
    生成模拟EEG数据用于测试
    输出格式和真实脑电设备完全一致: .fif 文件
    """
    # 通道名：按国际 10-20 系统命名
    ch_names = [f'EEG {i:03d}' for i in range(n_channels)]
    ch_types = ['eeg'] * n_channels

    # 创建通道信息对象
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # 生成伪脑电：随机噪声 + 10Hz alpha 波（闭眼时出现的大脑节律）
    n_samples = int(duration * sfreq)
    noise = np.random.randn(n_channels, n_samples) * 5e-6        # 5 微伏随机噪声
    t = np.arange(n_samples) / sfreq                              # 时间轴
    alpha = 20e-6 * np.sin(2 * np.pi * 10 * t)                   # 10Hz alpha 波，20 微伏振幅
    data = noise + alpha[np.newaxis, :]                           # 每通道加 alpha

    raw = mne.io.RawArray(data, info)

    # 保存
    if output_path is None:
        output_path = "data/eeg/synthetic_eeg.fif"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw.save(str(output_path), overwrite=True)

    print(f"模拟EEG已保存: {output_path}")
    print(f"  {n_channels}通道, {sfreq}Hz, {duration}秒, {n_samples}采样点")
    return str(output_path)
if __name__ == "__main__":
    # Step 1: 生成模拟EEG
    fif_path = generate_synthetic_eeg(duration=10.0)
    #                                 ↑ 只模拟10秒，跑得快

    # Step 2: 跑预处理流水线
    data = preprocess_eeg(fif_path)

    print(f"\n最终输出: {data.shape}")
    print(f"  解释: {data.shape[0]}个epoch, {data.shape[1]}通道, {data.shape[2]}个时间点")