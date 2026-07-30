"""双模态 Dataset — 遥感图像 + EEG 信号配对
文献 [2] SK-MMFMNet 双输入架构的数据层实现
"""
from pathlib import Path
import numpy as np
import tifffile
import mne
import torch
from torch.utils.data import Dataset


class DualModalDataset(Dataset):
    """
    同时加载遥感 TIFF 和 EEG(.fif/.edf) 文件
    返回 (遥感图像, EEG信号) pair —— SK-MMFMNet 的双输入
    """

    def __init__(self, tif_dir: str, eeg_dir: str, clip_percentile: float = 2.0):
        self.tif_paths = sorted(Path(tif_dir).glob("*.tif"))
        self.eeg_paths = sorted(Path(eeg_dir).glob("*.*"))
        # 按数量少的那个对齐
        self.n_samples = min(len(self.tif_paths), len(self.eeg_paths))

        if self.n_samples == 0:
            raise FileNotFoundError("需要至少一张遥感TIFF和一个EEG文件")

        self.clip_percentile = clip_percentile

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int):
        # === 遥感图像分支 ===
        tif_path = str(self.tif_paths[idx % len(self.tif_paths)])
        with tifffile.TiffFile(tif_path) as tif:
            img_data = tif.asarray()

        if img_data.ndim == 2:
            img_data = img_data[np.newaxis, :, :]
        elif img_data.ndim == 3 and img_data.shape[2] < img_data.shape[0]:
            img_data = np.moveaxis(img_data, -1, 0)
        img_data = np.nan_to_num(img_data, nan=0.0).astype(np.float64)

        for i in range(img_data.shape[0]):
            band = img_data[i]
            low = np.percentile(band, self.clip_percentile)
            high = np.percentile(band, 100 - self.clip_percentile)
            denom = high - low
            if denom > 1e-8:
                img_data[i] = np.clip((band - low) / denom, 0.0, 1.0)

        img_tensor = torch.from_numpy(img_data.astype(np.float32))

        # === EEG 信号分支 ===
        eeg_path = str(self.eeg_paths[idx % len(self.eeg_paths)])
        raw = mne.io.read_raw(eeg_path, preload=True)
        raw.filter(1.0, 40.0)

        # 取一个固定长度的 epoch
        n_times_target = 256  # 固定 256 个时间点
        if raw.n_times < n_times_target:
            raise ValueError(f"EEG 数据太短: {raw.n_times} < {n_times_target}")

        eeg_matrix = raw.get_data()[:, :n_times_target].astype(np.float32)
        eeg_tensor = torch.from_numpy(eeg_matrix)  # (n_channels, n_times)

        return img_tensor, eeg_tensor
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    # 用你之前的 test_data 和模拟 EEG
    TIF_DIR = r"C:\Users\Nie\Documents\New project\test_data"
    EEG_DIR = r"/remote_bci/preprocessing/data/eeg"

    ds = DualModalDataset(TIF_DIR, EEG_DIR)
    print(f"双模态样本数: {len(ds)}")

    loader = DataLoader(ds, batch_size=1, shuffle=True)
    for img, eeg in loader:
        print(f"\n遥感图像: {img.shape}")     # (1, C, H, W)
        print(f"EEG信号:  {eeg.shape}")        # (1, n_channels, n_times)
        break