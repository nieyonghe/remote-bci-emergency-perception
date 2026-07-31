"""双模态 Dataset — 遥感图像 + EEG 信号配对
支持 EuroSAT 数据集（按文件夹分类）+ 模拟 EEG
文献 [2] SK-MMFMNet 双输入架构的数据层实现
"""
from pathlib import Path
import numpy as np
import tifffile
from PIL import Image
import mne
import torch
from torch.utils.data import Dataset


class DualModalDataset(Dataset):
    """
    同时加载遥感图像和 EEG 信号
    支持两种模式：
      1. 文件夹模式（EuroSAT）：每个子文件夹是一个类别，自动生成标签
      2. 文件模式（原始 TIFF）：所有图像放在同一目录
    """

    def __init__(self, img_dir: str, eeg_dir: str, mode: str = "folder",
                 clip_percentile: float = 2.0):
        self.mode = mode
        self.clip_percentile = clip_percentile

        # === 加载图像路径和标签 ===
        if mode == "folder":
            # EuroSAT 模式：子文件夹 = 类别
            self.class_dirs = sorted([d for d in Path(img_dir).iterdir() if d.is_dir()])
            self.class_names = [d.name for d in self.class_dirs]
            self.img_paths = []
            self.labels = []
            for cls_idx, cls_dir in enumerate(self.class_dirs):
                for img_file in cls_dir.glob("*.*"):
                    if img_file.suffix.lower() in [".jpg", ".png", ".jpeg", ".tif", ".tiff"]:
                        self.img_paths.append(img_file)
                        self.labels.append(cls_idx)
            print(f"EuroSAT: {len(self.class_names)} 类, {len(self.img_paths)} 张图像")
            print(f"  类别: {self.class_names}")
        else:
            # 原始 TIFF 模式
            self.img_paths = sorted(Path(img_dir).glob("*.tif"))
            self.labels = [0] * len(self.img_paths)  # 假标签

        # === 预加载类别感知 EEG ===
        from preprocessing.eeg_class_generator import generate_class_aware_eeg
        self.eeg_cache = generate_class_aware_eeg(
            n_classes=len(self.class_names) if mode == "folder" else 1
        )
        print(f"类别感知 EEG 生成: {len(self.eeg_cache)} 类")

        self.n_samples = len(self.img_paths)

        if self.n_samples == 0:
            raise FileNotFoundError("未找到任何图像文件")

    def __len__(self):
        return self.n_samples

    def _load_image(self, idx: int):
        """加载并预处理遥感图像 → (C, H, W) float32 张量"""
        img_path = str(self.img_paths[idx % len(self.img_paths)])

        # 判断文件类型
        ext = Path(img_path).suffix.lower()
        if ext in [".tif", ".tiff"]:
            with tifffile.TiffFile(img_path) as tif:
                data = tif.asarray()
            if data.ndim == 2:
                data = data[np.newaxis, :, :]
            elif data.ndim == 3 and data.shape[2] < data.shape[0]:
                data = np.moveaxis(data, -1, 0)
        else:
            # JPG/PNG：PIL 读 → NumPy
            img = Image.open(img_path).convert("RGB")
            data = np.array(img, dtype=np.float64) / 255.0  # [0,255] → [0,1]
            data = np.moveaxis(data, -1, 0)  # (H, W, 3) → (3, H, W)

        data = np.nan_to_num(data, nan=0.0).astype(np.float64)

        # 辐射归一化（对 TIFF 有意义，JPG 已经归一化了就跳过）
        if ext in [".tif", ".tiff"]:
            for i in range(data.shape[0]):
                band = data[i]
                low = np.percentile(band, self.clip_percentile)
                high = np.percentile(band, 100 - self.clip_percentile)
                denom = high - low
                if denom > 1e-8:
                    data[i] = np.clip((band - low) / denom, 0.0, 1.0)

        return torch.from_numpy(data.astype(np.float32))

    def _load_eeg(self, idx: int):
        """返回该类别的 EEG —— 不同地物/灾害场景对应不同脑状态"""
        label = self.labels[idx % len(self.labels)]
        return self.eeg_cache[label].clone()

    def __getitem__(self, idx: int):
        img_tensor = self._load_image(idx)
        eeg_tensor = self._load_eeg(idx)
        label = self.labels[idx % len(self.labels)]
        return img_tensor, eeg_tensor, label


# ── 测试 ──
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    IMG_DIR = r"D:\PYTHON\untitled\remote_bci\data\remote_sensing\eurosat\2750"
    EEG_DIR = r"D:\PYTHON\untitled\remote_bci\preprocessing\data\eeg"

    ds = DualModalDataset(IMG_DIR, EEG_DIR, mode="folder")
    print(f"样本数: {len(ds)}")

    loader = DataLoader(ds, batch_size=4, shuffle=True)
    for img, eeg, lbl in loader:
        print(f"\n遥感: {img.shape}   数值范围: [{img.min():.2f}, {img.max():.2f}]")
        print(f"EEG:  {eeg.shape}")
        print(f"标签: {lbl}")
        break