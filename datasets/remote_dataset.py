"""遥感图像 PyTorch Dataset — 对接 preprocessing 管线
基于 [4] Revisiting pre-trained remote sensing model benchmarks
遥感专用数据标准化方案，输出的张量可直接喂给 SK-MMFMNet 的遥感分支
"""
from pathlib import Path
import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset, DataLoader


class RemoteSensingDataset(Dataset):
    """加载目录下所有 TIFF 文件，自动做辐射归一化"""

    def __init__(self, tif_dir: str, clip_percentile: float = 2.0):
        self.tif_paths = sorted(Path(tif_dir).glob("*.tif"))
        if len(self.tif_paths) == 0:
            raise FileNotFoundError(f"未找到任何 .tif 文件: {tif_dir}")
        self.clip_percentile = clip_percentile

    def __len__(self):
        return len(self.tif_paths)

    def __getitem__(self, idx: int):
        # 读取 TIFF
        tif_path = str(self.tif_paths[idx])
        with tifffile.TiffFile(tif_path) as tif:
            data = tif.asarray()

        # 统一形状 (C, H, W)
        if data.ndim == 2:
            data = data[np.newaxis, :, :]
        elif data.ndim == 3 and data.shape[2] < data.shape[0]:
            data = np.moveaxis(data, -1, 0)
        data = np.nan_to_num(data, nan=0.0).astype(np.float64)

        # 辐射归一化
        for i in range(data.shape[0]):
            band = data[i]
            low = np.percentile(band, self.clip_percentile)
            high = np.percentile(band, 100 - self.clip_percentile)
            denom = high - low
            if denom > 1e-8:
                data[i] = np.clip((band - low) / denom, 0.0, 1.0)

        # NumPy → PyTorch Tensor，float32
        tensor = torch.from_numpy(data.astype(np.float32))
        return tensor

    # —— 测试：加载 test_data 目录 ——


# —— 测试：加载 test_data 目录 ——
if __name__ == "__main__":
    TIF_DIR = r"C:\Users\Nie\Documents\New project\test_data"

    ds = RemoteSensingDataset(TIF_DIR)
    print(f"找到 {len(ds)} 张遥感影像")

    # DataLoader 是 PyTorch 的批量加载器
    loader = DataLoader(ds, batch_size=1, shuffle=True)

    for batch in loader:
        print(f"\n批次形状: {batch.shape}")  # 应为 (batch_size, C, H, W)
        print(f"数据类型: {batch.dtype}")  # 应为 torch.float32
        print(f"数值范围: [{batch.min():.4f}, {batch.max():.4f}]")  # 应接近 [0.0, 1.0]
        break  # 只跑一个 batch 就够了