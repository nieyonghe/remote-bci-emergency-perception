"""..."""
import numpy as np
import tifffile
from pathlib import Path
from typing import Tuple, Optional
def read_remote_tif(tif_path:str) -> Tuple[np.ndarray, int, int, int]:
    """读取遥感TIFF影像，返回 (波段矩阵, 宽, 高, 波段数)"""
    path = Path(tif_path)
    if not path.exists():
        raise FileNotFoundError(f"影像文件不存在：{tif_path}")

    with tifffile.TiffFile(str(path)) as tif:
        data = tif.asarray()

    if data.ndim == 2:
        data = data[np.newaxis,:,:]
    elif data.ndim == 3:
        if data.shape[2] < data.shape[0] and data.shape[2] < data.shape[1]:
            data = np.moveaxis(data,-1,0)

    data = np.nan_to_num(data,nan=0.0, posinf=0.0,neginf=0.0)
    band_count,height, width = data.shape
    return  data, width, height, band_count
def radiometric_normalize(
data: np.ndarray,
    clip_percentile: Optional[float] = None
) -> np.ndarray:
    data = np.asarray(data, dtype=np.float64)
    band_count = data.shape[0]
    result = np.zeros_like(data)
    for i in range(band_count):
        band = data[i]

        if clip_percentile is not None:
            # 取 [p, 100-p] 百分位区间，排除极端噪声点
            low = np.percentile(band, clip_percentile)
            high = np.percentile(band, 100 - clip_percentile)
        else:
            low = np.min(band)
            high = np.max(band)

        denom = high - low
        if denom < 1e-8:  # 分母接近 0 → 常量波段
            result[i] = band - low
        else:
            result[i] = (band - low) / denom  # 核心公式: (x - min) / (max - min)
            if clip_percentile is not None:
                result[i] = np.clip(result[i], 0.0, 1.0)  # 裁剪到 [0, 1]

    return result.astype(np.float32)  # 转 float32 省显存，PyTorch 默认 float32
def preprocessing_pipeline(
    tif_path: str,                              # 输入的遥感 TIFF 路径
    output_dir: str = "preprocess_output",      # 输出目录，有默认值
    clip_percentile: Optional[float] = 2.0      # 默认取 [2%, 98%] 区间做归一化
) -> Tuple[np.ndarray, np.ndarray]:             # 返回原始和归一化两份数据

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)  # 自动创建输出目录，已存在也不报错

    # Step 1: 调你前面写的 read_remote_tif
    data, width, height, bands = read_remote_tif(tif_path)
    print(f"[Step 1] ...")                     # f-string：f"{变量}" 直接把变量值插进字符串

    # Step 2: 调你前面写的 radiometric_normalize
    normalized = radiometric_normalize(data, clip_percentile=clip_percentile)

    # 打印每个波段的归一化前后对比，验证是否正常工作
    for i in range(min(bands, 5)):             # min 防止波段太多刷屏
        print(f"  Band {i+1}: [{np.min(data[i]):.2f}, {np.max(data[i]):.2f}] → ...")
        #      ↑ :.2f 是格式符，保留两位小数

    return data, normalized                    # 两份都返回，调用者自己决定用哪份
# ——下面是用来验证预处理管线的测试代码——
if __name__ == "__main__":
    TIF = r"C:\Users\Nie\Documents\New project\test_data\test_multispectral.tif"
    original, norm = preprocessing_pipeline(TIF)
    print(f"\n原始形状: {original.shape} → 归一化形状: {norm.shape}")