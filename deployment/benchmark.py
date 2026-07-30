"""端侧部署与时延基准测试 — 文献 [9] BrainFusion
任务：模型导出 ONNX → 推理时延测量 → 论文实验数据
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from models.sk_mmfmnet import SKMMFMNet


def measure_latency(model, *inputs, n_warmup: int = 10, n_measure: int = 100):
    """MBC-ATT 版和纯遥感版都跑，传 eeg 给纯遥感也没关系——它不用"""
    """
    测量单次推理时延（毫秒）
    文献 [9] BrainFusion：遥感事件 → AI推理 → 脑刺激输出 的端到端延迟
    """
    model.eval()

    # 预热：前几次调用包含 CUDA 初始化和缓存预热，不计入统计
    with torch.no_grad():
        for _ in range(n_warmup):
            model(*inputs)

    # 正式测量
    times = []
    with torch.no_grad():
        for _ in range(n_measure):
            t0 = time.perf_counter()
            model(*inputs)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # 秒 → 毫秒

    times = torch.tensor(times)
    return {
        "mean_ms": times.mean().item(),
        "std_ms": times.std().item(),
        "min_ms": times.min().item(),
        "max_ms": times.max().item(),
        "n_measure": n_measure,
    }


def export_onnx(model, img, eeg, save_path: str):
    """导出 ONNX 模型 — 文献 [9] 移动端部署"""
    model.eval()
    torch.onnx.export(
        model,
        (img, eeg),
        save_path,
        input_names=["remote_sensing", "eeg_signal"],
        output_names=["output"],
        dynamic_axes={
            "remote_sensing": {0: "batch"},
            "eeg_signal": {0: "batch"},
            "output": {0: "batch"},
        },
        opset_version=14,
    )
    print(f"ONNX 模型已导出: {save_path}")
    print(f"文件大小: {Path(save_path).stat().st_size / 1024:.1f} KB")


def main():
    device = torch.device("cpu")
    print(f"设备: {device}")

    # 加载模型（用已保存的权重，如果没有就新建）
    model = SKMMFMNet(img_channels=4, eeg_channels=32, num_classes=5)
    ckpt_path = Path(__file__).parent.parent / "experiments" / "sk_mmfmnet_checkpoint.pth"
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"已加载权重: {ckpt_path}")
    else:
        print("未找到权重，使用随机初始化参数")

    model.to(device)

    # 构造测试输入（和 DualModalDataset 输出形状一致）
    img = torch.randn(1, 4, 512, 512).to(device)
    eeg = torch.randn(1, 32, 256).to(device)

    # === 时延基准 ===
    print("\n=== 推理时延基准 ===")
    from models.mmfnet import MMFNet
    models_to_test = {
        "MMFNet (纯遥感)": MMFNet(in_channels=4, num_classes=5).to(device),
        "SK-MMFMNet (双模态)": model,
    }

    for name, m in models_to_test.items():
        if name == "MMFNet (纯遥感)":
            result = measure_latency(m, img)
        else:
            result = measure_latency(m, img, eeg)
        print(f"\n{name}:")
        print(f"  均值: {result['mean_ms']:.2f} ms")
        print(f"  标准差: {result['std_ms']:.2f} ms")
        print(f"  范围: [{result['min_ms']:.2f}, {result['max_ms']:.2f}] ms")

    # === ONNX 导出 ===
    print("\n=== ONNX 导出 ===")
    onnx_path = Path(__file__).parent / "sk_mmfmnet.onnx"
    export_onnx(model, img.cpu(), eeg.cpu(), str(onnx_path))

    # === 参数量对比 ===
    print("\n=== 模型参数量 ===")
    for name, m in models_to_test.items():
        params = sum(p.numel() for p in m.parameters())
        print(f"  {name}: {params:,}")


if __name__ == "__main__":
    main()