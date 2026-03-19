"""PyTorch / CUDA / Matplotlib 运行时配置。

提供
----
- configure_runtime()    强制使用非交互式 Matplotlib 后端；Windows 上的 OpenMP 变通方案。
- torch_runtime_info()   检测 CUDA 可用性、设备数量、设备名称。
- require_cuda()         返回 CUDA 设备，若不可用则抛出带诊断信息的异常。
- select_device()        "auto" / "cuda" / "cpu" 设备选择。
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass


def configure_runtime(*, matplotlib_backend: str = "Agg") -> None:
    """尽力进行运行时加固，确保 CLI 运行的可复现性。

    - 强制使用非交互式 Matplotlib 后端，避免仅保存图片时出现 Qt 线程关闭问题。
    - 解决 Windows 上加载多个 OpenMP 运行时（如 torch + numpy/pandas/opencv）时
      常见的崩溃问题。
    """

    os.environ.setdefault("MPLBACKEND", str(matplotlib_backend))

    if platform.system() == "Windows":
        # 变通方案："OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized."
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


@dataclass(frozen=True)
class TorchRuntimeInfo:
    torch_version: str
    cuda_available: bool
    torch_cuda_version: str | None
    device_count: int
    device_names: tuple[str, ...]


def torch_runtime_info() -> TorchRuntimeInfo:
    import torch

    device_count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    device_names: tuple[str, ...] = tuple(
        torch.cuda.get_device_name(i) for i in range(device_count)
    )
    return TorchRuntimeInfo(
        torch_version=str(torch.__version__),
        cuda_available=bool(torch.cuda.is_available()),
        torch_cuda_version=None if torch.version.cuda is None else str(torch.version.cuda),
        device_count=device_count,
        device_names=device_names,
    )


def require_cuda(*, device_index: int = 0) -> "torch.device":
    """返回 CUDA torch.device，若不可用则抛出带有帮助信息的异常。"""
    import torch

    info = torch_runtime_info()
    if not info.cuda_available:
        raise RuntimeError(
            "CUDA is required but was not detected.\n"
            f"- Detected torch: {info.torch_version}\n"
            f"- torch.version.cuda: {info.torch_cuda_version}\n"
            "Install a CUDA-enabled PyTorch build, then re-run.\n"
            "- PyTorch install selector: https://pytorch.org/get-started/locally/"
        )

    idx = int(device_index)
    if idx < 0 or idx >= info.device_count:
        raise RuntimeError(
            f"Invalid CUDA device index {idx}. Available device_count={info.device_count}."
        )

    return torch.device(f"cuda:{idx}")


def select_device(*, device: str = "auto", cuda_device: int = 0) -> "torch.device":
    """选择 torch.device。

    device:
      - "auto": 若 CUDA 可用则使用 CUDA，否则使用 CPU
      - "cuda": 要求 CUDA（不可用时抛出异常）
      - "cpu": 强制使用 CPU
    """
    import torch

    choice = str(device).lower().strip()
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        return require_cuda(device_index=int(cuda_device))
    if choice != "auto":
        raise ValueError(f"Unknown device {device!r}. Expected: auto|cpu|cuda")

    try:
        return require_cuda(device_index=int(cuda_device))
    except RuntimeError:
        return torch.device("cpu")
