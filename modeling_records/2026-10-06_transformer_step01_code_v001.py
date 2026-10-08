import ctypes
import os
import sys
from importlib.util import find_spec
from pathlib import Path

import numpy as np

# Windows：复用项目已安装的 CUDA DLL；不导入或调用 PyTorch。
torch_spec = find_spec("torch")
if torch_spec is None:
    raise RuntimeError("本项目的 CUDA DLL 路径未找到，请先核对项目解释器。")
cuda_library_path = Path(torch_spec.origin).parent / "lib"
cuda_dll_handle = os.add_dll_directory(str(cuda_library_path))
cuda_library_handles = [
    ctypes.WinDLL(str(cuda_library_path / name))
    for name in (
        "nvrtc64_130_0.dll", "cublasLt64_13.dll",
        "cublas64_13.dll", "curand64_10.dll",
    )
]

import cupy as cp


# ==================================================
# 计算设备：NumPy 读取数据，CuPy 在 GPU 上计算
# ==================================================

cp.cuda.Device(0).use()
gpu_name = cp.cuda.runtime.getDeviceProperties(0)["name"]
if isinstance(gpu_name, bytes):
    gpu_name = gpu_name.decode("utf-8")

print("Python:", sys.version.split()[0])
print("NumPy:", np.__version__)
print("CuPy:", cp.__version__)
print("CuPy 可见 GPU 数:", cp.cuda.runtime.getDeviceCount())
print("指定计算设备: GPU 0")
print("GPU 型号:", gpu_name)


# ==================================================
# MNIST 数据：复用已经下载的本地缓存
# ==================================================

data_path = Path.home() / ".keras" / "datasets" / "mnist.npz"
if not data_path.exists():
    raise FileNotFoundError(f"没有找到 MNIST 缓存: {data_path}")

with np.load(data_path, allow_pickle=False) as data:
    x_train = cp.asarray(data["x_train"][:5000], dtype=cp.float32) / 255.0
    y_train = cp.asarray(data["y_train"][:5000], dtype=cp.int64)

    x_test = cp.asarray(data["x_test"][:1000], dtype=cp.float32) / 255.0
    y_test = cp.asarray(data["y_test"][:1000], dtype=cp.int64)


# ==================================================
# 一张图片 = 28 行组成的序列，每行有 28 个像素
# ==================================================

sequence = x_train[0]
label = y_train[0]

print("x_train:", x_train.shape)
print("y_train:", y_train.shape)
print("x_test:", x_test.shape)
print("y_test:", y_test.shape)
print("一张图片的序列形状:", sequence.shape)
print("一行的输入形状:", sequence[0].shape)
print("像素范围:", float(x_train.min()), float(x_train.max()))
print("第一张图片的标签:", int(label))
print("训练输入的实际设备:", x_train.device)
print("测试输入的实际设备:", x_test.device)
