# 手写 MNIST Transformer：第 1 步

日期：2026-10-06（Asia/Shanghai）；版本：v001。

按此前教学方式，由你在 VS Code 逐段输入，使用普通函数和基础数组运算，自己写前向、梯度和参数更新。最终训练使用 GPU 数组后端 CuPy。助手没有创建或改写你的主 `transformer_from_scratch.py`。

## GPU 为什么可以使用

普通 NumPy 数组由 CPU 计算。CuPy 提供相近的数组接口并将数组放在 GPU；`cp.asarray(...)` 将本步 MNIST 输入从 CPU 传到 GPU。手写模型的数学公式仍由我们实现，不使用现成 Transformer 层、自动求导或框架优化器。[NumPy 官方互操作说明](https://numpy.org/doc/stable/user/basics.interoperability.html) · [CuPy 基础用法](https://docs.cupy.dev/en/stable/user_guide/basic.html)。

本项目解释器是 `F:\PythonProjects\deep_learning\.venv\Scripts\python.exe`。本轮已安装 CuPy 14.2.0、cuda-pathfinder 1.8.3、nvidia-cuda-runtime 13.2.51，保留现有 NumPy 2.5.3。Windows 启动段复用项目现有 torch/lib 内的 CUDA DLL；不导入 PyTorch、不使用其计算或自动求导。启动段可直接照用，之后的网络部分逐段讲解手写。

**已实际核验**：NVIDIA GeForce RTX 5060；CuPy 可见一张 GPU；数据数组和矩阵乘法结果位于 `<CUDA Device 0>`。完整 Transformer 尚未初始化或训练。

## 本步代码

在 VS Code 打开 `F:\PythonProjects\deep_learning`，选择项目 `.venv` 解释器，自己新建 `transformer_from_scratch.py`，输入下面代码并运行。本地 MNIST 缓存已验证存在，本步直接读取缓存，减少数据读取对训练框架的依赖。

```python
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
```

## 为什么保留 28 × 28

一张图片包含 28 行，每行 28 个像素。把它当作序列时，序列长度 T=28，每个位置的输入维度 D=28；`sequence[t]` 就是第 t 行。整张图片只有一个数字类别标签，不能把图片标签当作每一行的独立标签。

这是一种人为选定的空间扫描表示，MNIST 不是时间序列。Transformer 会同时处理这些位置，通过注意力建立行与行之间的联系。我们计划加入位置编码来表示上下行顺序。

本步应看到：

```text
x_train: (5000, 28, 28)
y_train: (5000,)
x_test: (1000, 28, 28)
y_test: (1000,)
一张图片的序列形状: (28, 28)
一行的输入形状: (28,)
像素范围: 0.0 1.0
第一张图片的标签: 5
训练输入的实际设备: <CUDA Device 0>
测试输入的实际设备: <CUDA Device 0>
```

## 后续路线（尚未执行）

1. 输入投影与正弦位置编码：28 维像素行映射到 d_model=32。
2. Q、K、V 投影和单头缩放点积注意力。
3. 四个注意力头，每头 8 维，合并与输出投影。
4. 残差连接、LayerNorm、两层前馈网络，组成一个 Encoder block。
5. 行特征平均池化、十类输出与交叉熵。
6. 逐段手写反向传播，并用小模型有限差分检查梯度。
7. 手写 SGD 更新；核验一次更新，再组织 GPU 小批次训练。
8. 使用独立测试集评价，保存预测、参数、代码快照和后续记录。

这是用于分类的一个简化 Transformer Encoder，未实现翻译用 Encoder–Decoder。图像分类注意力允许读取全部行；固定长度 28，无须因果掩码或 padding。核心结构参考 [Attention Is All You Need](https://arxiv.org/abs/1706.03762)，具体维度属于教学选择，并非最优参数结论。

[本步代码快照](2026-10-06_transformer_step01_code_v001.py) · [真实 GPU 验证结果](2026-10-06_transformer_data_gpu_validation_v001.json) · [建模记录](2026-10-06_transformer_mnist_v001.md)。
