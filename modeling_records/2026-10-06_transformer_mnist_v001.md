# MNIST 手写 Transformer 建模记录

日期：2026-10-06（Asia/Shanghai）；版本：v001。
阶段：教学结构规划、GPU 后端配置与真实数据准备验证。**未初始化 Transformer 参数，未执行完整前向、模型反向、模型训练或预测。**

## 1. 问题范围与约束

沿用此前 MNIST 手写 DNN、残差网络和 LSTM 的逐段教学方式，采用基础数组运算和普通函数，由用户在 VS Code 手写。用户明确要求梯度与参数更新也手写，随后要求最终在可用 GPU 上运行；据此采用 NumPy 读取缓存、CuPy 执行 GPU 数组运算。模型拟为单层 Transformer Encoder 十类分类。本轮没有创建、修改用户主 Transformer 文件，也没有覆盖既有网络。

## 2. 输入来源、处理与事实

实际读取本地缓存 `C:\Users\19367\.keras\datasets\mnist.npz`；SHA256 `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`。缓存原始训练图 `(60000,28,28)`、训练标签 `(60000,)`，测试图 `(10000,28,28)`、测试标签 `(10000,)`，dtype 均为 uint8。没有重新下载数据。

本轮保留训练集前 5000 图、测试集前 1000 图；图像先转 CuPy float32，再除以 255；标签转 CuPy int64，不归一化。训练/测试保持分离；没有追加清洗、洗牌或数据增强。两个输入的像素范围均为 0～1，全部有限，标签在 0～9，标签逐元素与缓存一致。GPU 归一化与 NumPy float32 参考值在 rtol=atol=1e-7 下匹配。首训练标签 5，首测试标签 7。此为实际 MNIST 缓存数据，不是合成样本。

## 3. 建模假设、变量与计划公式（未执行）

图片按行构成 T=28 的空间序列，每个位置 x_t∈R^28；每图一个数字类别 y∈{0,…,9}。计划 d_model=32、H=4、d_k=8、d_ff=64，一个 Encoder block，不使用 dropout。固定长度输入允许所有行相互注意，无须因果掩码或 padding。这些是教学简化与超参数选择，不是经验最优结论。

计划计算：

```text
E = X W_in + b_in + PE
Q_h = E W_Q,h ; K_h = E W_K,h ; V_h = E W_V,h
A_h = softmax(Q_h K_h^T / sqrt(d_k))
M = Concat(A_1 V_1, ..., A_H V_H) W_O + b_O
N = LayerNorm(E + M)
F = ReLU(N W_1 + b_1) W_2 + b_2
Z = LayerNorm(N + F)
r = mean(Z, axis=行序位置)
logits = r W_y + b_y
L = mean(-log softmax(logits)[y])
theta_new = theta_old - learning_rate * dL/dtheta
```

正弦位置编码、注意力、残差、LayerNorm、前馈网络参照 [Transformer 原始论文](https://arxiv.org/abs/1706.03762)；这里删去 Decoder 并接分类头。公式是计划，不构成本轮前向或梯度执行证据。

## 4. 方法选择理由（判断）

沿用熟悉的 MNIST 能把学习重点放到 Q/K/V、注意力、位置编码与梯度，按行表示使序列长度仅 28。一个 Encoder 与小维度减少初学者的计算图复杂度。CuPy 使基础数组写法尽量接近 NumPy，并满足 GPU 运行要求；不用自动求导，梯度正确性需要后续有限差分核验。对真实图像分类，CNN 或 ViT patch 表示也是可选方案，本轮不做性能比较，不宣称本方案更优。

## 5. 环境变更与设备核验（事实）

解释器 `F:\PythonProjects\deep_learning\.venv\Scripts\python.exe`；Python 3.13.15，NumPy 2.5.3，CuPy 14.2.0。本轮新增 cupy-cuda13x 14.2.0、cuda-pathfinder 1.8.3、nvidia-cuda-runtime 13.2.51；没有升级或替换现有 NumPy/PyTorch。依赖见 [GPU 后端版本](gpu_backend_requirements_v001.txt)。

复用现有 `F:\PythonProjects\deep_learning\.venv\Lib\site-packages\torch\lib` 的 NVRTC、cuBLAS 与 cuRAND DLL，使用 ctypes 预加载，不导入或调用 PyTorch。完整 CUDA 组件安装尝试在下载阶段取消，未完成安装；之后只安装上述三个所需包。第一次 GPU 数据运行因 NVRTC DLL 未被自动发现而失败；增加显式 DLL 加载后，最终独立验证通过。启动加载是环境适配，不是模型层封装。

机器硬件报告：`NVIDIA GeForce RTX 5060, 610.88, 8151 MiB`。CuPy 当前可见 GPU 数 1，指定 GPU 0，型号 NVIDIA GeForce RTX 5060，计算能力 12.0。CUDA Runtime API 版本 13020；Driver API 版本 13030。训练/测试输入及标签均为 CuPy ndarray，实际设备 `<CUDA Device 0>`。CPU 标识 `AMD64 Family 25 Model 33 Stepping 2, AuthenticAMD`，这是平台标识，不视作完整 CPU 商品型号。

## 6. 实际执行、验证与结果

在 PowerShell 使用项目解释器执行独立教学快照，**未声称在用户 VS Code 终端点击运行**。实际验证了数据形状、归一化、标签保持、有限性、GPU 数组设备，并调用 CUDA 同步等待计算完成。

后端基础检查：4×4 全 1 矩阵的 GPU 乘积所有元素为 4；稳定 softmax 行和约 0.99999994。复用熟悉的标量示例 y_hat=2w，真实 y=6，初值 w=1：手写梯度 -8，学习率 0.1，更新 w≈1.8，损失从 8 到 2.88000011。这是 GPU 数组与手写标量更新的后端检查，**不是 Transformer 训练**。

[教学说明](transformer_from_scratch_lesson_01.md) · [执行过的教学代码快照](2026-10-06_transformer_step01_code_v001.py) · [真实验证 JSON](2026-10-06_transformer_data_gpu_validation_v001.json) · [独立验证脚本](2026-10-06_transformer_validation_code_v001.py)。

教学源码 SHA256 `366c71e68a637bbd3a279b3296525a70cb8423ac745cbf37d7be1c7719838c32`。本轮无 Transformer 参数文件、预测文件、模型损失曲线或准确率，不能报告 Transformer 训练性能。

## 7. 版本变化与索引

Transformer 首版 v001，无上一版 Transformer 记录。本次新增路线、GPU 后端和第一步数据准备；既有 DNN、残差和 LSTM 记录保持不变。项目 modeling_records/index.md 新增本栏目；本交付目录保存可迁移副本和独立 index.md。后续每一步使用新的日期/版本，不覆盖本记录。

## 8. 限制与下一步

目前只有可执行的数据准备段，完整 Transformer 尚未实现。GPU 可运行已得到 CuPy 实际执行证据，但尚不能宣称模型使用 GPU 训练、训练提速或模型准确率。小模型、单样本操作和频繁打印可能受 GPU 启动与传输开销影响；后续采用小批次训练并实测耗时，不保证 GPU 一定更快。

固定首 5000/1000 子集是教学设置，未来测试结果不等同于完整 MNIST 基准。GPU float32 与 CPU float64 梯度核验将分开报告。NumPy 2.5.3 满足当前 CuPy 包的依赖约束，本轮所用基础操作已通过；其他尚未使用的 API 不能据此视作已验证。Windows CUDA DLL 复用依赖当前项目环境，如更换环境需再次检查。

下一步由用户输入并运行第 1 步，然后讲解和初始化输入投影、正弦位置编码与注意力参数，再逐步实现前向与手写反向，最后组织 GPU 训练和独立测试。该过程必须保留实际执行边界。
