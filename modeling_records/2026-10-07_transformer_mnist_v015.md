# MNIST 手写 Transformer 建模记录

日期：2026-10-07（Asia/Shanghai）；版本：v015。
阶段：前馈子层残差连接局部 GPU 前向。本轮真实执行验证，不是仅文档更新；未训练。

## 范围、来源与处理（事实）

用户贴出 ff_hidden `(28,64)`、W_ff2 `(64,32)`、b_ff2 `(32,)`、ff_output `(28,32)` 与 GPU 0 信息，与前版一致。本轮只把前馈输出和前馈输入相加。代码直接贴在聊天，不交付独立 Python 文件，不修改主文件。整体目标仍为逐步手写 Transformer 前向、反向、参数更新，并用 GPU 训练。

独立进程读取执行 `F:\PythonProjects\deep_learning\transformer_mnist.py` 前十四步，再执行片段；不声称操作用户终端或实时内存。沿用本地 MNIST `C:\Users\19367\.keras\datasets\mnist.npz`，SHA256 `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`。训练前 5000、测试前 1000 图，像素 float32 除以 255、标签 int64。本轮没有新增下载、清洗、划分、洗牌或增强，仅核验首张训练图局部前向，不评估测试集。

attention_norm 来源为注意力投影的残差后 LayerNorm，是前馈子层输入。ff_output 来源为 ReLU(attention_norm@W_ff1+b_ff1)@W_ff2+b_ff2。主文件已保存前十四步，不需补充旧片段。

## 变量、假设与公式

T=28，d_model=32，d_ff=64。A=attention_norm，F(A)=ff_output，均为 `(28,32)`。

```text
R_ffn = A + FFN(A)
ff_residual[i,k] = attention_norm[i,k] + ff_output[i,k]
ff_residual: (28,32)
```

对应位置、对应特征逐元素相加，跳连接回前馈子层输入 attention_norm。两个输入同形、finite float32、GPU 0 前提均实际核验。继续沿用图片行空间序列、无因果掩码、先残差再 LayerNorm 的结构。本步新增参数 0，当前累计 9408。

## 方法选择理由（判断）

以该子层输入作为直接分支，再叠加前馈变换结果，与前一注意力子层的残差结构一致。将残差与第二次 LayerNorm 分步教授，便于确认跳连起点及后续反向分支。采用基础 CuPy 加法，不调用现成层或自动求导；本轮不验证训练收益或梯度。

## 本步代码

```python
# ==================================================
# 前馈子层的残差连接
# ==================================================

# (28, 32) + (28, 32) → (28, 32)
ff_residual = attention_norm + ff_output

print("ff_residual:", ff_residual.shape)
print("一行残差相加后的特征:", ff_residual[0].shape)
print("ff_residual 的实际设备:", ff_residual.device)
```

## 实际执行、验证与结果（事实）

解释器 `F:\PythonProjects\deep_learning\.venv\Scripts\python.exe`；Python 3.13.15、NumPy 2.5.3、CuPy 14.2.0。可见 GPU 1 张，使用 GPU 0，型号 NVIDIA GeForce RTX 5060。独立 PowerShell 进程执行并同步 CUDA，CuPy 缓存在工作区 work/cupy_cache，Python 使用 -B。本轮未安装或调整依赖。

两个输入和输出均为 `(28,32)`、有限 float32、GPU 0，单行 `(32,)`。CPU float32 逐元素加法参考与 GPU 结果逐元素一致，最大绝对误差 `0.0`；两个输入执行前后不变。按参数 size 核对累计 9408。局部前向正确不证明模型识别能力。

主文件执行前后 SHA256 一致：`a263133925e03850a9c98d0832c8a585c5a51c7265f4da56d37a8b2f0f278b21`；片段 SHA256：`3f891e806ff5d716d992cc557dde88963b3aecc3f66a46acb160bb100e2fc9dd`。[本步代码](#本步代码) · [真实 GPU 验证证据](2026-10-07_transformer_ffn_residual_validation_v015.json)。JSON 保存来源、逐字代码、环境、设备和实际输出；相对链接已核对。

## 版本变化、限制与下一步

相比 v014 的前馈输出，本版新增 ff_residual `(28,32)`，参数新增 0、更新次数 0。保留历史记录，生成本版工作区索引并按约定追加项目 modeling_records/index.md。

尚未前馈残差后 LayerNorm、分类头、损失、梯度或训练；当前尚未完成整个编码器，没有准确率、梯度验证或 GPU 提速结论。下一步以独立 gamma_ffn/beta_ffn 对 ff_residual 逐行 LayerNorm，得到 encoder_output，然后继续分类和手写反向。
