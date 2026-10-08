# MNIST 手写 Transformer 建模记录

日期：2026-10-07（Asia/Shanghai）；版本：v012。
阶段：手写注意力残差后 LayerNorm 的局部前向 GPU 验证。本轮真实执行验证，不是仅文档更新；未训练。

## 范围与证据来源

用户输出 attention_residual `(28,32)`、单行 `(32,)`、GPU 0，与前一步一致。本轮只手写逐行 LayerNorm 的均值、总体方差、稳定归一化与可训练缩放/偏移。代码直接贴在聊天，不交付单独 Python 文件，不修改主文件；整体目标仍为手写完整前向、反向和参数更新并在 GPU 训练。

独立进程读取执行已保存的 `F:\PythonProjects\deep_learning\transformer_mnist.py` 前十一步，再执行本步片段；不声称操作编辑器终端或实时内存。

## 输入来源与处理（事实）

沿用本地 MNIST `C:\Users\19367\.keras\datasets\mnist.npz`，SHA256 `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`。训练前 5000 图、测试前 1000 图；像素转 CuPy float32 除以 255，标签 int64。本轮未下载、新增清洗、划分、洗牌或增强。只核验首张训练图的局部前向，不评估测试集。

输入来自含位置编码的输入投影，经多头自注意力、输出投影并与 encoder_input 残差相加。用户主文件已保存前十一步，无需补充旧片段。

## 变量、假设与公式

R=attention_residual `(28,32)`；每行特征数 D=32。归一化沿最后一维；均值和方差保持 `(28,1)`。逐行定义：

```text
mu[i] = sum_k R[i,k] / 32
v[i] = sum_k (R[i,k]-mu[i])**2 / 32
z[i,k] = (R[i,k]-mu[i]) / sqrt(v[i]+eps)
Y[i,k] = gamma[k]*z[i,k] + beta[k]
eps = float32(1e-5)，总体方差 ddof=0
gamma_attn: (32,) 初始化为 1
beta_attn: (32,) 初始化为 0
attention_norm: (28,32)
```

epsilon 放在开平方内部，防止零方差除零；使用 centered**2 的均值计算方差。gamma/beta 在全部位置共享、按特征广播，新增 64 个可训练参数，当前累计 5216。保留 centered_attn、inv_std_attn、normalized_attn，便于后续手写反向。

初始 gamma=1、beta=0 时，数学上每行输出均值为 0，方差为 v/(v+eps)，浮点运算允许误差；方差不严格为 1。训练后 gamma/beta 改变，最终输出不再保证均值 0、方差接近 1。继续采用图片行空间序列、无因果掩码、先残差再 LayerNorm 的教学结构。输入有限与维度一致假设本轮实际核验。

## 方法选择理由（判断）

逐行归一化让每个位置独立处理自己的 32 个表示特征；该计算轴与 softmax 沿键位置归一化的轴语义不同。显式写出每一步而不使用现成 LayerNorm 或自动求导，便于理解数值稳定性与后续反向。选择 eps=1e-5 作为教学配置，未比较其他 epsilon 的训练性能。

## 本步代码

```python
# ==================================================
# 注意力残差之后的 LayerNorm
# ==================================================

gamma_attn = cp.ones(d_model, dtype=cp.float32)
beta_attn = cp.zeros(d_model, dtype=cp.float32)
ln_eps = cp.float32(1e-5)

# 每一行的 32 个特征分别计算均值和方差
mean_attn = cp.mean(attention_residual, axis=-1, keepdims=True)
centered_attn = attention_residual - mean_attn
var_attn = cp.mean(centered_attn ** 2, axis=-1, keepdims=True)

# epsilon 放在开平方内部，避免零方差导致除零
inv_std_attn = cp.float32(1.0) / cp.sqrt(var_attn + ln_eps)
normalized_attn = centered_attn * inv_std_attn

# 可训练的逐特征缩放和偏移
attention_norm = normalized_attn * gamma_attn + beta_attn

print("gamma_attn:", gamma_attn.shape)
print("beta_attn:", beta_attn.shape)
print("mean_attn:", mean_attn.shape)
print("var_attn:", var_attn.shape)
print("attention_norm:", attention_norm.shape)
print("第 0 行输出的均值:", float(cp.mean(attention_norm[0])))
print("第 0 行输出的方差:", float(cp.var(attention_norm[0])))
print("attention_norm 的实际设备:", attention_norm.device)
```

## 实际执行、验证与结果（事实）

解释器 `F:\PythonProjects\deep_learning\.venv\Scripts\python.exe`；Python 3.13.15、NumPy 2.5.3、CuPy 14.2.0；可见 GPU 1 张，使用 GPU 0，型号 NVIDIA GeForce RTX 5060。在独立 PowerShell 进程执行并同步 CUDA，CuPy 缓存在工作区 work/cupy_cache，Python 使用 -B；本轮未安装或调整依赖。

参数 `(32,)`，均值/方差 `(28,1)`，输出 `(28,32)`；相关数组均为有限 float32、GPU 0，输入未改变。CPU float64 参考在 rtol=1e-5、atol=2e-6 下匹配，最大绝对误差 `3.318116182e-07`。各行输出均值与 0 接近，最大绝对值 `6.624031812e-08`；逐行方差与 v/(v+eps) 参考匹配。

第 0 行 GPU float32 输出均值 `-3.725290298e-08`、方差 `0.9999621511`。另使用非平凡逐特征 gamma/beta 核验广播与缩放/偏移，最大参考误差 `3.776990356e-07`；三行常数输入的零方差样例稳定输出 beta，逐元素一致。合成样例只验证数值性质，不代表 MNIST 训练。按实际参数数组 size 核对新增 64、累计 5216。

主文件执行前后 SHA256 一致：`139397175fdde4bc2c425858c4962606124232d6b229ecfd0a2653262c21400b`；片段 SHA256：`eb4557f57ba77b6804edc659b00e96deaae2d1491b2627e48271e2df886161f9`。[本步代码](#本步代码) · [真实 GPU 验证证据](2026-10-07_transformer_attention_layernorm_validation_v012.json)。JSON 保存逐字代码、来源、实际设备、形状、误差与输出；相对文件链接已核对。

## 相比上一版的变化

v011 只残差相加；v012 增加逐行 LayerNorm 与 gamma_attn/beta_attn，完成本教学注意力子层的残差后归一化。新增参数 64，更新次数仍为 0。保留历史记录，生成本版工作区索引并按约定追加项目 modeling_records/index.md。

## 未解决限制与下一步

尚未前馈子层、其残差与 LayerNorm、分类头、损失、梯度或优化更新，尚未完成整个编码器。没有训练损失、识别准确率或 GPU 提速结论，局部前向正确不证明完整反向正确。

下一步添加逐位置前馈网络，先将 32 维映射到较宽隐藏层再使用激活函数，随后映射回 32 维；之后补全分类和手写反向并做必要的梯度数值核验。
