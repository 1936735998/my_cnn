# MNIST 手写 Transformer 建模记录

日期：2026-10-07（Asia/Shanghai）；版本：v016。
阶段：前馈残差后第二次 LayerNorm 的局部 GPU 验证，简化单层 Encoder 块前向完成。本轮真实执行验证，不是仅文档更新；尚未分类或训练。

## 范围、来源与处理（事实）

用户贴出 ff_residual `(28,32)`、单行 `(32,)`、GPU 0，与前版一致。本轮只初始化第二组归一化参数并计算编码器块输出。代码直接贴在聊天，不交付独立 Python 文件，不修改主文件。整体目标仍为手写 Transformer 图像分类前向、反向与参数更新，在 GPU 训练。

独立进程读取执行已保存的 `F:\PythonProjects\deep_learning\transformer_mnist.py` 前十五步，再执行片段；不声称操作终端或实时内存。主文件已保存前十五步，无需补充旧片段。

沿用本地 MNIST `C:\Users\19367\.keras\datasets\mnist.npz`，SHA256 `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`。训练前 5000、测试前 1000 图，像素转 CuPy float32 除以 255、标签 int64。本轮没有下载、新增清洗、划分、洗牌或增强，仅核验首张训练图局部前向，不评估测试集。输入为 attention_norm 与其前馈结果的残差和。

## 变量、假设与公式

T=28，D=32，R=ff_residual `(28,32)`。

```text
mu[i] = mean_k R[i,k]
v[i] = mean_k (R[i,k]-mu[i])**2
z[i,k] = (R[i,k]-mu[i]) / sqrt(v[i]+eps)
encoder_output[i,k] = gamma_ffn[k]*z[i,k] + beta_ffn[k]
mean_ffn/var_ffn: (28,1)
gamma_ffn/beta_ffn: (32,)，初始化 1/0
encoder_output: (28,32)
```

逐行沿特征轴归一化，总体方差 ddof=0。epsilon 沿用固定 float32(1e-5)，位于开平方内部。两次 LayerNorm 的 gamma/beta 独立创建并分别训练，可以共用固定 epsilon。新增参数 64，当前累计 9472。

初始 gamma=1、beta=0 时，输出均值约 0、方差为 v/(v+eps)，不能要求严格为 1；训练后的仿射输出不保证这些初始统计性质。保留中间变量供后续手写反向。维度、finite float32、GPU 0 前提均核验通过；沿用图片行空间序列、无因果掩码与残差后归一化的结构。

## 方法选择理由（判断）

沿用已验证的 LayerNorm 公式，并对前馈子层结果应用独立参数，使两处归一化分别学习缩放和偏移。本轮核验新输入、结果和参数独立性，不重复上版已经通过的全部边界样例。采用基础 CuPy 操作，不调用现成层或自动求导。

## 本步代码

```python
# ==================================================
# 前馈残差之后的第二次 LayerNorm
# ==================================================

# 与第一次 LayerNorm 分别使用自己的可训练参数
gamma_ffn = cp.ones(d_model, dtype=cp.float32)
beta_ffn = cp.zeros(d_model, dtype=cp.float32)

# 继续对每行的 32 个特征计算均值和方差
mean_ffn = cp.mean(ff_residual, axis=-1, keepdims=True)
centered_ffn = ff_residual - mean_ffn
var_ffn = cp.mean(centered_ffn ** 2, axis=-1, keepdims=True)

# ln_eps 沿用第一次 LayerNorm 设置的固定常数
inv_std_ffn = cp.float32(1.0) / cp.sqrt(var_ffn + ln_eps)
normalized_ffn = centered_ffn * inv_std_ffn
encoder_output = normalized_ffn * gamma_ffn + beta_ffn

print("gamma_ffn:", gamma_ffn.shape)
print("beta_ffn:", beta_ffn.shape)
print("mean_ffn:", mean_ffn.shape)
print("var_ffn:", var_ffn.shape)
print("encoder_output:", encoder_output.shape)
print("第 0 行输出的均值:", float(cp.mean(encoder_output[0])))
print("第 0 行输出的方差:", float(cp.var(encoder_output[0])))
print("encoder_output 的实际设备:", encoder_output.device)
```

## 实际执行、验证与结果（事实）

解释器 `F:\PythonProjects\deep_learning\.venv\Scripts\python.exe`；Python 3.13.15、NumPy 2.5.3、CuPy 14.2.0。可见 GPU 1 张，使用 GPU 0，型号 NVIDIA GeForce RTX 5060。独立 PowerShell 进程执行并完成 CUDA 同步；CuPy 缓存在工作区 work/cupy_cache，Python 使用 -B，本轮未安装或调整依赖。

相关数组均有限 float32、GPU 0，均值/方差 `(28,1)`、参数 `(32,)`、输出 `(28,32)`。CPU float64 参考在 rtol=1e-5、atol=2e-6 下匹配，最大绝对误差 `3.780247688e-07`；各行均值最大绝对值 `3.119930625e-08`，逐行方差与 v/(v+eps) 参考匹配。第 0 行 GPU float32 输出均值 `1.490116119e-08`、方差 `0.9999919534`。

第二组 gamma/beta 与第一组对应参数是不同对象且 GPU 指针不同；输入未改变。按实际参数 size 核对新增 64、累计 9472。主文件执行前后 SHA256 一致：`1e964fbe975d06002d118a355a06d24ab61f13bb04a93c2d7e69f5e0595049cb`；片段 SHA256：`f9a85d49d9bd62df9b78645f3b5527619b133f889d49807724df837464e21e79`。[本步代码](#本步代码) · [真实 GPU 验证证据](2026-10-07_transformer_ffn_layernorm_validation_v016.json)。JSON 保存来源、逐字代码、环境、设备、实际误差与输出；相对链接已核对。局部计算核验不证明识别性能或完整反向正确。

## 版本变化、限制与下一步

相比 v015 仅残差相加，本版新增第二次 LayerNorm 和 encoder_output，完成简化单层 Encoder 块的前向；新增参数 64、更新次数 0。保留历史记录，生成本版工作区索引并按约定追加项目 modeling_records/index.md。

尚未图片级汇总、分类头、损失、梯度或训练，没有准确率、训练收益或 GPU 提速结论。下一步沿位置轴平均汇总 `(28,32)` 编码器输出为一张图的 `(32,)` 表示，再添加 10 类分类头，随后继续手写反向与更新并做必要的数值核验。
