# MNIST 手写 Transformer 建模记录

日期：2026-10-07（Asia/Shanghai）；版本：v007。
阶段：稳定 softmax 局部前向 GPU 验证。本次真实执行了局部前向验证，不是仅文档更新；未训练。

## 范围与证据来源

用户贴出 K_transposed `(4,8,28)`、attention_scores `(4,28,28)` 与 GPU 0 信息。这些是用户运行输出，支持前一步维度及设备正确，不能单独证明模型识别能力。本轮只教授将这些分数转换为注意力权重；后续目标仍为逐步手写 Transformer 前向、反向和参数更新，并使用 GPU。

代码直接贴在聊天里，不另行交付 Python 代码文件，也不修改用户主文件。独立验证通过读取执行已保存的 `F:\PythonProjects\deep_learning\transformer_mnist.py` 重建前置变量，然后执行本步片段；不声称操作用户编辑器终端或实时内存。

## 输入来源、清洗与既定设置（事实）

沿用本地 MNIST `C:\Users\19367\.keras\datasets\mnist.npz`，SHA256 `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`。训练取前 5000 图、测试取前 1000 图；像素转 CuPy float32 并除以 255、标签为 int64。本轮没有下载、新增清洗、洗牌或数据增强。只验证第一张训练图的局部前向，不评估测试集。

输入分数来自当前随机初始化的输入投影、正弦位置编码、Q/K/V 投影、多头重排及 QKᵀ/sqrt(8)。序列长度 T=28，模型宽度 32，头数 H=4，每头宽度 8。本轮主文件已包含前六步，无需补充旧片段。

## 变量、公式与假设

对每个头 h、每个查询位置 i，沿键位置 j 做 softmax：

```text
m[h,i] = max_j S[h,i,j]
E[h,i,j] = exp(S[h,i,j] - m[h,i])
A[h,i,j] = E[h,i,j] / sum_k E[h,i,k]
attention_scores S: (4,28,28)
max/denominator with keepdims=True: (4,28,1)
attention_weights A: (4,28,28)
weight_sums: (4,28)
```

axis=-1 对每个查询行内部的 28 个键位置归一化，不能沿头轴归一化。keepdims=True 保留长度为 1 的末轴以便广播。同一行减去相同常数不会改变 softmax；对于有限输入，减去最大值使指数不超过 1，且每行至少一项指数为 1，因此避免指数溢出或全部指数为零。非常小的权重仍可能下溢为 0，不承诺所有浮点权重严格为正。

输入假设为有限分数；本轮实际检查通过。继续沿用图像分类全部行可互相关注、无因果掩码的设置。权重是可归一化的数值，但随机未训练权重不能当作有效笔画关注或模型解释。本步没有新增可训练参数。

## 本步代码

```python
# ==================================================
# 稳定 softmax：把匹配分数转换成注意力权重
# ==================================================

# 每一行减去该行的最大分数，避免 exp 溢出
scores_shifted = attention_scores - cp.max(
    attention_scores, axis=-1, keepdims=True
)

exp_scores = cp.exp(scores_shifted)

# 对每个查询位置对应的 28 个键位置归一化
attention_weights = exp_scores / cp.sum(
    exp_scores, axis=-1, keepdims=True
)

weight_sums = cp.sum(attention_weights, axis=-1)

print("attention_weights:", attention_weights.shape)
print("每行权重和的形状:", weight_sums.shape)
print("第 0 个头前 5 行的权重和:", weight_sums[0, :5])
print("attention_weights 的实际设备:", attention_weights.device)
```

## 方法选择理由（判断）

采用减最大值的稳定 softmax，而不是直接对原分数取指数，以免后续较大分数造成溢出。将分数、归一化和加权 V 分成小步，使初学者明确每个轴的意义。使用 CuPy 基础运算，不调用现成 softmax/注意力层或自动求导；保留 attention_weights 供后续手写反传使用。

## 实际执行与验证（事实）

解释器 `F:\PythonProjects\deep_learning\.venv\Scripts\python.exe`，Python 3.13.15、NumPy 2.5.3、CuPy 14.2.0。可见 GPU 1 张，指定 GPU 0，型号 NVIDIA GeForce RTX 5060。在独立 PowerShell 进程运行验证，完成 CUDA 同步；CuPy 缓存在工作区 work/cupy_cache，使用 -B 不写解释器目录字节码。本轮未安装或调整依赖。

实际 attention_weights `(4,28,28)`，weight_sums `(4,28)`；相关数组均为有限 float32，设备 GPU 0。权重在闭区间 [0,1]；每个查询行的权重和在绝对容差 1e-6 内接近 1，最大绝对误差 `1.192092896e-07`。第 0 个头前五行权重和：`[1.0, 1.0, 1.0, 1.0, 1.0]`。

CPU float64 使用原分数的指数定义独立核对，在 rtol=1e-5、atol=1e-7 下匹配，最大绝对误差 `6.856624116e-09`。另将 `[10000,9999,9998]` 和 `[-10000,-10001,-10002]` 输入同一片段，两行均得到有限、归一化且匹配理论值的结果，验证大幅度分数下的稳定性。这是合成数值样例，不是 MNIST 训练结果。

本次实际权重范围 `0.02456626`～`0.05063576`；不据此推断识别性能。输入分数数组未改变。用户主文件执行前后 SHA256 一致：`7a3f40e63dfe05fb2f6315dbe6efa6dec38720f032e328e9ce916c919c22dcdc`。聊天片段 SHA256：`b9919c0085000986f72ee4ac9457d8f4cd405c587444dacc48dc074a225ebff6`。

[本步代码](#本步代码) · [真实 GPU 验证证据](2026-10-07_transformer_softmax_validation_v007.json)。JSON 保存逐字片段、来源、实际设备、环境、数值误差与输出；相对链接均已核对。

## 相比上一版的变化

v006 产生尚未归一化的匹配分数；v007 增加稳定 softmax，产生 `(4,28,28)` 的注意力权重并验证每行归一化。未新增训练参数或参数更新。保留旧版本，生成当前工作区索引，并按用户约定追加项目 modeling_records/index.md。

## 未解决限制与下一步

尚未加权汇总 V、合并多头、输出投影、残差、LayerNorm、前馈、分类头、损失或梯度。优化更新次数 0，没有训练损失、测试准确率或 GPU 提速结论。局部前向正确不能证明完整模型或反向正确。

下一步计算 attention_weights @ V_heads，得到每个头各查询行的加权特征 `(4,28,8)`。后续逐步手写反向与参数更新，并对梯度进行必要的数值核验。
