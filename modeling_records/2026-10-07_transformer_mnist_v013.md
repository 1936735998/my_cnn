# MNIST 手写 Transformer 建模记录

日期：2026-10-07（Asia/Shanghai）；版本：v013。
阶段：逐位置前馈第一层与 ReLU 的局部前向 GPU 验证。本轮真实执行验证，不是仅文档更新；未训练。

## 范围与输入来源（事实）

用户输出 attention_norm `(28,32)`、第 0 行均值 -3.725290298461914e-08、方差 0.9999621510505676、GPU 0，符合前版核验结果。本轮只初始化 32→64 的前馈第一层并手写 ReLU；代码直接贴在聊天，不交付单独 Python 文件，不修改用户主文件。完整目标仍为逐步手写 Transformer 前向、反向和参数更新，在 GPU 训练。

独立进程读取执行保存的 `F:\PythonProjects\deep_learning\transformer_mnist.py` 前十二步，再执行本片段；不声称操作用户终端或实时内存。

沿用本地 MNIST `C:\Users\19367\.keras\datasets\mnist.npz`，SHA256 `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`。训练前 5000 图、测试前 1000 图；像素转 CuPy float32 除以 255，标签 int64。本轮没有下载、新增清洗、划分、洗牌或增强。只验证首张训练图局部前向，不评估测试集。输入来自多头注意力输出、残差相加与逐行 LayerNorm。

## 变量、假设与公式

序列长度 T=28，d_model=32，d_ff=64。A=attention_norm `(28,32)`，W_ff1 `(32,64)`，b_ff1 `(64,)`。

```text
Z = A @ W_ff1 + b_ff1: (28,64)
H[i,k] = max(Z[i,k], 0): (28,64)
```

同一组权重与偏置用于全部 28 个行位置，每行独立映射；前馈本步不额外混合行位置。参数沿用 float32 randn*0.1 与零偏置，不重新设置随机种子。新增参数 32*64+64=2112，当前累计 7328。输入维度、数值有限与 GPU 0 前提本轮实际核验。沿用图片行空间序列、无因果掩码的分类设置。

ReLU 保留正值、将负值置零，零值仍为零。保留 ff_hidden_linear 供后续手写激活导数使用，计划零点导数取 0；本轮未执行梯度。

## 方法选择理由（判断）

采用 64 维作为小型教学模型的隐藏宽度，未比较其他宽度或声称最优。先扩展表示，再通过 ReLU 引入非线性，下一步映射回 32 维。将两层拆开便于观察维度与激活；使用 CuPy 基础 dot/maximum，不调用现成前馈层或自动求导。

## 本步代码

```python
# ==================================================
# 前馈网络第一层：32 → 64，再经过 ReLU
# ==================================================

d_ff = 64

W_ff1 = cp.random.randn(d_model, d_ff).astype(cp.float32) * 0.1
b_ff1 = cp.zeros(d_ff, dtype=cp.float32)

# 对全部 28 行使用同一组参数
# (28, 32) @ (32, 64) + (64,) → (28, 64)
ff_hidden_linear = cp.dot(attention_norm, W_ff1) + b_ff1

# ReLU：正数保留，负数变为 0
ff_hidden = cp.maximum(ff_hidden_linear, cp.float32(0.0))

print("W_ff1:", W_ff1.shape)
print("b_ff1:", b_ff1.shape)
print("ff_hidden_linear:", ff_hidden_linear.shape)
print("ff_hidden:", ff_hidden.shape)
print("ff_hidden 的实际设备:", ff_hidden.device)
```

## 实际执行、验证与结果（事实）

解释器 `F:\PythonProjects\deep_learning\.venv\Scripts\python.exe`；Python 3.13.15、NumPy 2.5.3、CuPy 14.2.0。可见 GPU 1 张，使用 GPU 0，型号 NVIDIA GeForce RTX 5060。在独立 PowerShell 进程执行并同步 CUDA；CuPy 缓存在工作区 work/cupy_cache，Python 使用 -B。本轮未安装或调整依赖。

实际参数 `(32,64)`/`(64,)`，线性结果及激活结果 `(28,64)`；相关数组均有限 float32、GPU 0。CPU float64 线性与 ReLU 参考在 rtol=1e-5、atol=2e-6 下匹配，最大绝对误差分别 `4.76604112e-07`、`4.004076066e-07`。对实际线性结果的 ReLU 映射逐元素一致，输出非负；输入未改变。实际线性负值 913 个、正值 879 个，两类激活行为均有覆盖。按参数数组 size 核对新增 2112、累计 7328。

主文件执行前后 SHA256 一致：`e908f2fbe7c2ee104048ab332ee304b57be24fd022f8e0e3e711939e464c95ea`；片段 SHA256：`8b561a4fd2f1e01e7b44ff35796c10f15c0a70e173beb0893e0e981f12b212e8`。[本步代码](#本步代码) · [真实 GPU 验证证据](2026-10-07_transformer_ffn_hidden_validation_v013.json)。JSON 保存逐字代码、来源、设备、形状与实际误差和输出；相对链接已核对。局部前向核验不能证明识别性能。

## 用户注释中的逻辑核对

用户新增残差注释称“使用 encoder_input 计算得到，所以两者形状相同”。推理缺少输出维度约束：注意力输出被 W_O 映射到 d_model=32，才与输入同形。本轮前馈同样依赖 32 维输入，却输出 64 维，就是反例。这是注释表述问题，不影响当前运行；本轮指出而不修改主代码。

## 相比上一版的变化

v012 完成注意力后的 LayerNorm；v013 新增前馈第一层和 ReLU，产生 `(28,64)` 的隐藏表示。新增参数 2112，参数更新次数仍为 0。保留历史版本，生成本版工作区索引并按约定追加项目 modeling_records/index.md。

## 未解决限制与下一步

尚未前馈第二层、其残差和 LayerNorm、分类头、损失或梯度，尚未完成整个编码器；没有训练损失、准确率或 GPU 提速结论。

下一步初始化 W_ff2 `(64,32)`、b_ff2 `(32,)` 并投影回 32 维，之后才能与 attention_norm 做残差相加。继续逐步完成前向、手写反向和参数更新，并做必要的梯度数值核验。
