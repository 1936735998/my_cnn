# MNIST 手写 Transformer 建模记录

日期：2026-10-07（Asia/Shanghai）；版本：v019。
阶段：分类稳定 softmax 的单图 GPU 前向验证。本轮真实执行验证，不是仅文档更新；未损失计算、反向或训练。

## 范围、来源与处理（事实）

用户输出 W_cls `(32,10)`、b_cls/logits `(10,)`、GPU 0，与前版一致。本轮只将 logits 归一化为十类概率。代码直接贴在聊天，不交付独立 Python 文件，不修改主文件；目标仍为手写完整前向、反向和参数更新，并在 GPU 训练。

独立进程读取执行已保存的 `F:\PythonProjects\deep_learning\transformer_mnist.py` 前十八步，再执行片段；不声称操作用户终端或实时内存。主文件已保存前十八步，无需补充旧片段。

沿用本地 MNIST `C:\Users\19367\.keras\datasets\mnist.npz`，SHA256 `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`。训练前 5000、测试前 1000 图，像素转 CuPy float32 除以 255、标签 int64。本轮没有下载、新增清洗、划分、洗牌或增强。仅核验首张训练图局部前向，不评估测试集。输入来自图片行序列经编码器、位置平均池化和线性分类头的 10 个分数。

## 变量、假设与公式

z=logits `(10,)`，c=0..9 对应数字类别；m=max_c z[c]。

```text
s[c] = z[c]-m
e[c] = exp(s[c])
p[c] = e[c] / sum_j e[j]
probabilities: (10,)
```

沿最后一维类别轴归一化，keepdims 保留归约轴便于广播。同一常数平移不改变 softmax；最大项指数为 1。极小概率可能浮点下溢为 0，不承诺严格大于 0。当前输入与各中间量有限、float32、GPU 0 的前提本轮实际核验。未新增参数，累计仍 9802。继续沿用图片行空间序列、无因果掩码分类设置。

## 方法选择理由（判断）

采用减最大值的稳定 softmax，复用注意力权重阶段已验证的方法。本次显式沿类别轴计算，保留 shifted_logits/exp_logits 供下一步稳定交叉熵使用。当前分类输入重新核验 CPU 参考与归一化，不重复以前已通过的大幅分数边界样例。采用 CuPy 基础操作，不调用现成 softmax 层或自动求导；随机未训练输出不能代表可信识别能力。

## 本步代码

```python
# ==================================================
# 稳定 softmax：分类分数 → 分类概率
# ==================================================

# 沿最后一维的 10 个类别，减去最大分数
shifted_logits = logits - cp.max(logits, axis=-1, keepdims=True)
exp_logits = cp.exp(shifted_logits)

# 每个类别的指数值除以全部类别的指数值之和
probabilities = exp_logits / cp.sum(exp_logits, axis=-1, keepdims=True)

print("probabilities:", probabilities.shape)
print("10 个分类概率:", probabilities)
print("概率之和:", float(cp.sum(probabilities)))
print("probabilities 的实际设备:", probabilities.device)
```

## 实际执行、验证与结果（事实）

解释器 `F:\PythonProjects\deep_learning\.venv\Scripts\python.exe`；Python 3.13.15、NumPy 2.5.3、CuPy 14.2.0。可见 GPU 1 张，使用 GPU 0，型号 NVIDIA GeForce RTX 5060。独立 PowerShell 进程执行并同步 CUDA，CuPy 缓存在工作区 work/cupy_cache，Python 使用 -B。本轮未安装或调整依赖。

原分数、中间量及概率均为 `(10,)`、有限 float32、GPU 0，概率在闭区间 [0,1]，输入分数未改变。概率和 `1`，与 1 的绝对误差 `0`，通过容差 1e-6。CPU float64 用 logaddexp.reduce 求对数归一化常数，再取指数作为独立参考，rtol=1e-5、atol=1e-7 下匹配，最大绝对误差 `1.722367667e-08`。参数累计按实际 size 核对仍为 9802；JSON 保存实际十类概率，不把它们当作准确率或训练结果。

主文件执行前后 SHA256 一致：`00c35fde45145337d9148aed801980b72b583a03b17f2e6bcb9cea57e5cd2e58`；片段 SHA256：`3b662ee20c78071b94392c2dd1afb41f5df8564fe3f331b2518eec92acc99dbe`。[本步代码](#本步代码) · [真实 GPU 验证证据](2026-10-07_transformer_class_softmax_validation_v019.json)。JSON 包含来源、逐字代码、环境、设备、误差和实际输出；相对链接已核对。

## 版本变化、限制与下一步

相比 v018 的原始分数，本版新增 shifted_logits、exp_logits 与 probabilities，获得未训练的分类概率分布。新增参数 0、更新次数 0。保留历史记录，生成本版工作区索引并按约定追加项目 modeling_records/index.md。

尚未交叉熵、预测展示、反向或训练，没有识别准确率、概率校准或 GPU 提速结论。下一步依据首图真实标签计算交叉熵，使用 log(sum(exp_logits))-shifted_logits[target] 避免概率下溢后取对数；随后逐步手写反向和 GPU 更新，并做必要数值核验。
