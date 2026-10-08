# MNIST 手写 Transformer 建模记录

日期：2026-10-07（Asia/Shanghai）；版本：v006。
阶段：多头缩放点积匹配分数验证。未计算 softmax、V 汇总、损失、反向或训练。

## 问题范围与用户证据

用户已贴出 Q/K/V 的 GPU 0 设备信息，以及 Q_heads/K_heads/V_heads `(4,28,8)`、每头 8 维、第 0 头 `(28,8)`。这些输出与 v005 设置一致。当前保存的用户主文件已包含前五步，本轮直接以其为前置准备，未临时补充以前片段。

本轮仅教授第 6 步匹配分数，代码直接贴在聊天里；不另行交付 Python 代码文件，不改写用户主 `F:\PythonProjects\deep_learning\transformer_mnist.py`。后续仍计划手写梯度、参数更新与 GPU 训练。

## 输入来源与处理（事实）

沿用本地 MNIST `C:\Users\19367\.keras\datasets\mnist.npz`，SHA256 `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`，与前版一致。训练前 5000 图、测试前 1000 图，像素 CuPy float32 除以 255、标签 int64。本轮未新增数据下载、清洗、划分、洗牌或数据增强。

助手在独立进程读取并执行用户保存的前五步，生成首图的多头 Q/K/V；没有读取或改变用户 VS Code 实时内存。当前 Q_heads 与 K_heads 均为 `(4,28,8)`，来源为含位置编码的 32 维输入通过不同投影后按头重排。原始行输入仍是空间行扫描表示，不是时间序列。

## 变量、假设与公式

头数 H=4、序列长度 T=28、每头查询与键维度 d_k=8。对每个头 h 独立计算：

```text
K_transposed: (4,8,28)
S[h] = Q_heads[h] @ K_heads[h].T / sqrt(8)
S[h,i,j] = sum_m(Q_heads[h,i,m] * K_heads[h,j,m]) / sqrt(8)
attention_scores: (4,28,28)
```

行 i 表示发起查询的图片行，列 j 表示被匹配的图片行。保留头轴，只交换 K 的后两个轴；cp.matmul 在各个头内执行矩阵乘法，不混合不同头。

公式参考 [Transformer 原论文第 3.2.1 节](https://arxiv.org/html/1706.03762v7)。缩放采用每头宽度 8，不能改用总宽度 32。缩放有助于控制较高维度下内积的尺度，但不会把分数限制在固定区间；这些分数可为负，尚未归一化，不能当作概率。本步没有因果掩码，沿用图像分类全部行可相互关注的设置。

本步不新增待训练参数，当前随机投影尚未经过任务训练。

## 本步代码

直接追加在多头拆分之后：

```python
# ==================================================
# 计算每个头的匹配分数
# ==================================================

K_transposed = K_heads.transpose(0, 2, 1)

attention_scores = cp.matmul(Q_heads, K_transposed)
attention_scores = attention_scores / cp.sqrt(cp.float32(head_dim))

print("K_transposed:", K_transposed.shape)
print("attention_scores:", attention_scores.shape)
print("第 0 个头的分数矩阵:", attention_scores[0].shape)
print("attention_scores 的实际设备:", attention_scores.device)
```

## 方法选择理由（判断）

单独学习转置、批量矩阵乘法和 sqrt(d_k) 缩放，再讲 softmax，便于区分匹配分数与注意力权重。保留所有 28 个行位置，为后续每行对全部行分配权重准备输入。采用基础 CuPy 操作，不调用现成注意力层或自动求导。

## 实际执行、验证与结果（事实）

通过 `F:\PythonProjects\deep_learning\.venv\Scripts\python.exe` 在 PowerShell 独立执行，不声称在用户 VS Code 终端点击运行。受限进程第一次未能启动项目 Python，随后通过执行权限审核成功运行同一脚本；无训练或模型参数更新。CuPy 缓存设置到当前工作区 work/cupy_cache，Python 使用 -B 不新增解释器目录字节码缓存。

Python 3.13.15、NumPy 2.5.3、CuPy 14.2.0；可见 GPU 1 张，指定 GPU 0，型号 NVIDIA GeForce RTX 5060。本轮没有安装或调整依赖。

实际 K_transposed `(4,8,28)`、attention_scores `(4,28,28)`；两者均为有限 float32、设备 `<CUDA Device 0>`。K 转置与 CPU 交换末两轴结果逐元素一致，头轴保持对应，原 Q_heads/K_heads 均未改变。

分别在 CPU float64 上计算四个头的 QKᵀ/sqrt(8)，GPU 结果在 `rtol=1e-5, atol=2e-6` 下匹配，最大绝对误差 `5.83343619e-08`。本次随机参数下分数范围约 `-0.49995145`～`0.35313633`，第 0 头 (0,0) 分数约 `-0.11117182`；这些是本次局部前向结果，不是识别性能。CUDA 同步完成。

用户主文件执行前后 SHA256 一致：`aa4472c370d3c55c3622dccfe6985ef9539808b7e2427b780670b5f1b076fa2a`。聊天片段 SHA256：`4acedfd53de9575992cd5c41a2b0461c0bdcb901b50485db6ba43508d9499391`。[本步代码](#本步代码) · [真实 GPU 验证结果](2026-10-07_transformer_attention_scores_validation_v006.json)。JSON 保存逐字代码、源信息、配置、形状、设备、参考误差与输出。

## 注释核对

用户新增注释将 dimension_index 描述为“16行1列”。实际 `cp.arange(0,d_model,2)` 是一维数组，形状 `(16,)`；position 经 reshape 才是 `(28,1)`。这是注释表述问题，本轮仅指出，不改写用户代码，不把它当作执行错误。

## 相比上一版的变化

v005 只拆分多头；v006 新增 K 的末两轴转置与四头缩放点积分数，输出由投影表示进入 `(4,28,28)` 的位置匹配表。没有新增训练参数或优化更新。保留所有此前版本与证据，生成本交付索引并按用户约定追加项目索引；不覆盖旧记录。

## 未解决限制与下一步

尚未执行 softmax、加权 V、多头合并与输出投影、残差、LayerNorm、前馈、分类头、损失或梯度。优化次数 0，没有训练损失、准确率或 GPU 提速结论。单图局部计算正确不能证明模型学会图像识别。

下一步沿最后一维（键位置 j）对每个头、每个查询行做稳定 softmax，将分数转换为注意力权重；再逐步加权汇总 V。后续完整反向需要单独有限差分验证。
