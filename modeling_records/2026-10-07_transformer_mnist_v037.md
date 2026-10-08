# MNIST 手写 Transformer 建模记录

日期：2026-10-07（Asia/Shanghai）；版本：v037。
阶段：QKV 拆头操作手写 GPU 反向、独立块映射及 CPU 局部差分验证。本轮真实执行，不是仅文档更新；未完成 QKV 投影、全网络反向、参数更新或训练。

## 范围、来源与处理（事实）

用户报告 d_Q_heads 与 d_K_heads `(4,28,8)`、GPU 0，前版已保存 d_V_heads。本版将 Q/K/V 三份头布局梯度还原为二维 d_Q、d_K、d_V。目标仍为逐节点手写 Transformer 全部反向与更新，最终 GPU 训练。代码贴于聊天，不改用户主文件、不交付完整 Python 模型文件。

独立进程读取执行已保存的 `F:\PythonProjects\deep_learning\transformer_mnist.py` 前三十六步，再执行本片段；未操作用户终端实时内存。主文件 SHA256 `fe20eeb41c1d6a280d0ccc577530f2a8814eea6875a589dde39cf25b59d3db1b`，执行前后不变。

沿用本地 MNIST `C:\Users\19367\.keras\datasets\mnist.npz`，SHA256 `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`。训练前 5000、测试前 1000 图，像素 CuPy float32 除以 255、标签 int64。本轮仅首张训练图、标签 5，无新增清洗、下载、划分、洗牌、增强或测试评估。输入为 Q/K/V 实际前向拆头布局和对应梯度。

## 假设、变量与公式

延续图片行空间序列、单层编码器、位置均值池化、十类单图交叉熵。头数 4、位置数 28、每头宽度 8、模型宽度 32。每个位置的特征按第 0、1、2、3 头连续分组，前向只是元素重新排列。

```text
X=Q/K/V: (28,32)
前向：X.reshape(28,4,8).transpose(1,0,2) -> X_heads (4,28,8)
X_heads[h,i,k] = X[i,h*8+k]
反向：d_X_heads.transpose(1,0,2).reshape(28,32) -> d_X (28,32)
d_X[i,h*8+k] = d_X_heads[h,i,k]
```

本步先转置恢复位置轴在前，再 reshape 拼回每行 32 个特征。仅还原排列，不求和、不缩放。最终 copy 分配独立存储，保存原始头梯度供核验。新增参数 0，当前参数仍 9802。

## 方法选择理由（判断）

按前向操作逆序处理三份梯度，使用相同规则分别对应独立的 Q/K/V 投影。显式保留顺序避免位置与头混合，后续线性投影反向得到匹配的二维梯度。使用基础 CuPy transpose/reshape/copy，无自动求导或现成层，参数保持前向值。

## 本步代码

```python
# ==================================================
# 反向第十七步：逆转 Q、K、V 的拆头
# ==================================================

# 先恢复位置轴在前，再拼回每个位置的 32 个特征
# (4, 28, 8) → (28, 4, 8) → (28, 32)
d_Q = d_Q_heads.transpose(1, 0, 2).reshape(seq_len, d_model).copy()
d_K = d_K_heads.transpose(1, 0, 2).reshape(seq_len, d_model).copy()
d_V = d_V_heads.transpose(1, 0, 2).reshape(seq_len, d_model).copy()

print("d_Q:", d_Q.shape)
print("d_K:", d_K.shape)
print("d_V:", d_V.shape)
print("d_Q 的实际设备:", d_Q.device)
print("d_K 的实际设备:", d_K.device)
print("d_V 的实际设备:", d_V.device)
```

## 实际执行、验证与结果（事实）

解释器 `F:\PythonProjects\deep_learning\.venv\Scripts\python.exe`；Python 3.13.15、NumPy 2.5.3、CuPy 14.2.0。可见 GPU 1 张，GPU 0 为 NVIDIA GeForce RTX 5060。真实执行并同步 CUDA，解释器带 -B、缓存在工作区 work/cupy_cache，本轮未调整依赖。

三份输出梯度均 `(28,32)`、float32、GPU 0、全部有限，具有独立连续存储。所有 20 组参数、相关前向缓存、loss、三份头梯度与直连梯度均未改变。

CPU 显式将每份头梯度按头序 concatenate，全部元素与 GPU 结果精确相同；再按每行 8 个特征切块并 stack，精确恢复原始头梯度，确认位置、头和特征轴顺序。

CPU float64 独立从二维前向 X 按每行 8 特征切块再 stack 成头布局，构造 sum(split(X)*固定头梯度)。Q/K/V 每份各 896 分量做 ±1e-5 中心差分，共 2688 分量；rtol=1e-6、atol=1e-9 下全部通过。最大绝对误差：d_Q `1.990768661e-14`，d_K `1.915827252e-14`，d_V `1.317168596e-12`。

这是三份拆头操作的局部雅可比向量积验证，不代表已经反向经过 QKV 投影或全网络验证。实际输出：

```text
d_Q: (28, 32)
d_K: (28, 32)
d_V: (28, 32)
d_Q 的实际设备: <CUDA Device 0>
d_K 的实际设备: <CUDA Device 0>
d_V 的实际设备: <CUDA Device 0>
```

[本步代码](#本步代码) · [真实 GPU、映射与中心差分证据](2026-10-07_transformer_qkv_unsplit_backward_validation_v037.json)。JSON 保存逐字代码、来源、设备、全部实际与差分梯度、误差和输出，片段 SHA256 `0ca6fde35eca336410567bcd702729f4278a03d616003557ba3d2058701af8fa`。相对链接已核验，本步无新增图。

## 与前版变化、限制与下一步

v036 得到头布局的 Q/K 梯度，V 梯度来自 v034；本版新增 d_Q、d_K、d_V，完成三份拆头操作反向。新增参数 0、更新次数 0。保留历史记录，生成本版工作区索引并追加项目 modeling_records/index.md；同步仅复制记录并更新索引，不重跑模型。

尚未反向经过 QKV 投影或输入投影，未训练或测量识别准确率。下一步从 d_Q/d_K/d_V 反向三个线性投影，计算对应权重、偏置和返回 encoder_input 的路径梯度；汇合各路径与直连贡献，完成输入投影反向后统一更新参数并组织 GPU 训练。
