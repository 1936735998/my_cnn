# MNIST 手写 Transformer 建模记录

日期：2026-10-06（Asia/Shanghai）；版本：v004。
阶段：Q/K/V 投影参数初始化与单图 GPU 前向验证。未拆分多头，未计算注意力分数、损失、梯度或训练。

## 范围与用户要求

用户已贴出 v003 输出，位置编码与 encoder_input 均为 `(28,32)`，首两行位置编码符合公式，encoder_input 显示 GPU 0。继续逐段手写并直接在聊天中给代码，梯度与更新后续自行实现，最终使用 GPU 训练。本轮不修改用户主 `F:\PythonProjects\deep_learning\transformer_mnist.py`，不另行交付 Python 代码文件；记录内嵌本步代码。

## 输入来源与处理（事实）

沿用本地 MNIST 缓存 `C:\Users\19367\.keras\datasets\mnist.npz`，SHA256 `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`，与 v001 一致。前 5000 张训练图、前 1000 张测试图；像素转 CuPy float32 并除以 255，标签保留 int64；本轮没有新下载、清洗、划分或数据增强。

验证输入是首训练图经过共享线性投影与固定正弦位置编码后的 encoder_input `(28,32)`。助手在独立进程读取执行用户已写的前三步，然后追加本步代码，未读取或修改用户 VS Code 实时内存，不能单凭输出形状宣称助手与用户参数逐元素相同。

## 变量、假设与公式

同一份输入 E=encoder_input 分别使用三组投影：

```text
Q = E W_Q + b_Q
K = E W_K + b_K
V = E W_V + b_V
```

三组 W 均为 `(32,32)`，偏置均为 `(32,)`，偏置广播到 28 行；Q/K/V 均为 `(28,32)`。每组投影在全部行之间共享自身参数，但 Q/K/V 三组参数独立。Query 用于查询匹配，Key 供 Query 匹配，Value 是后续按注意力权重汇总的内容表示。角色与多头投影参照 [Transformer 原论文第 3.2 节](https://arxiv.org/html/1706.03762v7)；偏置是本教学实现的线性层设置。

权重使用 CuPy 标准正态乘 0.1，偏置零；0.1 是教学初始化尺度，不称为最佳方案。本片段不重复设置随机种子，沿用已有 seed=42 后的随机流，连续生成三组权重。随机投影尚未通过识别任务学习，Q/K/V 名称表示它们在后续计算中的用途，不代表已经具备成熟语义。

本步新增训练参数 `3×(32×32+32)=3168`，加上已有输入投影 928 个，目前已初始化训练参数总数 4096。固定位置编码仍无训练参数。

## 本步代码

直接追加在第 3 步之后：

```python
# ==================================================
# Q、K、V 的投影参数
# ==================================================

W_Q = cp.random.randn(d_model, d_model).astype(cp.float32) * 0.1
W_K = cp.random.randn(d_model, d_model).astype(cp.float32) * 0.1
W_V = cp.random.randn(d_model, d_model).astype(cp.float32) * 0.1

b_Q = cp.zeros(d_model, dtype=cp.float32)
b_K = cp.zeros(d_model, dtype=cp.float32)
b_V = cp.zeros(d_model, dtype=cp.float32)

print("W_Q:", W_Q.shape)
print("W_K:", W_K.shape)
print("W_V:", W_V.shape)


# ==================================================
# 同一份输入，生成三种特征表示
# ==================================================

Q = cp.dot(encoder_input, W_Q) + b_Q
K = cp.dot(encoder_input, W_K) + b_K
V = cp.dot(encoder_input, W_V) + b_V

print("Q:", Q.shape)
print("K:", K.shape)
print("V:", V.shape)

print("Q 的实际设备:", Q.device)
print("K 的实际设备:", K.device)
print("V 的实际设备:", V.device)
```

## 方法选择理由（判断）

先生成完整 32 维 Q/K/V，再分别拆成计划的 4 个头、每头 8 维，便于先理解三个线性映射和角色，再理解维度重排。此步骤保持基础 CuPy 数组运算，不使用现成注意力层或自动求导。下一步多头拆分 `(4,28,8)` 尚未执行；每头注意力的缩放维度将是 8，不能把完整宽度 32 与每头宽度混用。

## 实际执行与验证（事实）

通过 `F:\PythonProjects\deep_learning\.venv\Scripts\python.exe` 在 PowerShell 独立执行本步；未声称助手在用户 VS Code 终端点击运行。Python 3.13.15、NumPy 2.5.3、CuPy 14.2.0；CuPy 可见 GPU 1 张，指定 GPU 0，型号 NVIDIA GeForce RTX 5060。本轮未安装或调整依赖。

三组 W `(32,32)`、b `(32,)` 和投影结果 `(28,32)` 均为 GPU 0 上的有限 CuPy float32 数组，所有偏置为零，三组权重逐对不同。Q/K/V 的实际设备均为 `<CUDA Device 0>`。

使用同一份输入及权重复制到 CPU，以 NumPy 计算三组投影参考，均在 `rtol=1e-5, atol=1e-6` 下匹配；最大绝对误差为 Q `2.086162567e-07`、K `2.384185791e-07`、V `1.788139343e-07`。GPU 完成同步；本步未改变原 encoder_input、W_in 或 b_in。

用户主文件执行前后 SHA256 一致：`2beb3767e6fe89aff84b52cc1a149c41ec84c222ae3e44a0f617dc0b6ac97681`。聊天片段 SHA256：`a299f9196e1b393b5c3e646c53cef74f141f68a1456e36cb5d7fdd3e65e69f40`。[本步代码](#本步代码) · [真实 GPU 验证结果](2026-10-06_transformer_qkv_validation_v004.json)。验证 JSON 保存逐字片段、源信息、配置、形状、设备、数值误差和控制台输出。

## 相比上一版的变化

v003 完成位置编码；v004 新增三组独立 Q/K/V 线性投影，新增 3168 个待训练参数。保留此前所有版本与证据，项目 modeling_records/index.md 与交付索引新增 v004。本轮是局部前向验证，不是训练或模型重训。

## 限制与下一步

尚未计算行之间的匹配分数、softmax 注意力权重、V 的加权汇总、多头合并、残差、LayerNorm、前馈、分类头、损失或反向。优化更新次数为 0，没有训练损失、测试准确率或完整 Transformer 执行结果。单图 GPU 投影正确不能证明模型学会识别数字或使用 GPU 能提速。

下一步把 Q/K/V 各拆成 4 个注意力头，每头 8 维，然后逐步计算缩放点积注意力。后续手写梯度将独立进行有限差分核验。
