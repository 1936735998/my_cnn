# MNIST 手写 Transformer 建模记录

日期：2026-10-06（Asia/Shanghai）；版本：v003。
阶段：固定正弦位置编码与输入特征相加的 GPU 前向验证。未实现注意力、损失、反向传播或训练。

## 问题范围与用户要求

用户已贴出第 2 步输出：W_in `(28,32)`、b_in `(32,)`、X `(28,28)`、embedding `(28,32)`，权重与 embedding 的设备均为 `<CUDA Device 0>`。这些输出符合教学设置，但没有给出用户进程内的具体权重值，不能据此声称助手与用户的参数逐元素相同。

本轮继续由用户逐段手写，代码直接贴在聊天里；助手不另行交付 Python 代码文件，不修改用户主 `F:\PythonProjects\deep_learning\transformer_mnist.py`。仅生成本步建模记录、验证证据与索引更新。用户要求手写梯度和更新，以及最终使用 GPU 的范围保持不变。

## 输入来源与处理（事实）

复用 v001/v002 的本地 MNIST 缓存 `C:\Users\19367\.keras\datasets\mnist.npz`，本轮 SHA256 `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`，与 v001 一致。沿用前 5000 张训练图和前 1000 张测试图，像素为 CuPy float32 且除以 255，标签为 CuPy int64。没有新增下载、清洗、划分、洗牌或数据增强。

首训练图按 28 行表示，每行 28 个像素；原输入投影共享 W_in 与 b_in，把每行映射成 32 维。为独立验证本步，助手在新 Python 进程读取并执行用户已写的数据准备与输入投影，再追加本记录内代码。没有读取或修改用户 VS Code 实时内存。当前 W_in 初始化仍沿用 seed=42、标准正态乘 0.1、偏置零。

## 假设、变量与公式

行位置从上到下编号 p=0,…,27。采用固定正弦位置编码，T=28、d_model=32，按 i=0,…,15 配对特征列：

```text
PE[p, 2i]   = sin(p / 10000^(2i / d_model))
PE[p, 2i+1] = cos(p / 10000^(2i / d_model))
encoder_input = embedding + PE
```

公式与输入加法参考 [Attention Is All You Need 第 3.5 节](https://arxiv.org/html/1706.03762v7)。10000 是该编码的常用固定基数，未从 MNIST 数据估计。这里的位置含义是图片的行序，不是时间。

position `(28,1)`、dimension_index `(16,)` 经广播得到 angles `(28,16)`；正弦填偶数列、余弦填奇数列，PE 和 encoder_input 均为 `(28,32)`。本片段采用等长奇偶列配对，适用当前偶数宽度 32。

PE 是由行号和公式直接确定的常量，本轮新增待训练参数 0；此前输入投影仍有 928 个待训练参数。固定编码为后续注意力提供位置线索，但不能证明模型已学会行顺序、位置关系或分类任务。

## 本步代码

直接追加在第 2 步之后，保留原 embedding 并创建 encoder_input：

```python
# ==================================================
# 正弦位置编码
# ==================================================

seq_len = embedding.shape[0]

# 28 个行位置，编号为 0～27
position = cp.arange(seq_len, dtype=cp.float32).reshape(seq_len, 1)

# 选择第 0、2、4、...、30 个特征位置
dimension_index = cp.arange(0, d_model, 2, dtype=cp.float32)

angles = position / (10000 ** (dimension_index / d_model))

position_encoding = cp.zeros((seq_len, d_model), dtype=cp.float32)

position_encoding[:, 0::2] = cp.sin(angles)
position_encoding[:, 1::2] = cp.cos(angles)


# ==================================================
# 输入特征 + 位置信息
# ==================================================

encoder_input = embedding + position_encoding

print("position_encoding:", position_encoding.shape)
print("encoder_input:", encoder_input.shape)
print("第 0 行位置编码的前 6 个值:", position_encoding[0, :6])
print("第 1 行位置编码的前 6 个值:", position_encoding[1, :6])
print("encoder_input 的实际设备:", encoder_input.device)
```

## 方法选择理由（判断）

固定编码不引入额外训练参数，便于逐段学习正弦/余弦、切片和矩阵广播。按行扫描的图像表示需要把空间行序作为输入信息，固定编码与现有 32 维表示可以直接相加。学习式位置编码也是一种可选方案，但本轮不引入或比较其效果，不声称固定方案最佳。

本步与下一步 Q/K/V 投影分开教学，让行号编码、内容特征和注意力计算的作用各自清楚；没有一次展开完整网络。

## 实际执行、验证与结果（事实）

通过 `F:\PythonProjects\deep_learning\.venv\Scripts\python.exe` 在 PowerShell 执行独立验证；没有声称助手在用户 VS Code 终端点击运行。Python 3.13.15、NumPy 2.5.3、CuPy 14.2.0，可见 GPU 1 张，指定 GPU 0，型号 NVIDIA GeForce RTX 5060。本轮没有安装或调整依赖。

position、dimension_index、angles、position_encoding、encoder_input 均为 GPU 0 上的有限 CuPy float32 数组。实际生成 PE `(28,32)`，encoder_input `(28,32)`，设备 `<CUDA Device 0>`。

独立使用 Python math 和 CPU float64，逐个位置与特征对计算原公式。GPU float32 结果与参考在 `rtol=1e-05, atol=2e-06` 下匹配，最大绝对差 `5.532527247e-07`。正弦/余弦对平方和与 1 的最大绝对差 `1.192092896e-07`。当前 28 个位置编码行逐行不同。

第 0 行前六项 `[0.0, 1.0, 0.0, 1.0, 0.0, 1.0]`，偶数列精确为 0、奇数列精确为 1；第 1 行前六项约 `[0.84147102, 0.54030228, 0.53316844, 0.84600914, 0.3109836, 0.95041525]`。输入相加与 CPU 相同数据参考一致，原 embedding、W_in、b_in 均未被本步改变；CUDA 同步完成。

用户主文件执行前后 SHA256 一致：`e91ed226c7a1653c941d5cd7e7d9c7bc5245400010f434e55b353ad9155bdaee`。聊天片段 SHA256：`cceee9eaf4de739dc501031da4fb1ba18c06b41dcc3a4ad64106066ea438fce4`。验证 JSON 包含逐字代码片段、输入源信息、形状、设备、误差、首两行编码及完整输出。

[本步代码](#本步代码) · [真实 GPU 验证结果](2026-10-06_transformer_position_encoding_validation_v003.json)。

## 相比上一版的变化

v002 完成输入投影；v003 新增固定位置编码与 encoder_input，输入宽度仍为 32，新增训练参数为 0。保留 v001/v002 记录与证据，不覆盖旧文件。项目 modeling_records/index.md 与交付索引新增 v003。

## 未解决限制与下一步

尚未实现 Q/K/V、注意力权重、残差、LayerNorm、前馈网络、分类头、损失和梯度。优化更新次数为 0，没有 Transformer 训练损失、预测准确率或 GPU 训练提速证据。本轮只有输入预处理与局部前向验证。

当前编码以偶数 d_model=32 编写，奇数宽度需调整余弦列的赋值范围；未来更改维度时须相应检查。固定行序和 5000/1000 子集属于教学设置，不等同于图像建模最优方案或完整 MNIST 基准。GPU float32 与 CPU float64 的微小数值差不代表公式错误。

下一步以 encoder_input 为输入初始化并计算 Q、K、V，之后再逐步构造缩放点积注意力；完整反向与参数更新留到相应教学阶段，并单独数值验证。
