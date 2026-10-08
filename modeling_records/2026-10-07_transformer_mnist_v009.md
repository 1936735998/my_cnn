# MNIST 手写 Transformer 建模记录

日期：2026-10-07（Asia/Shanghai）；版本：v009。
阶段：合并多头的局部前向 GPU 验证。本轮真实执行了验证，不是仅文档更新；未训练。

## 范围与证据来源

用户输出 head_output `(4,28,8)`、第 0 头 `(28,8)`、GPU 0，维度与前一步一致。本轮只把每个位置的四个头输出拼接成 32 维表示，暂不执行输出投影。代码直接贴在聊天，不交付单独 Python 代码文件，不修改用户主文件。

独立进程读取执行用户保存的 `F:\PythonProjects\deep_learning\transformer_mnist.py` 前八步后，再执行本步片段；不声称读取编辑器实时内存或操作用户终端。整体目标仍是手写完整前向、反向、参数更新并在 GPU 训练。

## 输入来源与处理（事实）

沿用本地 MNIST `C:\Users\19367\.keras\datasets\mnist.npz`，SHA256 `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`。训练取前 5000 图、测试取前 1000 图；像素转 CuPy float32 并除以 255，标签 int64。本轮没有下载、新增清洗、洗牌、划分或增强。只验证第一张训练图的局部前向，不计算测试准确率。

head_output 来源为输入投影、固定位置编码、随机 Q/K/V 投影、四头拆分、缩放点积分数、稳定 softmax 与加权 V。用户主文件已保存前八步，验证无需补充旧片段。

## 变量、假设与公式

H=4，T=28，d_head=8，d_model=32=H*d_head。

```text
O[h,i,m] = head_output: (4,28,8)，轴为(头,位置,特征)
B[i,h,m] = O[h,i,m]: (28,4,8)，轴为(位置,头,特征)
C[i,h*8+m] = O[h,i,m]: (28,32)
```

每个位置按头 0、1、2、3 的顺序依次保留各 8 个特征。先把位置放到第一个轴，再把头和特征展平，保证同一行拼接的都是同一查询位置。没有新增训练参数。继续使用图像行作为空间序列与无因果掩码的分类设置；输入维度、有限数值及设备本轮均已核验。

## 方法选择理由（判断）

采用 transpose(1,0,2) 后 reshape，明确体现从“头优先”到“位置优先”的轴变化。该步骤保留四个头全部特征，供下一步学习输出投影时混合信息；直接按头优先顺序 reshape 会把不同位置混入同一行。采用基础 CuPy 操作，不调用现成注意力层或自动求导。

## 本步代码

```python
# ==================================================
# 合并四个头的输出
# ==================================================

# (4, 28, 8) → (28, 4, 8)
# 把同一行的四个头放在一起
head_output_by_position = head_output.transpose(1, 0, 2)

# (28, 4, 8) → (28, 32)
# 将每行的 4 × 8 个特征拼接起来
concatenated_heads = head_output_by_position.reshape(seq_len, d_model)

print("head_output_by_position:", head_output_by_position.shape)
print("concatenated_heads:", concatenated_heads.shape)
print("一行拼接后的特征:", concatenated_heads[0].shape)
print("concatenated_heads 的实际设备:", concatenated_heads.device)
```

## 实际执行、验证与结果（事实）

解释器 `F:\PythonProjects\deep_learning\.venv\Scripts\python.exe`，Python 3.13.15、NumPy 2.5.3、CuPy 14.2.0；可见 GPU 1 张，使用 GPU 0，型号 NVIDIA GeForce RTX 5060。在独立 PowerShell 进程执行并完成 CUDA 同步；CuPy 缓存置于工作区 work/cupy_cache，Python 使用 -B。本轮未安装或调整依赖。

实际 head_output_by_position `(28,4,8)`，concatenated_heads `(28,32)`，单行 `(32,)`；数组均为有限 float32、GPU 0。CPU 将四个头的 `(28,8)` 数组沿特征轴显式 concatenate，结果与 GPU 拼接逐元素一致，最大绝对误差为 0。另逐头核对输出每 8 列与该头全部位置完全对应；逆变换恢复原 `(4,28,8)` 数组，逐元素一致，输入 head_output 未改变。重排验证通过不能证明识别性能。

主文件执行前后 SHA256 一致：`5a021fac3fdde21ae36fda13d1d924f74b2194c7400862609ff64129dbd41c41`；本步片段 SHA256：`93a2260082b0243e694f8365238fc8cbc0ffd8946b83a80aac1af93b3181f86b`。[本步代码](#本步代码) · [真实 GPU 验证证据](2026-10-07_transformer_head_merge_validation_v009.json)。JSON 保存逐字代码、来源、配置、设备与实际输出；相对文件链接已核对。

## 相比上一版的变化

v008 得到分头输出 `(4,28,8)`；v009 将同一位置的四头特征拼接为 `(28,32)`。未新增可训练参数，参数更新次数仍为 0。保留旧记录，生成工作区本版索引并按约定追加项目 modeling_records/index.md。

## 未解决限制与下一步

尚未执行 W_O 输出投影、残差、LayerNorm、前馈、分类头、损失或梯度；没有训练损失、测试准确率或 GPU 提速结论。当前参数仍随机初始化。转置与 reshape 的内存是否复制不是本轮性能结论，不据此声称零开销。

下一步初始化 W_O `(32,32)` 和 b_O `(32,)`，计算拼接结果的输出投影。后续逐步补全前向、手写反向和参数更新，并执行必要的梯度数值核验。
