# MNIST 手写 Transformer 建模记录

日期：2026-10-07（Asia/Shanghai）；版本：v022。
阶段：分类头手写反向，GPU 梯度与 CPU 中心差分核验。本轮真实执行验证，不是仅文档更新；未完成全网络反向、参数更新或训练。

## 范围、来源与处理（事实）

用户输出 d_logits `(10,)`、真实类第 5 项约 -0.8607427、梯度和 -2.9802322387695312e-08、GPU 0，与前版一致。本轮只反向通过线性分类头，得到权重、偏置和图片特征梯度。代码贴在聊天，不交付独立 Python 文件，不修改主文件。目标仍为逐层手写 Transformer 反向和更新，在 GPU 训练。

独立进程读取执行保存的 `F:\PythonProjects\deep_learning\transformer_mnist.py` 前二十一步，再执行本片段；不声称操作终端或实时内存。主文件已保存前二十一步，无需补充旧片段。

沿用本地 MNIST `C:\Users\19367\.keras\datasets\mnist.npz`，SHA256 `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`。训练前 5000、测试前 1000 图，像素 CuPy float32 除以 255、标签 int64。本轮未下载、新增清洗、划分、洗牌或增强，只检查首张训练图当前局部分类头，不评估测试集。输入为池化图片表示、分类参数和交叉熵对 logits 的已核验梯度。

## 变量、公式与假设

f=image_features `(32,)`，W=W_cls `(32,10)`，b=b_cls `(10,)`，g=d_logits `(10,)`，前向 z=f@W+b。

```text
dW_cls[k,c] = image_features[k] * d_logits[c]: (32,10)
db_cls[c] = d_logits[c]: (10,)
d_image_features[k] = sum_c W_cls[k,c] * d_logits[c]: (32,)
```

权重梯度是特征向量与上游梯度的外积，不发生特征求和；输入梯度在类别轴求和。使用本次前向的 W_cls 计算输入梯度。当前单图损失不额外平均，位置均值池化的梯度分配留到下一节点。新增参数 0，累计 9802。延续图片行空间序列、单层编码器、位置均值池化和十类无权重交叉熵设置。

## 方法选择理由（判断）

用链式法则将损失梯度传过线性分类器，分清参数梯度与向前一层传递的输入梯度。外积和 dot 明确表示不同的收缩轴。先完成所有节点反向后再统一更新参数，避免前向参数与反向参数不一致。采用基础 CuPy 操作，没有自动求导或现成层。

## 本步代码

```python
# ==================================================
# 反向第二步：分类头
# ==================================================

# 外积：(32,) 与 (10,) → (32, 10)
dW_cls = cp.outer(image_features, d_logits)

# 偏置直接接收对应类别的梯度
db_cls = d_logits.copy()

# (32, 10) @ (10,) → (32,)
# 使用本次前向计算时的 W_cls
d_image_features = cp.dot(W_cls, d_logits)

print("dW_cls:", dW_cls.shape)
print("db_cls:", db_cls.shape)
print("d_image_features:", d_image_features.shape)
print("dW_cls 的实际设备:", dW_cls.device)
print("db_cls 的实际设备:", db_cls.device)
print("d_image_features 的实际设备:", d_image_features.device)
```

## 实际执行、验证与结果（事实）

解释器 `F:\PythonProjects\deep_learning\.venv\Scripts\python.exe`；Python 3.13.15、NumPy 2.5.3、CuPy 14.2.0。可见 GPU 1 张，使用 GPU 0，型号 NVIDIA GeForce RTX 5060。独立 PowerShell 进程执行并同步 CUDA；CuPy 缓存在工作区 work/cupy_cache，Python 使用 -B。本轮未安装或调整依赖。

三组梯度 `(32,10)`、`(10,)`、`(32,)`，均为有限 float32、GPU 0。分类参数、前向输入和缓存、上游 d_logits、loss 均未改变，偏置梯度存储与 d_logits 独立。实际参数 size 核对仍 9802。

用当前实际 f/W/b 转为 CPU float64 重建完整局部分类器 z=f@W+b，再计算稳定交叉熵。分别在 320 个权重、10 个偏置和 32 个输入特征上做 ±1e-5 中心差分，共检查 362 个分量，rtol=1e-5、atol=2e-6 下全部通过。GPU float32 logits 与 CPU 双精度重算存在舍入差异，因此不要求精确相等。

最大绝对误差：dW_cls `7.418782388e-08`，db_cls `1.889486267e-08`，d_image_features `1.728169255e-08`。此验证只覆盖局部分类头及其输入梯度，不代表全网络反向已正确。

主文件执行前后 SHA256 一致：`6e87821f45ddc531197e77924e4bc9a4df38624b7ee13937e09c4d7c5bd49da4`；片段 SHA256：`a9676604b86e8a799c9477e37ed5a2f0d04f2a0be565cb428e0b81182b403a9d`。[本步代码](#本步代码) · [真实 GPU 与差分证据](2026-10-07_transformer_classifier_backward_validation_v022.json)。JSON 保存逐字代码、来源、设备、全部实际梯度和差分值、误差与输出，相对链接已核对。

## 版本变化、限制与下一步

v021 得到 d_logits；本版新增 dW_cls、db_cls、d_image_features，完成分类头反向。新增参数 0、更新次数 0。保留历史版本，生成本版工作区索引并按约定追加项目 modeling_records/index.md。

尚未池化、编码器或输入投影反向，没有训练、识别准确率或 GPU 提速结论。下一步把 d_image_features 沿平均池化反向传给 28 行 encoder_output，每行接收其 1/28，再继续 LayerNorm、前馈和注意力反向，逐步核验后组织 GPU 训练。
