# MNIST 手写 Transformer 建模记录

日期：2026-10-07（Asia/Shanghai）；版本：v042。
阶段：首次单样本 SGD 更新与更新后完整 GPU 前向核验。本轮真实执行；没有数据集训练循环或测试集评估。

## 范围、输入来源与处理（事实）

用户报告输入映射反向的 dW_in `(28,32)`、db_in `(32,)`、d_X `(28,28)` 均在 GPU 0。完整单图反向已完成。本版将20组参数与各自梯度配对，执行一次 SGD，并在独立核验进程重算同一图片的损失。模型代码仍贴在聊天，不改用户主文件、不交付完整 Python 模型文件。

独立进程执行用户已保存的 `F:\PythonProjects\deep_learning\transformer_mnist.py` 前四十一步，随后执行本片段。主文件 SHA256 `854fe5c714ceba2b80c369a7791e96ad8ed2e08f160d8cbb70029de040d6e94c`，执行前后未改变；未操作用户终端的实时内存。此进程的更新参数没有写回主文件或保存训练检查点。

沿用本地 MNIST `C:\Users\19367\.keras\datasets\mnist.npz`，SHA256 `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`，训练前5000、测试前1000，像素 CuPy float32 除以255，标签 int64。本轮只使用首张训练图与标签 5，无新增清洗、下载、数据划分、洗牌或增强。

## 假设、变量与公式

保持单层32维、4头、前馈64维的 post-LN Transformer、固定正弦位置编码、均值池化与十类单图交叉熵。完整前向和反向使用更新前参数，之后统一应用同次反向得到的全部梯度。

```text
theta_new = theta_old - learning_rate * gradient
learning_rate = float32(0.01)
20 个参数数组，总计9802个可训练标量
```

参数及梯度按同名键配对；字典引用原数组，CuPy增广赋值原地修改参数。固定位置编码与像素输入不属于 params。所有梯度来自同一轮前向/反向，更新后旧loss、概率及缓存不会自动重算。

## 方法选择理由（判断）

先用基础 SGD 展示手写参数更新，不引入优化器封装、动量、Adam、自动求导或现成神经网络层。学习率0.01是本次示范的候选值；本轮重新前向实测支持其对当前样本使损失下降，不据此认定它是完整训练的最佳学习率。下一次常规SGD更新应先重新前向及反向，不重复使用这次旧梯度。

## 本步代码

```python
# ==================================================
# 参数更新第一步：手写 SGD
# ==================================================

learning_rate = cp.float32(0.01)

# 保存原参数数组的引用，不复制参数
params = {
    "W_in": W_in, "b_in": b_in,
    "W_Q": W_Q, "b_Q": b_Q,
    "W_K": W_K, "b_K": b_K,
    "W_V": W_V, "b_V": b_V,
    "W_O": W_O, "b_O": b_O,
    "gamma_attn": gamma_attn, "beta_attn": beta_attn,
    "W_ff1": W_ff1, "b_ff1": b_ff1,
    "W_ff2": W_ff2, "b_ff2": b_ff2,
    "gamma_ffn": gamma_ffn, "beta_ffn": beta_ffn,
    "W_cls": W_cls, "b_cls": b_cls,
}

# 每个参数对应本次反向传播得到的梯度
grads = {
    "W_in": dW_in, "b_in": db_in,
    "W_Q": dW_Q, "b_Q": db_Q,
    "W_K": dW_K, "b_K": db_K,
    "W_V": dW_V, "b_V": db_V,
    "W_O": dW_O, "b_O": db_O,
    "gamma_attn": dgamma_attn, "beta_attn": dbeta_attn,
    "W_ff1": dW_ff1, "b_ff1": db_ff1,
    "W_ff2": dW_ff2, "b_ff2": db_ff2,
    "gamma_ffn": dgamma_ffn, "beta_ffn": dbeta_ffn,
    "W_cls": dW_cls, "b_cls": db_cls,
}

# 留下更新前的损失；参数变化不会自动改变旧 loss
loss_before_update = loss.copy()

# 原地更新：参数 = 参数 - 学习率 × 梯度
for name in params:
    params[name] -= learning_rate * grads[name]

print("学习率:", f"{float(learning_rate):.2f}")
print("已更新的参数数组数量:", len(params))
print("参数标量总数:", sum(parameter.size for parameter in params.values()))
print("更新前的损失:", float(loss_before_update))
print("W_in 的实际设备:", W_in.device)
print("W_cls 的实际设备:", W_cls.device)
```

## 实际执行、验证与结果（事实）

解释器 `F:\PythonProjects\deep_learning\.venv\Scripts\python.exe`，Python 3.13.15、NumPy 2.5.3、CuPy 14.2.0。可见GPU 1 张，GPU0为 NVIDIA GeForce RTX 5060。真实执行并同步CUDA，未调整依赖。

20组参数原地更新，原变量与字典引用、数组对象及存储地址保持一致；全部参数仍为float32、GPU0且有限。旧梯度、固定位置编码、X及抽查的旧前向缓存loss/probabilities/embedding保持不变，保存的loss_before_update为独立数组。

核验进程中的基础CuPy前向函数重新计算输入投影、位置相加、QKV、拆头、缩放softmax注意力、输出投影、残差及两次LayerNorm、前馈、均值池化、分类与稳定交叉熵，不重新初始化参数，也不覆写用户当前缓存。更新前重新前向与原loss/概率一致；更新后loss、概率和logits均为GPU0 float32有限值。

同一训练图的GPU损失从 `1.97143173218` 变为 `1.72921168804`，下降 `0.242220044136`。真实类别概率从 `0.139257326722` 变为 `0.177424222231`；预测从 6 变为 5。更新后概率和为 `1.00000011921`。

用v039独立NumPy float64完整前向，对实际更新前/后GPU参数快照核验。CPU更新前损失 `1.97143177161`、更新后 `1.72921171482`，更新后CPU/GPU绝对差 `2.677673061e-08`，均在rtol=2e-6、atol=2e-6下通过。JSON保存每组参数的实际更新前/后值、梯度、更新范数及与双精度公式的舍入差异，便于复核；本轮不重复已通过的反向差分。

实际聊天片段输出（只打印更新前缓存loss）：

```text
学习率: 0.01
已更新的参数数组数量: 20
参数标量总数: 9802
更新前的损失: 1.9714317321777344
W_in 的实际设备: <CUDA Device 0>
W_cls 的实际设备: <CUDA Device 0>
```

[本步代码](#本步代码) · [真实更新、完整GPU核验函数和CPU参考证据](2026-10-07_transformer_sgd_update_validation_v042.json)。相对链接已核验，本轮无新增图。

## 与前版变化、限制及下一步

v041完成完整单图反向并核验；本版首次执行统一单样本参数更新，新增参数0、更新次数1。本次更新只存在于独立核验进程内存，未保存训练检查点。保留历史记录，更新项目建模索引；同步仅复制记录证据，不重跑模型。

本轮当前样本损失下降支持更新方向有效，不能推断训练集或测试集准确率。尚未组织数据集训练循环、批量训练、训练收敛评估或泛化检验。下一步在聊天代码中用更新参数重新前向得到新loss/概率，然后逐步封装可复用前向/反向与GPU训练循环。新一轮反向需使用新前向缓存。
