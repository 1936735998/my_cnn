# MNIST 手写 Transformer 建模记录

日期：2026-10-07（Asia/Shanghai）；版本：v045。
阶段：单图完整 GPU 训练步骤封装与一次更新核验。本轮真实执行，不是仅文档更新；未运行数据集训练循环或测试集评估。

## 问题范围、输入来源与处理（事实）

用户报告v044完整手写反向得到20组梯度、9802参数标量，新输入权重、注意力投影及LayerNorm参数梯度和像素梯度都在GPU0。本版新增train_one_image，将新的前向、新缓存对应的完整反向与一次SGD连接；用训练索引1的第二张图片演示，训练函数返回更新前的单样本loss和概率，再独立前向观察更新后结果。

模型代码仍贴在聊天，不修改用户主文件，不交付模型Python主文件。独立进程读取执行 `F:\PythonProjects\deep_learning\transformer_mnist.py` 已保存前四十四步，包括先前首图一次SGD，再执行本片段；主文件SHA256 `ea2466dd073465a7e518feb8dde9a7f42518cd6f88b69e95cb921bd56eab0edb`，执行前后不变；未操作用户终端实时内存。核验脚本与fragment仅作为工作区核验材料，精确聊天代码保留如下与JSON中。

沿用本地MNIST `C:\Users\19367\.keras\datasets\mnist.npz`，SHA256 `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`，训练前5000、测试前1000张，像素CuPy float32除以255，标签int64。本轮新增实际训练只用索引1、真实标签0这一张，输入 `(28,28)`。无新下载、清洗、划分、增强、洗牌或测试集评估。

## 假设、变量与公式

沿用28行位置、每行28像素、模型宽度32、4头每头8维、前馈ReLU宽度64、单层post-LN编码器、固定正弦/余弦位置编码、均值池化、十类交叉熵，参数20数组共9802标量。缓存必须来自本次当前参数的前向，完整反向结束前不改变参数。

```text
(L_before, P_before, cache) = forward(X, y, theta_current)
G = backward(theta_current, cache)
对全部20参数：theta_new = theta_current - float32(0.01) * G
return L_before, P_before
(L_after, P_after, cache_after) = forward(X, y, theta_new)
```

参数字典引用原数组，SGD增广赋值原地更新，像素与固定位置编码不更新。返回loss是此次更新前的在线单样本损失，概率也是更新前预测；参数更新后已有loss、概率和缓存不会自动重算。训练函数每调用一次都重新前向及反向，可用于后续多图训练。

## 方法选择理由（判断）

将之前已推导和核验的两个函数连接，保留fresh forward→fresh backward→统一SGD的顺序，避免重复使用旧grads或旧cache。用第二张图片说明输入和标签由调用参数决定；学习率0.01仍是教学示范候选值，本轮有限样本结果不能认定其为完整训练最佳值。

v044已经对完整手写反向全部9802参数和784像素做过独立完整模型中心差分，本轮主要风险是把更新前后结果混淆、缓存配错或更新次数错误，因此检查全部20参数与fresh梯度的一次float32 SGD精确一致，并用独立CPU双精度完整前向验证更新前后loss；本轮没有重复10586分量差分。

## 本步代码

```python
# ==================================================
# 训练准备：单张图片的完整训练步骤
# ==================================================

def train_one_image(
    X, target, params, position_encoding, num_heads, ln_eps, learning_rate
):
    # 1. 用当前参数做一次新的前向传播
    loss_before, probabilities_before, cache = forward_one_image(
        X, target, params, position_encoding, num_heads, ln_eps
    )

    # 2. 根据这次前向的缓存，计算新的梯度
    grads, _ = backward_one_image(params, cache)

    # 3. 所有梯度计算完毕后，统一更新参数
    for name in params:
        params[name] -= learning_rate * grads[name]

    # 返回本步更新前的损失与概率
    return loss_before, probabilities_before


# 索引 1 对应第二张训练图片
sample_index = 1
X_step = x_train[sample_index]
target_step = int(y_train[sample_index])

loss_before_step, probabilities_before_step = train_one_image(
    X_step, target_step, params, position_encoding,
    num_heads, ln_eps, learning_rate
)

# 再做一次前向，观察这一步更新后的结果
loss_after_step, probabilities_after_step, cache_step = forward_one_image(
    X_step, target_step, params, position_encoding, num_heads, ln_eps
)

print("本次训练图片索引:", sample_index)
print("真实标签:", target_step)
print("本步更新前的损失:", float(loss_before_step))
print("本步更新后的损失:", float(loss_after_step))
print("损失减少量:", float(loss_before_step - loss_after_step))
print("更新前的预测:", int(cp.argmax(probabilities_before_step)))
print("更新后的预测:", int(cp.argmax(probabilities_after_step)))
print("训练损失的实际设备:", loss_before_step.device)
print("W_in 的实际设备:", params["W_in"].device)
```

## 实际执行、验证与结果（事实）

解释器 `F:\PythonProjects\deep_learning\.venv\Scripts\python.exe`，Python 3.13.15、NumPy 2.5.3、CuPy 14.2.0；可见GPU 1张，GPU0为 NVIDIA GeForce RTX 5060。实际GPU执行并同步CUDA，未调整依赖。

核验先用当前参数对同一索引1图片重新前向及反向获得fresh梯度，不更新参数；执行聊天片段后，全部20参数精确等于一次fresh梯度的float32 SGD结果，保持原数组对象、存储地址、原变量引用、float32、GPU0与有限性。新参数0，参数仍9802标量。旧grads、grads_new、cache_after_update全部内容、元数据及数组引用保持不变；训练/测试数据、X、固定位置编码、旧loss/概率/像素梯度均未改变。

返回的loss与概率逐元素匹配更新前fresh前向，loss形状 `()`、概率 `(10,)`，均float32 GPU0有限；更新后的loss、概率及cache_step与当前更新后参数再次前向精确一致。概率和更新前 `1`、更新后 `1`。

本次真实类别0，GPU损失从 `2.9217672348` 到 `2.61194705963`，变化为下降；定义更新前减更新后的差 `0.309820175171`。预测从 5 到 5，真实类别概率从 `0.0538384467363` 到 `0.0733915194869`。未硬编码标签、预测或损失一定下降。

v039独立NumPy float64完整参考，从当前图片及实际更新前/后参数快照重算全部模型。CPU更新前损失 `2.92176743741`、更新后 `2.61194692892`，分别与GPU绝对差 `2.026028922e-07`、`1.30714751e-07`，rtol=2e-6、atol=2e-6下通过。JSON保存精确聊天片段及哈希、主文件哈希、数据哈希、独立CPU完整源码及哈希、实际inspect提取的前向/反向源代码，以及每组参数前/后值、fresh梯度、更新范数、float32匹配与双精度公式舍入误差。

引用之前v044真实证据：全部9802参数分量检查通过，最大绝对误差 `1.142849937e-07`；证据路径与SHA256保留在本版JSON中。这是先前状态的梯度核验；本版未宣称对第二张图片重复全分量差分。

实际片段输出：

```text
本次训练图片索引: 1
真实标签: 0
本步更新前的损失: 2.921767234802246
本步更新后的损失: 2.6119470596313477
损失减少量: 0.30982017517089844
更新前的预测: 5
更新后的预测: 5
训练损失的实际设备: <CUDA Device 0>
W_in 的实际设备: <CUDA Device 0>
```

[本步代码](#本步代码) · [真实更新、独立完整参考与可复核参数证据](2026-10-07_transformer_train_step_validation_v045.json)。相对链接已核验，本版无新增图。

## 与前版变化、限制及下一步

v044整合完整反向并计算首图新梯度；本版新增完整单图训练函数，使用第二张训练图fresh前向与反向后更新一次。本独立进程重放之前首图一次SGD，加上本段新执行一次，共两次单样本更新；额外核验前向与反向不更新参数。没有数据集训练循环、epoch训练、测试评估或训练检查点，更新仅存在本核验进程内存。保留模型与历史记录，追加项目记录索引；同步仅复制已生成记录与证据，不重跑模型。

当前单图变化不能推断训练集或测试集准确率、收敛或泛化。函数返回更新前在线单样本loss，后续逐步平均此loss属于参数持续变化时的训练过程指标，不能解释为epoch结束后固定参数的训练集损失。下一步逐步组织多图GPU训练循环，明确在线训练指标与固定参数评估指标，再记录实际数据集损失、准确率与局限。
