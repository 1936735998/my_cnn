# MNIST 手写 Transformer 建模记录

日期：2026-10-08（Asia/Shanghai）；版本：v047。
阶段：训练后固定参数评估。真实运行128张训练子集与1000张测试子集的GPU前向，无新增反向传播或优化更新；本版不是仅文档更新。

## 问题范围、输入来源与处理（事实）

用户报告v046前128张图片训练1轮：新增128次SGD，在线平均loss约2.3527、在线accuracy14.06%，18/128张在各自更新前预测正确。此次按照教学步骤固定轮末参数，重新计算训练过的128图和已加载1000张测试图的平均loss与accuracy，不把在线指标当作固定模型评估。

用户主模型文件 `F:\PythonProjects\deep_learning\transformer_mnist.py` 的SHA256为 `8b3e36ed7eaa53223b40e36b290ee5e9bf6226eea617818d27322dc67a68bef9`，核验前后不变。代码仍贴在聊天，由用户自行加入主文件；核验没有编辑主文件或操作其终端实时内存。独立进程runpy重放主文件至v046，重放历史130次更新（2次单图演示加128次循环），其后20组参数的全部值精确匹配v046独立证据中的after快照。精确聊天片段随后新增1128次评估前向，新增SGD更新0次。

沿用本地MNIST `C:\Users\19367\.keras\datasets\mnist.npz`，SHA256 `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`，np.load原先allow_pickle=False。已加载原训练集前5000张和测试集前1000张；像素GPU0 float32归一化除以255、标签int64。本版用训练前128张以及全部已加载1000张测试图，不是原MNIST完整10000张测试集。数据划分、清洗、归一化、增强和标签均未改变。evaluate函数每次复制标签到CPU用于Python整数索引；图片、参数、前向、概率、损失累计与正确计数在GPU0，末端少量指标转成Python数值供显示。

## 假设、变量与公式

继续使用28行像素位置、每行28维输入、模型宽32、4个8维头、ReLU前馈宽64、单个post-LN编码器、固定正弦/余弦位置编码、均值池化与十类稳定softmax交叉熵。20组参数合计9802标量。在整个评估期间，theta固定为上一轮末参数，无dropout或其他随机前向层。

```text
对指定子集第i张图：L_i,P_i = forward(theta_fixed, X_i, y_i)
mean_loss = sum(L_i)/N
accuracy = sum(argmax(P_i) == y_i)/N
```

mean_loss与accuracy返回float32的GPU0标量，accuracy为0到1比例，打印乘100显示百分数；N分别128和1000。在线指标使用随训练变化的theta_i，而本版全子集使用同一theta_fixed，两者统计对象不同，不能直接把差异称作准确率提升。test子集仅用于评估，没有用于训练、反向、更新或选择超参数。

## 方法选择理由（判断）

采用逐图调用已有forward_one_image，在学习训练循环后引入固定模型评估，便于理解计数、平均与参数冻结的区别。反向与SGD完全不需要；现有函数的数值梯度已验证，本次不重复10586分量的有限差分。主要检查指标统计正确、标签对应、测试数据独立于训练、参数与历史缓存保持不变。

核验使用透明前向委托hook观察原GPU函数，按输入视图存储地址逐项识别数据源和索引，再确认训练0..127、测试0..999顺序和原标签对应。禁止反向与训练函数被调用，末端严格比对全部参数、全数据数组、固定位置编码、原全局CuPy/NumPy数组及旧grads/cache字典的数值、对象身份和GPU存储地址。独立NumPy float64完整模型按相同固定参数计算1128张图交叉熵参考；预测计数由日志概率另用NumPy argmax核算。

## 本步代码

```python
# ==================================================
# 固定参数评估：只做前向传播
# ==================================================

def evaluate_images(
    images, labels, params, position_encoding, num_heads, ln_eps
):
    num_images = images.shape[0]
    labels_cpu = cp.asnumpy(labels)

    loss_sum = cp.zeros((), dtype=cp.float32)
    correct_sum = cp.zeros((), dtype=cp.int64)

    for index in range(num_images):
        target_i = int(labels_cpu[index])
        loss_i, probabilities_i, _ = forward_one_image(
            images[index], target_i, params,
            position_encoding, num_heads, ln_eps
        )

        loss_sum += loss_i
        correct_sum += (cp.argmax(probabilities_i) == target_i)

    mean_loss = loss_sum / cp.float32(num_images)
    accuracy = correct_sum.astype(cp.float32) / cp.float32(num_images)
    return mean_loss, accuracy


# 评估刚才训练过的前 128 张图片
train_eval_loss, train_eval_accuracy = evaluate_images(
    x_train[:num_train], y_train[:num_train], params,
    position_encoding, num_heads, ln_eps
)

# 评估已经加载的 1000 张测试图片
test_eval_loss, test_eval_accuracy = evaluate_images(
    x_test, y_test, params, position_encoding, num_heads, ln_eps
)

print("训练子集样本数:", num_train)
print("训练子集平均损失:", float(train_eval_loss))
print("训练子集准确率:", f"{float(train_eval_accuracy) * 100.0:.2f}%")
print("测试集样本数:", x_test.shape[0])
print("测试集平均损失:", float(test_eval_loss))
print("测试集准确率:", f"{float(test_eval_accuracy) * 100.0:.2f}%")
print("评估损失的实际设备:", test_eval_loss.device)
print("评估准确率的数据类型:", test_eval_accuracy.dtype)
```

## 实际执行、验证与结果（事实）

解释器 `F:\PythonProjects\deep_learning\.venv\Scripts\python.exe`，Python 3.13.15、NumPy 2.5.3、CuPy 14.2.0；可见GPU 1张，GPU0 NVIDIA GeForce RTX 5060。CUDA同步后完成实际评估，无安装或更换依赖。前向1128次、反向0次、训练函数0次、新SGD0次。所有20参数仍float32、GPU0、有限，9802标量全部严格不变，参数对象和地址未变。所有旧全局数组、数据、PE、grads、grads_new、cache_after_update与cache_step未改变；观察与禁止调用hook已恢复。

| 固定模型评估范围 | 样本数 | 正确数 | GPU平均loss | GPUaccuracy |
|---|---:|---:|---:|---:|
| 已训练的前128张训练图 | 128 | 24 | 2.2608435154 | 18.75% |
| 原测试集前1000张 | 1000 | 162 | 2.31624889374 | 16.2000000477% |

真实输出：

```text
训练子集样本数: 128
训练子集平均损失: 2.260843515396118
训练子集准确率: 18.75%
测试集样本数: 1000
测试集平均损失: 2.316248893737793
测试集准确率: 16.20%
评估损失的实际设备: <CUDA Device 0>
评估准确率的数据类型: float32
```

1128张逐图GPUloss与独立CPU float64参考在rtol=2e-6、atol=2e-6下全部通过，最大逐图绝对差 `2.64268965378e-07`。GPU返回均值与同顺序float32日志累计吻合，accuracy严格等于按独立NumPy argmax统计的正确数除以N。CPU float64平均loss与GPU float32均值绝对差：训练 `7.76786492906e-08`，测试 `9.79927493816e-07`，均通过相同容差；允许float32累计舍入差异。

观察评估用时 `3.7789` 秒，含逐图同步、日志和hook开销；CPU参考用时 `0.483523` 秒。它们不是模型性能基准。JSON保存精确聊天代码，主模型/数据/参考的SHA256和参考完整源代码，inspect取得的三模型函数完整源代码，1128张逐项标签、loss、概率、预测、正确标记及CPU参考，所有参数前后实际快照、环境与历史130次/新增0次更新语义。

[本步代码](#本步代码) · [逐图真实评估、参数与数值核验证据](2026-10-08_transformer_evaluation_validation_v047.json)。相对链接已核验，本版无新增图。

## 与前版变化、未解限制及下一步

v046提供128图在线训练指标；本版新增固定轮末参数的训练子集和测试子集评估，明确二者与在线accuracy14.06%的统计区别。今日建立v047，保留昨日及更早版本，向项目建模记录索引追加链接。--sync-project只复制已经运行产生的记录和JSON，不导入模型、不重跑训练或评估。

模型此前只训练过前128张1轮和两次单图演示，没有训练完整5000图，没有评估完整10000图原测试集，也未保存checkpoint。低样本与少轮数阶段的指标不能证明模型收敛或稳定泛化。本版没有预设准确率门槛或宣称SGD训练效果足够；单图Python循环存在GPU利用率限制。下一步扩展到当前5000张训练图的多轮训练；若需要选择学习率或轮数，应从训练数据单独划分validation，仅用训练/验证结果选择，测试集保留作评估。独立核验进程的参数不代表用户终端实时状态，但重放后的快照与先前证据一致。
