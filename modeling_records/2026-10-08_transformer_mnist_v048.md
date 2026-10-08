# MNIST 手写 Transformer 建模记录

日期：2026-10-08（Asia/Shanghai）；版本：v048。
阶段：5000张训练图片三轮GPU训练及最终固定参数评估。本版真实执行训练和评估，不是仅文档更新。

证据字段名称修订（2026-10-08，仅文档）：将“既有数组不变”明确为“既有非参数数组不变”；参数值已按SGD更新。此项修订未重跑模型，未改变任何数值结果。

## 问题范围、输入来源与处理（事实）

用户已报告v047固定评估：训练前128张loss约2.26084、准确率18.75%；测试前1000张loss约2.31625、准确率16.20%。本版继续教学，沿用这些参数与单图手写前向、反向、SGD函数，在当前已加载5000张训练图上随机打乱三轮；每轮每图恰好一次，共新增15000次更新。最后固定轮末参数，只前向评估5000张训练图和1000张测试图。仅训练集参与优化；测试集只在三轮结束后评估一次，不用于选择轮数或超参数。

代码贴在聊天，用户自行加入主文件。本核验不编辑 `F:\PythonProjects\deep_learning\transformer_mnist.py`，独立进程runpy重放已经保存至v047的代码（先前130次参数更新及固定评估），全部20参数逐项精确匹配v047证据after快照，然后执行精确聊天片段。主文件SHA256 `4bde855fdaf87c216d410b265ae30367f486995480ca7c35fd447f66e8edc37d`，执行前后不变。本核验不修改用户正在运行的终端内存。

输入为本地MNIST `C:\Users\19367\.keras\datasets\mnist.npz`，文件SHA256 `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`。沿用前5000训练图、前1000测试图；像素GPU0 float32除以255，标签int64，无新增清洗、数据增强或重划分。标签一次复制到CPU，每轮CuPy生成5000个随机索引再复制到CPU供Python循环；图片、参数、前向、手写反向、原地更新与指标累积仍在GPU0。数据、标签、固定位置编码的值、对象与存储均未改变。

## 假设、变量与公式

模型沿用28行作为28个位置、每行28像素，d_model=32，4头且每头8维，ReLU前馈宽度64，单个post-LN编码器，固定正弦/余弦位置编码，均值池化，10类交叉熵。共20参数数组、9802标量；学习率float32(0.01)。没有自动求导、现成网络层或优化器。

```text
每轮order=permutation(0,...,4999)
第t步：L_t,P_t,cache_t=forward(theta_t,X_i,y_i)
G_t=backward(theta_t,cache_t)
theta_(t+1)=theta_t-learning_rate*G_t
online_mean_loss=(sum更新前L_t)/已训练数
online_accuracy=(更新前预测正确数)/已训练数
fixed_mean_loss=(sum_i L(theta_final,X_i,y_i))/N
fixed_accuracy=sum_i[argmax P(theta_final,X_i)==y_i]/N
```

在线指标使用不断变化的参数，固定评估使用同一组最终参数，两者语义不同。train_history_full每轮保留在线均值与0到1准确率比例。打印准确率时乘100。评估期间没有反向、参数更新；没有用旧全局梯度或缓存训练。

## 方法选择理由（判断）

沿用单图SGD将前面逐步手写的机制直接用于完整已加载训练范围。三轮是便于理解多轮训练的教学设置，学习率0.01为原有候选值；两者没有经过最优参数搜索，不保证达到特定准确率或收敛。每轮打乱顺序可避免一直按固定顺序训练；不改变每图每轮一次的定义。当前简单Python单图循环GPU利用率可能偏低，批量训练可另作改进，但本版先保持已验证的函数与数学结构。

## 本步代码

```python
# 延续当前参数，用已加载的全部 5,000 张训练图片训练 3 轮。
num_train_full = x_train.shape[0]
num_epochs_full = 3
train_labels_cpu_full = cp.asnumpy(y_train)
train_history_full = []

for epoch_full in range(num_epochs_full):
    # 每轮打乱顺序；每张训练图片在本轮恰好使用一次。
    order_full = cp.asnumpy(cp.random.permutation(num_train_full))
    loss_sum_full = cp.zeros((), dtype=cp.float32)
    correct_sum_full = cp.zeros((), dtype=cp.int64)

    for step_full, index_full in enumerate(order_full, start=1):
        index_full = int(index_full)
        target_full = int(train_labels_cpu_full[index_full])

        # 内部依次完成：前向传播、手写反向传播、一次 SGD 更新。
        # 返回的是本次更新之前的损失和概率。
        loss_i_full, probabilities_i_full = train_one_image(
            x_train[index_full], target_full, params,
            position_encoding, num_heads, ln_eps, learning_rate
        )

        loss_sum_full += loss_i_full
        correct_sum_full += (cp.argmax(probabilities_i_full) == target_full)

        if step_full % 500 == 0 or step_full == num_train_full:
            mean_loss_full = float(loss_sum_full / cp.float32(step_full))
            online_accuracy_full = float(
                correct_sum_full.astype(cp.float32) / cp.float32(step_full)
            ) * 100

            print(
                f"第 {epoch_full + 1}/{num_epochs_full} 轮 | "
                f"已训练 {step_full}/{num_train_full} | "
                f"在线平均损失: {mean_loss_full:.4f} | "
                f"在线准确率: {online_accuracy_full:.2f}%"
            )

    train_history_full.append({
        "epoch": epoch_full + 1,
        "online_mean_loss": float(loss_sum_full / cp.float32(num_train_full)),
        "online_accuracy": float(
            correct_sum_full.astype(cp.float32) / cp.float32(num_train_full)
        ),
    })

print("本段新增更新次数:", num_train_full * num_epochs_full)

# 训练结束后，使用同一组最终参数分别评估训练集与测试集。
full_train_eval_loss, full_train_eval_accuracy = evaluate_images(
    x_train, y_train, params, position_encoding, num_heads, ln_eps
)
full_test_eval_loss, full_test_eval_accuracy = evaluate_images(
    x_test, y_test, params, position_encoding, num_heads, ln_eps
)

print("完整训练集样本数:", x_train.shape[0])
print("完整训练集平均损失:", float(full_train_eval_loss))
print(f"完整训练集准确率: {float(full_train_eval_accuracy) * 100:.2f}%")
print("测试集样本数:", x_test.shape[0])
print("测试集平均损失:", float(full_test_eval_loss))
print(f"测试集准确率: {float(full_test_eval_accuracy) * 100:.2f}%")
print("W_in 的实际设备:", params["W_in"].device)
print("W_cls 的实际设备:", params["W_cls"].device)
print("W_in 的数据类型:", params["W_in"].dtype)
```

## 实际执行、验证与结果（事实）

解释器 `F:\PythonProjects\deep_learning\.venv\Scripts\python.exe`，Python 3.13.15、NumPy 2.5.3、CuPy 14.2.0；可见GPU 1张，GPU0 NVIDIA GeForce RTX 5060。训练15000次fresh前向、15000次同缓存完整反向、15000次调用SGD步骤；重放130次加新增15000次共15130次更新。三轮顺序各是0..4999的完整排列，存储地址对应真实训练图，标签全部核对原y_train，没有参数初始化或重新seed。

透明hook委托原函数实际执行，只观察，不替代模型计算或额外训练。每一步核对同次fresh前向缓存被完整反向使用以及返回更新前loss/probabilities；读取11个float32值检查所有返回loss及10类概率有限、归一化，并记录sample、target、loss、预测、真实类别概率。活动cache完成后释放，不保留15000份历史cache或跨生命周期比较ID。

为了控制开销，完整20参数梯度的GPU0/float32/形状/有限性和精确一次SGD对照仅在每轮第1、1000、2000、3000、4000、5000步检查，共18次（360个参数数组转移），不是逐项验证全部15000次参数转移。原训练函数源码保留在JSON；每一次训练调用均实际委托此函数。每500步的30条真实打印行独立核对GPU累计loss、correct_count与日志同顺序float32累计；三轮history逐项核对。

| 轮数 | 在线平均损失 | 在线准确率 | 在线正确数 |
|---|---:|---:|---:|
| 1 | 1.66007185 | 41.42% | 2071/5000 |
| 2 | 0.62978381 | 79.60% | 3980/5000 |
| 3 | 0.38978881 | 87.68% | 4384/5000 |

三轮结束后固定同一组参数，5000训练图、1000测试图顺序仅前向，共6000次前向。真实标签、每图loss和预测记录独立累计，对照返回GPU均值和准确率；评估前后全部参数值严格一致，参数身份、存储地址保留，评估反向与训练调用均为0。

| 最终固定模型评估范围 | 图片数 | 平均损失 | 准确率 | 正确数 |
|---|---:|---:|---:|---:|
| 训练集 | 5000 | 0.34107816 | 89.24% | 4462 |
| 测试集 | 1000 | 0.41187373 | 87.50% | 875 |

独立NumPy float64完整模型只抽查训练索引[0,1,127,4999]与测试索引[0,1,999]，共7图，loss全部在rtol=2e-6、atol=2e-6下通过；最大绝对差 `2.659052525e-07`。本版没有将全部6000图逐项再次与CPU前向比较，也没有重复全参数中心差分；先前v047已核验同评估函数的1128图。

真实输出：

```text
第 1/3 轮 | 已训练 500/5000 | 在线平均损失: 2.2966 | 在线准确率: 14.00%
第 1/3 轮 | 已训练 1000/5000 | 在线平均损失: 2.2057 | 在线准确率: 19.60%
第 1/3 轮 | 已训练 1500/5000 | 在线平均损失: 2.1183 | 在线准确率: 22.47%
第 1/3 轮 | 已训练 2000/5000 | 在线平均损失: 2.0606 | 在线准确率: 24.90%
第 1/3 轮 | 已训练 2500/5000 | 在线平均损失: 2.0030 | 在线准确率: 27.52%
第 1/3 轮 | 已训练 3000/5000 | 在线平均损失: 1.9531 | 在线准确率: 29.47%
第 1/3 轮 | 已训练 3500/5000 | 在线平均损失: 1.8923 | 在线准确率: 32.49%
第 1/3 轮 | 已训练 4000/5000 | 在线平均损失: 1.8134 | 在线准确率: 35.28%
第 1/3 轮 | 已训练 4500/5000 | 在线平均损失: 1.7419 | 在线准确率: 38.02%
第 1/3 轮 | 已训练 5000/5000 | 在线平均损失: 1.6601 | 在线准确率: 41.42%
第 2/3 轮 | 已训练 500/5000 | 在线平均损失: 0.8486 | 在线准确率: 71.60%
第 2/3 轮 | 已训练 1000/5000 | 在线平均损失: 0.7884 | 在线准确率: 74.40%
第 2/3 轮 | 已训练 1500/5000 | 在线平均损失: 0.7532 | 在线准确率: 75.53%
第 2/3 轮 | 已训练 2000/5000 | 在线平均损失: 0.7177 | 在线准确率: 76.15%
第 2/3 轮 | 已训练 2500/5000 | 在线平均损失: 0.6844 | 在线准确率: 77.48%
第 2/3 轮 | 已训练 3000/5000 | 在线平均损失: 0.6757 | 在线准确率: 77.73%
第 2/3 轮 | 已训练 3500/5000 | 在线平均损失: 0.6616 | 在线准确率: 78.40%
第 2/3 轮 | 已训练 4000/5000 | 在线平均损失: 0.6547 | 在线准确率: 78.53%
第 2/3 轮 | 已训练 4500/5000 | 在线平均损失: 0.6448 | 在线准确率: 78.98%
第 2/3 轮 | 已训练 5000/5000 | 在线平均损失: 0.6298 | 在线准确率: 79.60%
第 3/3 轮 | 已训练 500/5000 | 在线平均损失: 0.4323 | 在线准确率: 86.00%
第 3/3 轮 | 已训练 1000/5000 | 在线平均损失: 0.4451 | 在线准确率: 86.20%
第 3/3 轮 | 已训练 1500/5000 | 在线平均损失: 0.4249 | 在线准确率: 86.27%
第 3/3 轮 | 已训练 2000/5000 | 在线平均损失: 0.4093 | 在线准确率: 86.95%
第 3/3 轮 | 已训练 2500/5000 | 在线平均损失: 0.3928 | 在线准确率: 87.56%
第 3/3 轮 | 已训练 3000/5000 | 在线平均损失: 0.3859 | 在线准确率: 87.67%
第 3/3 轮 | 已训练 3500/5000 | 在线平均损失: 0.3891 | 在线准确率: 87.69%
第 3/3 轮 | 已训练 4000/5000 | 在线平均损失: 0.3859 | 在线准确率: 87.73%
第 3/3 轮 | 已训练 4500/5000 | 在线平均损失: 0.3869 | 在线准确率: 87.64%
第 3/3 轮 | 已训练 5000/5000 | 在线平均损失: 0.3898 | 在线准确率: 87.68%
本段新增更新次数: 15000
完整训练集样本数: 5000
完整训练集平均损失: 0.34107816219329834
完整训练集准确率: 89.24%
测试集样本数: 1000
测试集平均损失: 0.4118737280368805
测试集准确率: 87.50%
W_in 的实际设备: <CUDA Device 0>
W_cls 的实际设备: <CUDA Device 0>
W_in 的数据类型: float32
```

全部20参数最终float32、GPU0、有限、保留对象身份/存储地址/原变量引用；所有既有非参数数组、完整数据集、固定位置编码、历史grads/grads_new/cache_after_update/cache_step及NumPy索引数组内容未变。hook已恢复。主文件、数据和CPU参考文件均未改变。观察训练加评估用时 `205.589` 秒，包含日志、CPU小复制、检查及hook开销，不是GPU吞吐或训练性能基准。

JSON保存3轮完整order、15000步更新前日志、30个在线指标对照、18次精确单步检查范围、6000图固定评估日志、最终指标、全部参数实际前后快照、精确聊天片段及SHA256、主文件和数据哈希、inspect提取的forward/backward/train/evaluate源码及哈希、v039独立CPU参考全文与哈希、7图CPU对照、环境和真实stdout。

[本步代码](#本步代码) · [真实训练、固定评估及参数证据](2026-10-08_transformer_full_training_validation_v048.json)。相对链接已核验，本版无新增图。

## 与前版变化、未解限制及下一步

v047只有固定评估，没有新增优化。本版沿用其参数扩大到已加载全部5000张训练图，新增3轮15000次优化及最终6000图固定评估，区分online指标和固定模型指标。旧记录保留，项目索引追加v048；同步只复制已有Markdown/JSON，不导入模型或重跑。

当前训练范围仅MNIST前5000张、测试范围仅前1000张，并非完整原数据集；这些结果不能外推为完整MNIST成绩。没有额外验证集、学习率/轮数搜索、随机种子重复实验或收敛证明。前三轮实际结果为本次运行事实，不保证其他环境或随机顺序逐位相同。单图GPU实现适合教学，但Python循环与小矩阵限制利用率。参数仅在独立核验进程内存中更新，未保存训练checkpoint；用户按聊天执行的参数在其自己的进程中。后续可保存参数、查看预测错误图片，或在保持手写梯度的前提下学习批量计算。
