# MNIST 手写 Transformer 建模记录

日期：2026-10-07（Asia/Shanghai）；版本：v046。
阶段：前128张训练图片的一轮GPU训练循环和在线指标核验。本轮真实执行，不是仅文档更新；没有完成5000张训练数据的完整训练，没有测试集评估。

## 问题范围、输入来源与处理（事实）

用户报告v045单图训练步骤输出，第二张训练图片真实标签0，更新前loss约2.92177、更新后约2.61195，预测仍为5。本版继续教学，将已经写好的train_one_image用于多图循环：前128张图片洗牌一次，依次fresh前向、完整手写反向和一次SGD，共一轮128次更新；每32步打印累计在线平均损失与准确率。

模型代码贴在聊天，用户自行加入主文件；本次核验不编辑 `F:\PythonProjects\deep_learning\transformer_mnist.py`。独立进程runpy执行其已经保存至v045的代码，重放先前两次单图更新，然后执行精确聊天片段；主文件SHA256 `65af79b543b3e139b7532e60fb7b9d9ca18a6627bc525b4a3d638b7b85a19524`，运行前后不变。没有修改用户终端实时内存，没有保存训练检查点。

沿用本地MNIST `C:\Users\19367\.keras\datasets\mnist.npz`，数据SHA256 `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`；已取前5000训练图、前1000测试图，像素在GPU0 float32并除以255，标签int64。本轮使用训练前128张，未改变数据划分、清洗、增强、归一化或标签。标签切片只复制一次到CPU，洗牌使用CuPy生成随机排列再将128个索引复制到CPU供Python循环使用；输入图片、参数、前向、反向、SGD及累计指标计算仍在GPU0。打印将少量指标转为Python数值。

## 假设、变量、公式与指标语义

继续使用28行位置、每行28像素、d_model=32、4个8维头、ReLU前馈宽度64、单个post-LN编码器、固定正弦/余弦位置编码、均值池化及十类交叉熵；20组参数共9802个标量，学习率float32(0.01)。所有梯度来自当前步骤当前参数的前向缓存，完整反向后统一原地SGD。

```text
order = permutation(0,...,127)
对第t张图i_t：L_t,P_t=forward(theta_t,X_i_t,y_i_t)
G_t=backward(theta_t,cache_t)
theta_(t+1)=theta_t-learning_rate*G_t
online_loss(t)=(L_1+...+L_t)/t
online_accuracy(t)=sum(argmax(P_j)==y_i_j,j=1..t)/t
```

loss与预测均为该图本次更新前结果，参数随每一步变化，因此online_accuracy不是轮末固定模型的训练集准确率，也不是测试准确率。固定最终参数的评价需另行只前向遍历数据，期间不更新参数。样本每轮恰好出现一次，顺序随机；本轮1个epoch不声称收敛。

## 方法选择理由（判断）

先用128图一轮，让初学者看清训练顺序、调用次数和指标累积，再逐步加入固定参数评价与更完整训练。单图SGD便于接续已有手写函数；它适合作为教学阶段，但GPU小张量和Python循环的效率不能据此代表批量实现。

本轮主要风险是旧缓存/旧梯度复用、参数重置、更新次数错误和在线指标与固定模型准确率混淆。已有完整反向梯度验证无需重复；核验采用透明委托hook观察原函数，128次逐步核实fresh前向、fresh缓存、fresh完整反向与每组参数的一次float32 SGD，再将128条实际记录独立累计，核对GPU指标。hook只观察和计算对照，不替代原函数或额外更新参数。

## 本步代码

```python
# ==================================================
# 第一次多图训练：前 128 张图片，训练 1 轮
# ==================================================

num_train = min(128, x_train.shape[0])
num_epochs = 1

# 标签只复制一次，方便 Python 循环取出整数
train_labels_cpu = cp.asnumpy(y_train[:num_train])

for epoch in range(num_epochs):
    # 每轮打乱训练顺序，每张图片恰好使用一次
    order = cp.asnumpy(cp.random.permutation(num_train))

    loss_sum = cp.zeros((), dtype=cp.float32)
    correct_sum = cp.zeros((), dtype=cp.int64)

    for step, index in enumerate(order, start=1):
        index = int(index)
        target_i = int(train_labels_cpu[index])

        loss_i, probabilities_i = train_one_image(
            x_train[index], target_i, params, position_encoding,
            num_heads, ln_eps, learning_rate
        )

        # 累计每次更新前的损失和预测结果
        loss_sum += loss_i
        correct_sum += (cp.argmax(probabilities_i) == target_i)

        if step % 32 == 0 or step == num_train:
            mean_loss = float(loss_sum / cp.float32(step))
            online_accuracy = float(
                correct_sum.astype(cp.float32) / cp.float32(step)
            ) * 100.0

            print(
                f"第 {epoch + 1}/{num_epochs} 轮 | 已训练 {step}/{num_train} | "
                f"在线平均损失: {mean_loss:.4f} | "
                f"在线准确率: {online_accuracy:.2f}%"
            )

print("本段新增更新次数:", num_train * num_epochs)
print("本轮在线预测正确数量:", int(correct_sum))
print("W_in 的实际设备:", params["W_in"].device)
print("W_cls 的实际设备:", params["W_cls"].device)
print("W_in 的数据类型:", params["W_in"].dtype)
```

## 实际执行、验证与结果（事实）

解释器 `F:\PythonProjects\deep_learning\.venv\Scripts\python.exe`，Python 3.13.15、NumPy 2.5.3、CuPy 14.2.0；可见GPU 1张，GPU0 NVIDIA GeForce RTX 5060。真实GPU执行后同步CUDA，没有安装或更换依赖。

每个样本通过输入视图存储地址对应原x_train索引，标签与原始y_train严格一致。实际顺序为0..127的排列，每图恰好一次；fresh前向128次、fresh缓存128个、完整反向128次、新SGD更新128次。完整20组参数每次精确匹配fresh梯度的一次float32 SGD公式，不要求近零梯度的偏置每次都显著改变。先前保存代码重放2次更新，加本片段128次，总130次。没有在这段循环中重置参数或使用旧全局梯度。

所有20参数保留原对象身份、存储地址和原变量引用，float32、GPU0、有限，参数数仍9802。全训练/测试数据、固定位置编码及历史全局数组、grads、grads_new、cache_after_update、cache_step内容和引用全部未变。透明hook运行后已恢复原函数。

实际128条更新前损失均有限。GPU累计损失 `301.146911621` 与按日志同顺序float32累计绝对差 `0`；累计正确数量严格匹配日志为 18/128。轮末在线平均损失 `2.35271024704`，在线准确率 `14.0625%`。这些是实际训练过程指标，不预设loss必降或accuracy达到某值。

真实输出：

```text
第 1/1 轮 | 已训练 32/128 | 在线平均损失: 2.2917 | 在线准确率: 18.75%
第 1/1 轮 | 已训练 64/128 | 在线平均损失: 2.3376 | 在线准确率: 14.06%
第 1/1 轮 | 已训练 96/128 | 在线平均损失: 2.3377 | 在线准确率: 15.62%
第 1/1 轮 | 已训练 128/128 | 在线平均损失: 2.3527 | 在线准确率: 14.06%
本段新增更新次数: 128
本轮在线预测正确数量: 18
W_in 的实际设备: <CUDA Device 0>
W_cls 的实际设备: <CUDA Device 0>
W_in 的数据类型: float32
```

训练结束后用当前参数只前向核验训练索引0单图，与v039独立NumPy float64完整模型比较：GPUloss `2.74152803421`，CPUloss `2.74152824243`，绝对差 `2.08223585e-07`，rtol=2e-6、atol=2e-6下通过。此一次单图实现核验不是固定模型的数据集评价，没有测试集结果。没有额外训练或重复全参数中心差分。

核验循环用时 `3.32408` 秒，包含逐步骤复制、CUDA同步及hook核验开销，不是模型性能基准。JSON保存实际完整128条order/样本/标签/pre-update loss/概率/预测/正确计数，每32步在线指标、所有参数实际前后快照与变化范数、计数130（新增128），精确聊天片段、主文件和数据SHA256、inspect提取的完整forward/backward/train函数源代码及哈希、独立CPU参考源代码及哈希和GPU环境。

[本步代码](#本步代码) · [真实训练顺序、指标及参数证据](2026-10-07_transformer_training_loop_validation_v046.json)。相对链接已核验；本版没有新增图。

## 与前版变化、限制与下一步

v045完成第二张图片的单图训练步骤；本版将该函数用于前128张训练图的一轮随机循环，并明确在线更新前指标。记录新版本，保留旧记录，向项目索引追加。记录同步只复制已生成记录和JSON，不导入模型或重跑训练。

本核验过程参数只存在独立进程内存，不保存checkpoint、不修改用户主文件；没有5000张完整训练或1000张测试集评价。128图一次只能展示训练流程；不能证明收敛、泛化或指定准确率。单图SGD教学实现不是GPU性能最优选择。学习率0.01为沿用教学候选值，不宣称最优。下一步保持参数固定，仅前向计算明确数据范围上的平均loss和准确率，随后再决定更完整训练范围。
