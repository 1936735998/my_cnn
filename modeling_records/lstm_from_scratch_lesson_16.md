# 手写 LSTM 第 16 步：遍历训练图片与 epoch 平均损失
日期：2026-10-06；版本：v016。

接续[第15步](lstm_from_scratch_lesson_15.md)。用户单图train_one输出loss从0.0881374096811086降至0.08695209627476536，与前版吻合。

将以下片段追加到F:\PythonProjects\deep_learning\lstm_from_scratch.py末尾。每轮遍历当前选取的5000张训练图片，共5轮；沿用已经存在的参数与设备信息输出，未在循环内初始化权重。助手未修改用户主文件。

```python
# ==================================================
# 遍历训练图片，统计每一轮的平均损失
# ==================================================

epochs = 5
learning_rate = 0.01
loss_history = []

for epoch in range(epochs):
    total_loss = 0.0

    for j in range(len(x_train)):
        loss = train_one(x_train[j], y_train[j], learning_rate)
        total_loss += loss

    average_loss = total_loss / len(x_train)
    loss_history.append(average_loss)

    print(f"epoch: {epoch + 1} loss: {average_loss:.6f}")
```

## 理解循环与损失

- epoch是完整遍历当前5000张训练图片一次。5轮新增25000次单图参数更新；加上前面的2次首图演示更新，完整脚本总计25002次。
- 外层循环控制轮次；内层j依次取0至4999，train_one接收图片与其真实标签，完成一次前向、完整反向、更新。
- total_loss每轮置零，只累加这一轮的损失；loss已经是10个输出的MSE，除以图片数即可，不再除以10。
- average_loss是本轮各样本更新前loss的均值，参数在轮内不断变化。它不是轮末固定模型对全训练集重算得到的误差，更不是准确率。
- loss_history记录每轮平均损失，.6f只控制打印保留6位小数，不改变实际计算精度。
- 本步继续固定数据顺序，不新增打乱、mini-batch、自动求导或优化器；每张图片的h/c由forward重新置零。

## 本轮实际训练结果

本机项目Python3.13.15、NumPy2.5.3，NumPy float64 CPU运行新增循环一次，训练阶段实测约74.06秒。同一环境、已保存前置代码与当前数据条件下实际输出：

```text
epoch: 1 loss: 0.082248
epoch: 2 loss: 0.072087
epoch: 3 loss: 0.064978
epoch: 4 loss: 0.059144
epoch: 5 loss: 0.054003
```

每次样本损失与每轮参数有限性检查通过，共25000次新增更新。先前2次演示更新在独立命名空间重放以恢复当前起点；没有再次重复5轮训练。这里loss下降是训练结果，识别准确率需要下一步用独立测试图片评估。

保存了[完整代码快照](2026-10-06_lstm_training_code_v016.py)、[训练后参数](2026-10-06_lstm_parameters_v016.npz)、[真实结果JSON](2026-10-06_lstm_training_validation_v016.json)与[建模记录](2026-10-06_lstm_mnist_v016.md)。参数文件重新加载后十组数组与训练结果逐元素一致；其用途是后续验证复用本次训练，用户主文件的上述片段本身没有加入保存语句。
