# 手写 LSTM 第 17 步：测试集预测与准确率
日期：2026-10-06；版本：v017。

接续[第16步](lstm_from_scratch_lesson_16.md)。用户5轮训练输出与前版在线平均MSE的6位小数吻合。训练loss下降不直接给出准确率，本步使用原MNIST测试划分的前1000张图片统计正确率。

把下面代码追加到F:\PythonProjects\deep_learning\lstm_from_scratch.py训练循环之后，由用户继续逐步编写；助手未修改主源。

```python
# ==================================================
# 使用测试集评估准确率
# ==================================================

correct = 0

for j in range(len(x_test)):
    output_test, _ = forward(x_test[j])
    prediction_test = np.argmax(output_test)
    label_test = y_test[j]

    if prediction_test == label_test:
        correct += 1

    # 展示前 10 张测试图片的预测
    if j < 10:
        print(
            "测试图片:", j,
            "真实标签:", label_test,
            "预测类别:", prediction_test
        )

accuracy = correct / len(x_test)

print("测试图片数:", len(x_test))
print("预测正确数:", correct)
print(f"测试准确率: {accuracy:.2%}")
```

## 读懂评估流程
1. forward每次从零h/c读取一张图片，返回10个类别的线性分数与缓存。下划线接收本次不用的缓存。
2. prediction_test=np.argmax(output_test)取最大分数的位置；位置0至9对应数字类别。分数本身不是概率，分类取最大值无需softmax。
3. 预测类别与真实label_test相同，就让correct增加1。accuracy=正确图片数/全部参与测试的图片数。
4. if j<10只是控制台展示前10张，统计仍然覆盖当前x_test的1000张。
5. 测试阶段只调用forward，不调用train_one/backward、不更新参数。测试图片不能用于本步骤的训练更新。

## 实际核验
独立执行复用[v016训练参数](2026-10-06_lstm_parameters_v016.npz)，从用户主源提取sigmoid/forward定义，与v016代码中的定义一致；加载相同本地MNIST测试划分前1000图、除以255，不重放训练或前置演示SGD。Python3.13.15、NumPy2.5.3、NumPy float64 CPU，新增优化次数0。1000次输出均有限，每图初始状态零，十组权重评估前后逐元素一致；参数文件及用户主源字节保持不变。

本次正确634/1000，准确率63.40%，错误366张；预测阶段约0.741秒。按相同代码、训练状态和数据条件，预计得到：

```text
测试图片: 0 真实标签: 7 预测类别: 7
测试图片: 1 真实标签: 2 预测类别: 2
测试图片: 2 真实标签: 1 预测类别: 1
测试图片: 3 真实标签: 0 预测类别: 6
测试图片: 4 真实标签: 4 预测类别: 4
测试图片: 5 真实标签: 1 预测类别: 1
测试图片: 6 真实标签: 4 预测类别: 7
测试图片: 7 真实标签: 9 预测类别: 9
测试图片: 8 真实标签: 5 预测类别: 4
测试图片: 9 真实标签: 9 预测类别: 7
测试图片数: 1000
预测正确数: 634
测试准确率: 63.40%
```

这是测试集前1000张的实测结果，不是全部10000张测试集的准确率。这里使用助手前一版实际训练保存的checkpoint，没有读取用户VS Code运行进程内存；用户运行结果待其报告。

## 保存文件
[本课建模记录](2026-10-06_lstm_mnist_v017.md) · [完整训练加评价代码快照](2026-10-06_lstm_evaluation_code_v017.py) · [全部1000条真实预测与评价JSON](2026-10-06_lstm_test_validation_v017.json)。JSON还保存混淆矩阵、各类样本数与正确数；没有新训练或图表。
