# 手写 MNIST DNN：第 9 步
日期：2026-10-04；版本：v009。

用户报告的 5 轮训练损失逐项与 v008 独立运行一致。下一步在训练循环结束后追加测试评价，测试循环必须顶格，不能缩进到训练轮循环中。

```python
# ==================================================
# 测试集评价
# ==================================================

correct = 0

for i in range(len(x_test)):

    x = x_test[i]
    label = y_test[i]

    # 只做前向传播
    z1 = np.dot(x, W1) + b1
    a1 = relu(z1)
    output = np.dot(a1, W2) + b2

    prediction = np.argmax(output)

    if prediction == label:
        correct += 1

    if i < 10:
        print(
            "测试图片:", i,
            "真实标签:", label,
            "预测类别:", prediction
        )

accuracy = correct / len(x_test)

print("测试图片数:", len(x_test))
print("预测正确数:", correct)
print(f"测试准确率: {accuracy:.2%}")
```

correct 统计预测正确数量；prediction 是十个输出分数中最大值所在的下标；accuracy=correct/测试数量。测试阶段只前向，不计算梯度或更新参数。这里的数据已在第 1 步限定为前 1000 张测试图片。训练中的损失表示输出与 one-hot 的距离，准确率只看最大分数对应类别是否正确，二者不是同一指标。

助手直接复用 v008 保存的训练参数，不重复训练。真实评价结果：1000 张中正确 876 张，准确率 0.876，即 87.60%；前十张中第 8 号（从零计数）真实为 5、预测为 6，其余前十张预测正确。完整预测明细已保存。该结果限于前 5000 个训练样本训练 5 轮、前 1000 个测试样本评价的设置，不是完整 MNIST 基准成绩。

至此，基础 NumPy DNN 的数据准备、参数初始化、前向、损失、反向、更新、训练和测试流程都已覆盖。助手仍没有修改用户主 Python 文件，由用户在 VS Code 追加代码并运行。

[本步建模记录](2026-10-04_dnn_mnist_v009.md) · [真实测试结果](2026-10-04_dnn_test_validation_v009.json) · [逐样本预测](2026-10-04_dnn_test_predictions_v009.csv)
