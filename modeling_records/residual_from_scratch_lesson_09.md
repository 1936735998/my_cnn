# 手写 MNIST 残差网络：第 9 步
日期：2026-10-04；版本：v009。

助手读取用户已保存主文件，确认训练循环已写入，但没有据此声称用户终端训练已成功完成。把下面代码追加在整个训练循环之后，测试循环顶格，不能放进 epoch 循环。

```python
# ==================================================
# 测试集评价
# ==================================================

correct = 0

for i in range(len(x_test)):

    x = x_test[i]
    label = y_test[i]

    # 输入映射
    z1 = np.dot(x, W1) + b1
    h = relu(z1)

    # 残差分支
    z2 = np.dot(h, W2) + b2
    u = relu(z2)
    F_h = np.dot(u, W3) + b3

    # 相加与输出
    s = h + F_h
    r = relu(s)
    output = np.dot(r, W4) + b4

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

测试保持同一残差前向：输入映射、两层分支、h+F_h、ReLU、十维输出。只计算类别与正确数，不做梯度或更新参数。accuracy=correct/len(x_test)。测试数据仍是前 1000 张，标签只用于评价。

助手复用 v008 已保存八组参数，不重训，实际得到 872/1000，准确率 87.20%。前十张的第 8 号（从零计数）真实为 5、预测为 6，其余正确。完整逐样本记录已保存。结果限于本次 5000 训练/1000 测试的教学设置，不是完整 MNIST 基准，也不证明残差网络优于先前 DNN。

至此基础全连接残差网络的数据、前向、损失、完整手写反向、学习更新、循环训练与测试评价代码已齐。用户主文件由用户自行追加和运行，助手没有修改它。

[本步建模记录](2026-10-04_residual_mnist_v009.md) · [真实测试结果](2026-10-04_residual_test_validation_v009.json) · [逐样本预测](2026-10-04_residual_test_predictions_v009.csv)
