# MNIST 手写 Transformer 建模记录

日期：2026-10-08（Asia/Shanghai）；记录版本：v053；教学步骤：继续第51步。
阶段：独立文件中的无标签单图预测与三图实际GPU一致性核验。本步实际推理，未训练、未完整评估；不是仅文档更新。

## 范围、输入来源与处理（事实）

用户希望在新独立验证文件末尾继续第51步教学。本版添加predict_one_image：复用已有infer_one_image的概率，再int(cp.argmax)得到预测数字，返回数字和GPU概率数组。使用新文件现有params、position_encoding、num_heads、ln_eps；无需旧脚本loaded_*变量或真实标签。代码由用户自行接到新文件末尾，核验没有改写用户新文件或原训练主文件。

实际存档为 F:\PythonProjects\deep_learning\outputs\checkpoints\transformer_mnist_20261008_115104_418995.npz，SHA256 349eff0650c37163be783b58d4f39d256b796091884c624428f131d2a320f218；参数源与v052完全相同。原数据 C:\Users\19367\.keras\datasets\mnist.npz，SHA256 731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1；初始化依原存档pixel_divisor=255读取原前1000测试图并归一化一次，标签CPU int64已由独立文件加载但不传入新预测。没有新的清洗、增强或训练数据读取。教学调用直接使用已归一化x_test[0]，不再次除255。实际后台额外预测仅x_test[1]、x_test[999]。

## 假设、变量与公式

保持既有28行token、输入28、d_model32、4头各8维、前馈64、单post-LN编码器、固定位置编码、均值池化和10分类。20参数数组共9802标量，全部GPU0 float32。X为同原MNIST朝向的GPU float32 28×28图像，已按训练规则归一化。infer_one_image返回logits与概率P，新封装丢弃logits并返回Python int(argmax(P))、GPU float32 shape(10,)的P。概率索引0～9对应数字0～9；概率和float32舍入容差内为1。

~~~text
_, P = infer_one_image(X, params, position_encoding, num_heads, ln_eps)
prediction = int(argmax(P))
return prediction, P
~~~

## 方法选择理由（判断）

独立文件已含完整手写前向，本步直接复用它以减少重复数学代码。封装只新增返回预测数字的接口，不改变参数、网络或预处理。真实标签用于评估，单图预测不需要标签。复用已有前向仍保持手写基础CuPy运算，没有自动求导或网络层。

## 本步代码

[精确片段附件](transformer_step51_standalone_fragment.txt) · [独立文件代码副本](transformer_validation_v052.py)。

~~~python
def predict_one_image(X, params, position_encoding, num_heads, ln_eps):
    # 复用新文件中的手写前向，不需要真实标签。
    _, probabilities = infer_one_image(
        X, params, position_encoding, num_heads, ln_eps
    )

    # 概率最大的下标，就是预测的数字。
    prediction = int(cp.argmax(probabilities))
    return prediction, probabilities


# x_test 已经归一化，这里直接使用，不再除以 255。
inference_sample_index = 0
inference_prediction, inference_probabilities = predict_one_image(
    x_test[inference_sample_index], params,
    position_encoding, num_heads, ln_eps
)

print("预测图片索引:", inference_sample_index)
print("预测数字:", inference_prediction)
print("概率数组对应的数字:", list(range(10)))
print("10 个分类概率:", inference_probabilities)
print("预测类别的概率:", float(inference_probabilities[inference_prediction]))
print("概率之和:", float(cp.sum(inference_probabilities)))
print("概率数组的形状:", inference_probabilities.shape)
print("概率数组的实际设备:", inference_probabilities.device)
print("概率数组的数据类型:", inference_probabilities.dtype)
~~~

## 实际执行、验证与结果（事实）

解释器 F:\PythonProjects\deep_learning\.venv\Scripts\python.exe，Python 3.13.15、NumPy 2.5.3、CuPy 14.2.0，GPU0 NVIDIA GeForce RTX 5060。
核验从已核验v052源码的AST只执行初始化、存档/测试数据读取及两个前向函数定义，截断在infer_one_image定义末尾。未执行原独立文件的1000图顶层评估循环，未导入/执行旧训练主文件。

精确聊天片段实际执行一次第0图预测，输出为：

~~~text
预测图片索引: 0
预测数字: 7
概率数组对应的数字: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
10 个分类概率: [3.2103888e-08 1.6617516e-06 9.4790786e-04 4.2048637e-03 8.1715461e-06
 1.2660606e-04 2.2808545e-10 9.9433112e-01 1.1790193e-06 3.7854185e-04]
预测类别的概率: 0.9943311214447021
概率之和: 1.0000001192092896
概率数组的形状: (10,)
概率数组的实际设备: <CUDA Device 0>
概率数组的数据类型: float32
~~~

后台仅另外调用第1及999图预测，总计三次predict调用。AST确认每次封装只调用infer一次，因此共3次完整前向；没有额外参考前向、反向、训练或参数更新。三图所有10类概率逐项与v051已保存GPU结果逐位一致，最大绝对差均0.0；返回预测类别为Python int，概率shape(10,)、GPU0 float32、有限且范围0～1、概率和在容差内为1。所有20参数、固定位置编码和测试图/标签内容哈希不变；实际存档、主文件、源码片段、原数据及v048～v052旧记录字节哈希不变。

JSON保存执行前缀源码、精确片段与哈希、数据/参数/文件保护哈希、配置、三图全部概率和真实输出。[实际执行证据](2026-10-08_transformer_standalone_prediction_validation_v053.json)。相对链接已核验，无新增图。v052测试平均损失0.4118737280368805、875/1000、87.50%仅作为历史结果引用，本步没有重算完整准确率。

## 与前版变化、限制及下一步

v052给出了独立存档加载和1000图验证。本记录v053把原教学第51步适配到该文件已有变量，通过简单封装新增无标签predict_one_image；教学步骤仍是第51步，记录版本单独递增。保留旧记录并追加索引，文档同步仅复制既有证据/代码，不运行模型。

核验进程独立于用户终端内存，仅检查同一checkpoint和三张原MNIST图；不覆盖任意输入，不声称泛化、收敛或概率校准。新预测依赖独立文件先完成初始化与已有infer定义。用户如果运行包含v052完整评估循环的新文件，会先完成1000图评估再执行追加的首图预测；本次后台核验有意跳过该循环，并不声称已重新执行完整文件。外部图片仍需与训练一致的尺寸、灰度、前景/背景与归一化。下一步可复用predict_one_image查看其他已加载测试图，或继续教学错误样本分析。
