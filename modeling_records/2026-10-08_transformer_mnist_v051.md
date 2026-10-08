# MNIST 手写 Transformer 建模记录

日期：2026-10-08（Asia/Shanghai）；版本：v051。
阶段：实际用户存档的无标签单图GPU预测教学与三图概率一致性核验。本步实际执行模型推理，没有新增训练，没有全集评估；不是仅文档更新。

## 问题范围、输入来源与处理（事实）

用户已报告v050加载结果：20参数数组、9802标量、GPU0 float32、固定位置编码28×32；全部参数及位置编码与当前模型相同。测试第0图原模型及恢复模型都预测7，loss=0.005684994161128998，概率最大差0.0。当前步骤接在原脚本后面，定义无需真实标签的predict_one_image，使用loaded_params和加载配置直接返回预测数字及10类概率。代码贴聊天，由用户自行接入；核验没有编辑或运行主文件。

实际读取用户此前保存的 F:\PythonProjects\deep_learning\outputs\checkpoints\transformer_mnist_20261008_115104_418995.npz。原存档大小50006 bytes、SHA256 349eff0650c37163be783b58d4f39d256b796091884c624428f131d2a320f218；使用allow_pickle=False读取26个非object成员。20参数的shape、float32 dtype及内容字节SHA256逐项与v048实际训练后的快照一致；位置编码哈希一致，配置值及dtype通过。数组直接cp.asarray复制到GPU0，没有再次训练、随机初始化或重放原步骤。原用户存档不改写；记录附件是该文件的逐字节复制，没有重新序列化。

原MNIST数据 C:\Users\19367\.keras\datasets\mnist.npz，SHA256 731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1。只加载原前1000张测试图，float32像素依存档pixel_divisor=255除一次，标签int64；数组哈希与v048一致。教学片段直接使用已归一化的x_test，没有再次除255。原标签只在后台与旧forward的CE参考对照时使用，故意不放入教学片段命名空间。没有新增清洗、增强或数据划分，没有加载训练集。

原主文件 F:\PythonProjects\deep_learning\transformer_mnist.py SHA256 7177a3632cf49ab8829c8744aa13d6ca1f65cbb3ee8a3cf7de4da42c37170aec。核验只通过AST执行CUDA启动段至import cupy和forward_one_image函数定义；原forward源码逐字节匹配v048已验证代码。没有执行初始化、训练循环、原评估、保存参数或旧加载演示。v048证据SHA256 7a0d42e1e0e7a65e2693c37a14fe853ff9330d961801a94f5dc60a81d115ef12，所有有关v048/v049/v050旧Markdown和JSON、主文件、数据、原存档在本次核验前后哈希不变。

## 假设、变量与公式

结构保持28行token、每行28输入特征、d_model32、4头各8维、前馈64、单post-LN编码器、固定位置编码、均值池化、10分类。20参数数组共9802标量。输入X是GPU float32的28×28灰度数组，并已经按原训练预处理归一化到0～1；输入行顺序和像素方向与原MNIST一致。加载配置num_heads=4、ln_eps=float32(1e-5)、位置编码28×32，文件format_version=1。

~~~text
E = X @ W_in + b_in + position_encoding
Q_h, K_h, V_h = E的三个线性投影，拆为4个28×8头
A_h = softmax(Q_h @ K_h.T / sqrt(8))
Z = concat(A_h @ V_h) @ W_O + b_O
N = LayerNorm(E + Z)
F = ReLU(N @ W_ff1 + b_ff1) @ W_ff2 + b_ff2
encoder_output = LayerNorm(N + F)
features = mean(encoder_output, axis=0)
logits = features @ W_cls + b_cls
P_j = exp(logits_j - max(logits)) / sum_k exp(logits_k - max(logits))
prediction = Python int(argmax(P))
返回prediction及GPU float32概率P；没有真实标签、CE或反向缓存。
~~~

新函数参数是X、params、position_encoding、num_heads、ln_eps。返回类别为Python int，概率为GPU0 float32 shape(10,)。P中的索引0～9对应数字0～9。模型概率表示当前softmax输出，尚无概率校准结果；不等同于该预测正确的经验证频率。

## 方法选择理由（判断）

训练forward_one_image为计算监督损失必须接收target，并返回反向传播缓存。单图预测只需从输入计算logits、softmax、argmax，因此沿用同一前向数学，删除loss和cache，直接写独立的无标签预测函数。这样可以在没有真实答案时调用恢复模型，继续保持基础CuPy算子手写网络，没有新增依赖、网络层或自动求导。

本步对网络计算至probabilities的每个语句做AST等价比较，再对三张测试图片比较全部10类概率，检查删除训练用途代码后推理是否一致。原forward此前已有完整模型数值核验，本版没有必要重做有限差分或重训练；三图一致性范围清楚，不将其当作全集准确率复评。

## 本步代码

~~~python
def predict_one_image(X, params, position_encoding, num_heads, ln_eps):
    seq_len = X.shape[0]
    d_model = params["W_in"].shape[1]
    head_dim = d_model // num_heads

    # 1. 输入映射与位置编码
    embedding = cp.dot(X, params["W_in"]) + params["b_in"]
    encoder_input = embedding + position_encoding

    # 2. Q、K、V 投影与拆头
    Q = cp.dot(encoder_input, params["W_Q"]) + params["b_Q"]
    K = cp.dot(encoder_input, params["W_K"]) + params["b_K"]
    V = cp.dot(encoder_input, params["W_V"]) + params["b_V"]

    Q_heads = Q.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)
    K_heads = K.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)
    V_heads = V.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)

    # 3. 缩放点积注意力
    attention_scores = cp.matmul(Q_heads, K_heads.transpose(0, 2, 1))
    attention_scores = attention_scores / cp.sqrt(cp.float32(head_dim))
    scores_shifted = attention_scores - cp.max(
        attention_scores, axis=-1, keepdims=True
    )
    exp_scores = cp.exp(scores_shifted)
    attention_weights = exp_scores / cp.sum(exp_scores, axis=-1, keepdims=True)

    head_output = cp.matmul(attention_weights, V_heads)
    concatenated_heads = head_output.transpose(1, 0, 2).reshape(seq_len, d_model)
    attention_output = cp.dot(concatenated_heads, params["W_O"]) + params["b_O"]

    # 4. 注意力残差与第一次 LayerNorm
    attention_residual = encoder_input + attention_output
    mean_attn = cp.mean(attention_residual, axis=-1, keepdims=True)
    centered_attn = attention_residual - mean_attn
    var_attn = cp.mean(centered_attn ** 2, axis=-1, keepdims=True)
    inv_std_attn = cp.float32(1.0) / cp.sqrt(var_attn + ln_eps)
    normalized_attn = centered_attn * inv_std_attn
    attention_norm = normalized_attn * params["gamma_attn"] + params["beta_attn"]

    # 5. 前馈网络、残差与第二次 LayerNorm
    ff_hidden_linear = cp.dot(attention_norm, params["W_ff1"]) + params["b_ff1"]
    ff_hidden = cp.maximum(ff_hidden_linear, cp.float32(0.0))
    ff_output = cp.dot(ff_hidden, params["W_ff2"]) + params["b_ff2"]
    ff_residual = attention_norm + ff_output

    mean_ffn = cp.mean(ff_residual, axis=-1, keepdims=True)
    centered_ffn = ff_residual - mean_ffn
    var_ffn = cp.mean(centered_ffn ** 2, axis=-1, keepdims=True)
    inv_std_ffn = cp.float32(1.0) / cp.sqrt(var_ffn + ln_eps)
    normalized_ffn = centered_ffn * inv_std_ffn
    encoder_output = normalized_ffn * params["gamma_ffn"] + params["beta_ffn"]

    # 6. 均值池化、分类概率与预测类别
    image_features = cp.mean(encoder_output, axis=0)
    logits = cp.dot(image_features, params["W_cls"]) + params["b_cls"]
    shifted_logits = logits - cp.max(logits)
    exp_logits = cp.exp(shifted_logits)
    probabilities = exp_logits / cp.sum(exp_logits)
    prediction = int(cp.argmax(probabilities))

    return prediction, probabilities


# x_test 已归一化；预测函数不需要真实标签。
inference_sample_index = 0
inference_prediction, inference_probabilities = predict_one_image(
    x_test[inference_sample_index], loaded_params,
    loaded_position_encoding, loaded_num_heads, loaded_ln_eps
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

解释器 F:\PythonProjects\deep_learning\.venv\Scripts\python.exe，Python 3.13.15、NumPy 2.5.3、CuPy 14.2.0；GPU0 NVIDIA GeForce RTX 5060，可见GPU数1。精确聊天片段实际执行一次新预测，输入原测试第0图；该片段不读取标签，不计算loss，不保留供反向传播的cache，不调用backward或SGD。

精确片段输出：

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

后台对照另外调用新预测两次，输入测试索引1及999；索引0复用教学演示的返回结果，没有重复调用。每张图再以其真实标签调用已验证旧forward一次，仅用于后台参考CE及概率。总计新预测3次、参考forward3次、完整前向计算6次、反向0次、训练函数0次、参数更新0次。未进行全集评估或新的有限差分。新预测签名、函数体及教学片段静态检查均没有target、loss、cache、backward或y_test引用；新函数到probabilities的41条语句AST与已验证原forward完全相等。注释差异不参与AST比较。

测试索引0：预测7，后台参考标签7，概率和1.0000001192092896，10类概率与原forward逐位相等，最大差0.0。
测试索引1：预测2，后台参考标签2，概率和1.0000001192092896，10类概率与原forward逐位相等，最大差0.0。
测试索引999：预测9，后台参考标签9，概率和1.0，10类概率与原forward逐位相等，最大差0.0。

三图返回类别都是Python int，概率shape(10,)、GPU0、float32、有限非负、范围0～1，概率和在float32容差内为1；argmax与返回数字一致。测试第0图预测7，该类概率0.9943311214447021。所有20个参数、固定位置编码、输入图片及后台标签的内容哈希、对象身份、存储地址都不变；loaded_params字典对象和键顺序保持，配置值不变。原主文件、原存档及所有相关旧记录保持字节一致。

核验期间禁止随机生成、seed和随机状态公共API，调用数0，结束恢复原函数；没有获取或改变随机数状态。新预测及原参考函数的实际调用通过独立profiling观察，观察器不改动函数返回或教学片段，执行后恢复原profile。

v048训练集准确率89.24%、测试集准确率87.50%是历史固定评估结果，本步没有重算，不由三张图的预测得出准确率。JSON保存原文件/数据/旧记录哈希、存档26成员信息、20参数GPU身份及哈希、精确代码/函数/环境启动源码和SHA256、AST比较结果、3图完整概率及参考CE、实际调用计数、不可变性检查、环境和stdout。[本步代码](#本步代码) · [预测一致性实际证据](2026-10-08_transformer_prediction_validation_v051.json) · [实际用户存档字节副本](checkpoints/transformer_mnist_20261008_115104_418995.npz)。相对链接已核验，无新增图。

## 与前版变化、未解限制及下一步

v050教学读取已保存参数到GPU，并比较一张图的加载前后loss和概率。v051新增无标签predict_one_image，使用已经加载的参数返回类别和10类概率；网络数学、预处理、参数值、训练过程均未改变。保留所有旧版本，项目索引追加v051。同步仅复制已经生成的Markdown、JSON和原存档字节附件，不运行模型。

核验进程独立于用户终端内存，当前loaded_params从实际用户文件重新读取且完整对照v048参数快照；不能直接读取或假称修改用户终端对象。本步用户仍需此前NumPy/CuPy启动环境、loaded_params/配置和x_test，代码没有变成独立部署程序。predict函数没有通用输入检查，使用者需保证28×28 GPU float32和相同归一化规则。

本次只测试3张原测试图片的推理等价性，不能覆盖任意图像、保证泛化或证明概率已校准。原模型仍只在MNIST前5000张训练，历史测试指标只覆盖前1000张；没有声称完整MNIST基准、收敛或最优。新手写数字图片还需处理尺寸、前景/背景、灰度范围和方向等实际输入差异，不能仅将任意图片直接传入。

下一步可查看预测错误的已加载测试样本，区分分类概率、真实标签及模型错误；若后续使用外部原始图片，另行教学与训练一致的预处理。
