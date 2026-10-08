# MNIST 手写 Transformer 建模记录

日期：2026-10-07（Asia/Shanghai）；版本：v043。
阶段：完整 GPU 前向函数整合，计算更新后的新损失、概率和缓存。本轮真实执行，不是仅文档更新；没有新增参数更新或数据集训练。

## 问题范围、输入来源与清洗（事实）

用户报告上一轮首次 SGD 更新20个数组、9802个参数、学习率0.01、更新前loss为1.9714317321777344，参数仍GPU0。本版将先前已学过的完整前向整合为 forward_one_image，读取更新后的参数并返回loss、probabilities及下一轮反向所需cache；仍贴代码到聊天，不改用户主文件，不交付完整 Python 模型文件。

独立进程读取执行 `F:\PythonProjects\deep_learning\transformer_mnist.py` 已保存前四十二步，包括一次原地SGD更新，然后执行本片段。主文件SHA256 `a264471540c2758482798b896e9b7f52938f33a577a9424cb2c0bfad4cd091c2`，执行前后未变；未操作用户终端实时内存，未保存模型检查点。

沿用本地MNIST `C:\Users\19367\.keras\datasets\mnist.npz`，SHA256 `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`，训练前5000、测试前1000，像素CuPy float32除以255，标签int64。本步展示首张训练图，标签5；另用训练索引1、标签0验证函数接收不同输入与标签。无新清洗、下载、划分、增强、洗牌或测试集评估。

## 假设、变量与公式

延续28行作为序列、每行28像素、32维模型、4头每头8维、64维ReLU前馈、单层post-LN编码器、固定正弦/余弦位置编码、位置均值池化、十类单图交叉熵。前向仅读取当前params，函数不重新初始化参数、不改变随机种子、不更新参数。

```text
X → X@W_in+b_in → +位置编码 → QKV与拆头
  → softmax(QK^T/sqrt(8))V → 拼头及输出投影
  → 残差与LN1 → FFN → 残差与LN2
  → mean(axis=0) → 分类logits → 概率与交叉熵
return loss: (), probabilities: (10,), cache: dict
```

交叉熵为log(sum(exp(logits-max(logits))))-(logits-max(logits))[target]。两次LayerNorm分别使用独立gamma/beta与epsilon。cache保存本次前向的必要数组及target/num_heads/head_dim，引用会保留数组生命周期，不额外复制所有中间值。对应反向完成前不能改变参数、输入及缓存。概率反向需先copy再减1，避免污染缓存。

## 方法选择理由（判断）

训练要对每张图片重复前向，函数化减少重复拼接代码，同时保留此前逐步推导的变量名与计算次序。此时一并返回新缓存，下一步可以在该缓存上组织手写反向，避免再次重写整个前向。新变量名明确区分更新前全局缓存与更新后局部缓存。无新网络层、自动求导或框架封装。

## 本步代码

```python
# ==================================================
# 训练准备：整理完整前向函数
# ==================================================

def forward_one_image(X, target, params, position_encoding, num_heads, ln_eps):
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

    # 3. 缩放分数、softmax 与加权汇总
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

    # 6. 均值池化、分类概率与交叉熵
    image_features = cp.mean(encoder_output, axis=0)
    logits = cp.dot(image_features, params["W_cls"]) + params["b_cls"]
    shifted_logits = logits - cp.max(logits)
    exp_logits = cp.exp(shifted_logits)
    probabilities = exp_logits / cp.sum(exp_logits)
    loss = cp.log(cp.sum(exp_logits)) - shifted_logits[target]

    # 留下这一轮反向传播需要的中间结果
    cache = {
        "X": X, "encoder_input": encoder_input,
        "Q_heads": Q_heads, "K_heads": K_heads, "V_heads": V_heads,
        "attention_weights": attention_weights,
        "concatenated_heads": concatenated_heads,
        "inv_std_attn": inv_std_attn, "normalized_attn": normalized_attn,
        "attention_norm": attention_norm,
        "ff_hidden_linear": ff_hidden_linear, "ff_hidden": ff_hidden,
        "inv_std_ffn": inv_std_ffn, "normalized_ffn": normalized_ffn,
        "encoder_output": encoder_output, "image_features": image_features,
        "probabilities": probabilities, "target": target,
        "num_heads": num_heads, "head_dim": head_dim,
    }
    return loss, probabilities, cache


# params 已经更新过；这里读取更新后的参数重新前向
loss_after_update, probabilities_after_update, cache_after_update = forward_one_image(
    X, target, params, position_encoding, num_heads, ln_eps
)

print("更新前的损失:", float(loss_before_update))
print("更新后的损失:", float(loss_after_update))
print("损失减少量:", float(loss_before_update - loss_after_update))
print("真实标签:", target)
print("更新后的预测:", int(cp.argmax(probabilities_after_update)))
print("probabilities_after_update:", probabilities_after_update.shape)
print("loss_after_update 的实际设备:", loss_after_update.device)
print("新缓存 encoder_input 的实际设备:", cache_after_update["encoder_input"].device)
```

## 实际执行、验证与结果（事实）

解释器 `F:\PythonProjects\deep_learning\.venv\Scripts\python.exe`，Python 3.13.15、NumPy 2.5.3、CuPy 14.2.0。可见GPU1张，GPU0为 NVIDIA GeForce RTX 5060。实际执行并同步CUDA，未改依赖。

loss_after_update为GPU0 float32标量，概率为GPU0 float32 `(10,)`，17个缓存数组形状逐一核验，全部float32、GPU0且有限。所有参数、旧20组梯度、固定位置编码与抽查的旧前向缓存没有改变。本片段新增参数更新次数0。

首图loss从更新前 `1.97143173218` 变为 `1.72921168804`，减少 `0.242220044136`。新预测5，真实标签5；新loss/概率与v042独立GPU核验一致。用实际当前参数在v039独立NumPy float64完整前向重新计算，loss为 `1.72921171482`，CPU/GPU绝对差 `2.677673061e-08`，rtol=2e-6、atol=2e-6下通过。FFN预激活也与独立参考相符，最大绝对误差 `3.044154038e-07`。

训练索引1的另一张图前向loss为 `2.9217672348`，CPU参考为 `2.92176743741`，绝对差 `2.026028922e-07`。只用于核实函数可复用，并没有据此报告数据集准确率。

实际输出：

```text
更新前的损失: 1.9714317321777344
更新后的损失: 1.729211688041687
损失减少量: 0.24222004413604736
真实标签: 5
更新后的预测: 5
probabilities_after_update: (10,)
loss_after_update 的实际设备: <CUDA Device 0>
新缓存 encoder_input 的实际设备: <CUDA Device 0>
```

[本步代码](#本步代码) · [实际GPU、完整CPU参考源码与结果](2026-10-07_transformer_forward_function_validation_v043.json)。相对链接已核验，本步无新增图。

## 与前版变化、限制及下一步

v042首次单图SGD；本版新增可复用完整前向函数与新缓存，20个参数数组仍为9802标量，不再额外更新参数。保留历史记录，更新项目记录索引；同步只复制文档证据，不重跑模型。

已观察当前训练图损失下降，但不能推断完整训练集或测试集表现。新缓存尚未用于新的反向，旧全局grads仍对应更新前参数；此时不能直接重复上一段SGD循环。下一步把已完成的手写反向整理为读取新cache和params的函数，重新生成20组梯度，再组织训练循环。完整数据集训练、批量化、收敛及泛化评估尚未执行。
