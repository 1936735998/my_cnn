# MNIST 手写 Transformer 建模记录

日期：2026-10-07（Asia/Shanghai）；版本：v044。
阶段：完整手写 GPU 反向函数整合，使用更新后新缓存计算新梯度。本轮真实执行与核验，不是仅文档更新；无新增参数更新或数据集训练。

## 问题范围、输入来源与清洗（事实）

用户报告更新后首图损失1.729211688041687、预测5、真实标签5，新概率与缓存仍在GPU0。本版将此前逐步完成的反向整理为 backward_one_image(params, cache)，返回20组参数梯度和归一化像素梯度；仍贴代码到聊天，不改用户主文件，不交付模型 Python 主文件。

独立进程读取执行 `F:\PythonProjects\deep_learning\transformer_mnist.py` 已保存前四十三步，包括此前一次单图SGD更新和更新后的新前向缓存，然后执行本片段。主文件SHA256 `423e63d71819816ee647cf5ea6a83ac61ae2d3c293f1d4c18d049fd6c0f6fb37`，前后不变；未操作用户终端实时内存。工作区fragment.txt仅供核验，精确代码保留在下方及JSON，不向用户交付主文件。

数据沿用本地MNIST `C:\Users\19367\.keras\datasets\mnist.npz`，SHA256 `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`。训练前5000张、测试前1000张，像素CuPy float32除以255、标签int64。本轮完整差分使用首张训练图，标签5；另用训练索引1、标签0核实前向和反向函数可复用。无新下载、清洗、划分、增强、洗牌或测试集评估。

## 假设、变量与公式

延续28个图片行位置、每行28像素、模型宽度32、四头每头8维、ReLU前馈宽度64、单层post-LN编码器、固定正弦/余弦位置编码、均值池化及十类单图交叉熵。反向必须使用与cache同一次前向的params；函数仅读取参数与缓存，不重新初始化或更新参数。

```text
d_logits = probabilities.copy(); d_logits[target] -= 1
dW_cls = outer(image_features, d_logits); db_cls = d_logits
d_image_features = W_cls @ d_logits
d_encoder_output = broadcast(d_image_features / seq_len)
LN: H=G*gamma; dX=inv_std*(H-mean(H)-x_hat*mean(H*x_hat))
    dgamma=sum(G*x_hat, axis=0); dbeta=sum(G, axis=0)
FFN: 残差分流 → FF2 → ReLU(正数掩码) → FF1 → 路径汇合
注意力: 残差分流 → 输出投影 → 拆拼头逆变换
P@V: dP=dH@V^T; dV=P^T@dH
softmax: dS=P*(dP-sum(dP*P, axis=-1, keepdims=True))
QK: dR=dS/sqrt(head_dim); dQ=dR@K; dK=dR^T@Q
QKV投影: dW=encoder_input.T@dQ; db=sum(dQ, axis=0)
d_encoder_input=残差直连+dQ@W_Q.T+dK@W_K.T+dV@W_V.T
d_embedding=d_encoder_input（固定位置编码不更新）
dW_in=X.T@d_embedding; db_in=sum(d_embedding, axis=0)
d_X=d_embedding@W_in.T
return grads_new: 20参数键的字典, d_X_new: (28,28)
```

LayerNorm均值沿最后特征轴，epsilon包含在inv_std中；两层gamma/beta彼此独立。除均值池化外没有额外平均。ReLU零点沿用导数0的约定，本轮基础点均非零；数值核验逐次检查扰动前后的实际掩码，而非强行固定前向分支。

## 方法选择理由（判断）

函数复用之前已经推导的基础CuPy链式规则，不调用自动求导或现成神经网络层。保留变量名称和反向顺序，便于对应每一教学步骤。前版已整理前向与新缓存，此时完成反向函数后才能用新梯度组织重复训练。新梯度另命名grads_new，旧grads保留，避免把更新前梯度误用于更新后参数。

完整函数整合可能产生缓存、头轴、残差路径或参数配对错误，因此本轮对全部20组9802个参数分量和784像素分量，用独立NumPy float64完整模型交叉熵中心差分核验；这是该固定单图、当前参数状态的检查，不是训练收敛或泛化证明。

## 本步代码

```python
# ==================================================
# 训练准备：整理完整反向函数
# ==================================================

def backward_one_image(params, cache):
    X = cache["X"]
    seq_len, d_model = cache["encoder_input"].shape
    num_heads = cache["num_heads"]
    head_dim = cache["head_dim"]
    grads = {}

    # 1. 分类损失、分类层与均值池化
    d_logits = cache["probabilities"].copy()
    d_logits[cache["target"]] -= cp.float32(1.0)
    grads["W_cls"] = cp.outer(cache["image_features"], d_logits)
    grads["b_cls"] = d_logits.copy()
    d_image_features = cp.dot(params["W_cls"], d_logits)
    d_encoder_output = cp.broadcast_to(
        d_image_features / cp.float32(seq_len), (seq_len, d_model)
    ).copy()

    # 2. 第二次 LayerNorm
    normalized_ffn = cache["normalized_ffn"]
    grads["gamma_ffn"] = cp.sum(d_encoder_output * normalized_ffn, axis=0)
    grads["beta_ffn"] = cp.sum(d_encoder_output, axis=0)
    d_normalized_ffn = d_encoder_output * params["gamma_ffn"]
    d_ff_residual = cache["inv_std_ffn"] * (
        d_normalized_ffn
        - cp.mean(d_normalized_ffn, axis=-1, keepdims=True)
        - normalized_ffn * cp.mean(
            d_normalized_ffn * normalized_ffn, axis=-1, keepdims=True
        )
    )

    # 3. 前馈残差、第二个线性层、ReLU、第一个线性层
    d_attention_norm_skip = d_ff_residual.copy()
    d_ff_output = d_ff_residual.copy()
    grads["W_ff2"] = cp.dot(cache["ff_hidden"].T, d_ff_output)
    grads["b_ff2"] = cp.sum(d_ff_output, axis=0)
    d_ff_hidden = cp.dot(d_ff_output, params["W_ff2"].T)
    d_ff_hidden_linear = d_ff_hidden * (
        cache["ff_hidden_linear"] > cp.float32(0.0)
    )
    grads["W_ff1"] = cp.dot(cache["attention_norm"].T, d_ff_hidden_linear)
    grads["b_ff1"] = cp.sum(d_ff_hidden_linear, axis=0)
    d_attention_norm_from_ffn = cp.dot(d_ff_hidden_linear, params["W_ff1"].T)
    d_attention_norm = d_attention_norm_skip + d_attention_norm_from_ffn

    # 4. 第一次 LayerNorm
    normalized_attn = cache["normalized_attn"]
    grads["gamma_attn"] = cp.sum(d_attention_norm * normalized_attn, axis=0)
    grads["beta_attn"] = cp.sum(d_attention_norm, axis=0)
    d_normalized_attn = d_attention_norm * params["gamma_attn"]
    d_attention_residual = cache["inv_std_attn"] * (
        d_normalized_attn
        - cp.mean(d_normalized_attn, axis=-1, keepdims=True)
        - normalized_attn * cp.mean(
            d_normalized_attn * normalized_attn, axis=-1, keepdims=True
        )
    )

    # 5. 注意力残差、输出投影与拼头操作
    d_encoder_input_skip = d_attention_residual.copy()
    d_attention_output = d_attention_residual.copy()
    grads["W_O"] = cp.dot(cache["concatenated_heads"].T, d_attention_output)
    grads["b_O"] = cp.sum(d_attention_output, axis=0)
    d_concatenated_heads = cp.dot(d_attention_output, params["W_O"].T)
    d_head_output = d_concatenated_heads.reshape(
        seq_len, num_heads, head_dim
    ).transpose(1, 0, 2).copy()

    # 6. 注意力加权汇总与 softmax
    attention_weights = cache["attention_weights"]
    Q_heads = cache["Q_heads"]
    K_heads = cache["K_heads"]
    V_heads = cache["V_heads"]
    d_attention_weights = cp.matmul(d_head_output, V_heads.transpose(0, 2, 1))
    d_V_heads = cp.matmul(attention_weights.transpose(0, 2, 1), d_head_output)
    weighted_gradient_sum = cp.sum(
        d_attention_weights * attention_weights, axis=-1, keepdims=True
    )
    d_attention_scores = attention_weights * (
        d_attention_weights - weighted_gradient_sum
    )

    # 7. 缩放 QK 分数与 Q、K、V 拆头操作
    d_qk_product = d_attention_scores / cp.sqrt(cp.float32(head_dim))
    d_Q_heads = cp.matmul(d_qk_product, K_heads)
    d_K_heads = cp.matmul(d_qk_product.transpose(0, 2, 1), Q_heads)
    d_Q = d_Q_heads.transpose(1, 0, 2).reshape(seq_len, d_model).copy()
    d_K = d_K_heads.transpose(1, 0, 2).reshape(seq_len, d_model).copy()
    d_V = d_V_heads.transpose(1, 0, 2).reshape(seq_len, d_model).copy()

    # 8. Q、K、V 投影与四条输入路径汇合
    encoder_input = cache["encoder_input"]
    grads["W_Q"] = cp.dot(encoder_input.T, d_Q)
    grads["b_Q"] = cp.sum(d_Q, axis=0)
    grads["W_K"] = cp.dot(encoder_input.T, d_K)
    grads["b_K"] = cp.sum(d_K, axis=0)
    grads["W_V"] = cp.dot(encoder_input.T, d_V)
    grads["b_V"] = cp.sum(d_V, axis=0)
    d_encoder_input = (
        d_encoder_input_skip
        + cp.dot(d_Q, params["W_Q"].T)
        + cp.dot(d_K, params["W_K"].T)
        + cp.dot(d_V, params["W_V"].T)
    )

    # 9. 固定位置编码相加与输入映射
    d_embedding = d_encoder_input.copy()
    grads["W_in"] = cp.dot(X.T, d_embedding)
    grads["b_in"] = cp.sum(d_embedding, axis=0)
    d_X = cp.dot(d_embedding, params["W_in"].T)

    return grads, d_X


# 使用更新后参数对应的新缓存，计算这一轮的新梯度
grads_new, d_X_new = backward_one_image(params, cache_after_update)

print("新梯度数组数量:", len(grads_new))
print("新参数梯度标量总数:", sum(gradient.size for gradient in grads_new.values()))
print("W_in 的新梯度:", grads_new["W_in"].shape)
print("W_Q 的新梯度:", grads_new["W_Q"].shape)
print("gamma_ffn 的新梯度:", grads_new["gamma_ffn"].shape)
print("d_X_new:", d_X_new.shape)
print("W_in 新梯度的实际设备:", grads_new["W_in"].device)
print("d_X_new 的实际设备:", d_X_new.device)
```

## 配套前向函数来源

以下使用inspect.getsource从本轮实际执行的主文件提取，来源及SHA256同时保留在JSON；没有改写用户主文件。

```python
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
```

## 实际执行、验证与结果（事实）

解释器 `F:\PythonProjects\deep_learning\.venv\Scripts\python.exe`，Python 3.13.15、NumPy 2.5.3、CuPy 14.2.0。可见GPU1张，GPU0为 NVIDIA GeForce RTX 5060；实际GPU执行并同步CUDA，未调整依赖。

新字典20个键与PARAM_NAMES精确一致，全部梯度形状与参数匹配、float32、GPU0且有限；d_X_new为float32 GPU0 `(28,28)`。所有20参数、旧grads、新cache_after_update数组与元数据、固定位置编码及抽查的输入/旧loss/概率/像素梯度保持不变。此次新增参数更新0；独立进程只是重放主文件已有一次SGD。

v039独立CPU参考从归一化像素完整重算损失，采用显式头切块、einsum、logaddexp及双精度。基础CPU损失 `1.72921171482`，GPU损失 `1.72921168804`，绝对差 `2.677673061e-08`，rtol=2e-6、atol=2e-6下通过；基础CPU/GPU ReLU掩码一致。

全部10586个中心差分分量通过rtol=2e-5、atol=2e-7检查，初始步长1e-5，如跨ReLU折点则最多折半20次。实际扰动最终均保持掩码，折半总计0次，整体最大绝对误差 `1.142849937e-07`；其中9802参数分量最大绝对误差 `1.142849937e-07`，784像素分量最大绝对误差 `1.826357732e-09`。逐组结果：

- W_in：896 分量，最大绝对误差 `8.733920254e-09`，步长 `1e-05`～`1e-05`，折半合计 0 次。
- b_in：32 分量，最大绝对误差 `1.684636076e-08`，步长 `1e-05`～`1e-05`，折半合计 0 次。
- W_Q：1024 分量，最大绝对误差 `6.288090049e-10`，步长 `1e-05`～`1e-05`，折半合计 0 次。
- b_Q：32 分量，最大绝对误差 `3.623661373e-10`，步长 `1e-05`～`1e-05`，折半合计 0 次。
- W_K：1024 分量，最大绝对误差 `7.17488291e-10`，步长 `1e-05`～`1e-05`，折半合计 0 次。
- b_K：32 分量，最大绝对误差 `3.274180926e-10`，步长 `1e-05`～`1e-05`，折半合计 0 次。
- W_V：1024 分量，最大绝对误差 `2.227551477e-08`，步长 `1e-05`～`1e-05`，折半合计 0 次。
- b_V：32 分量，最大绝对误差 `2.084183137e-08`，步长 `1e-05`～`1e-05`，折半合计 0 次。
- W_O：1024 分量，最大绝对误差 `2.179321257e-08`，步长 `1e-05`～`1e-05`，折半合计 0 次。
- b_O：32 分量，最大绝对误差 `2.034488486e-08`，步长 `1e-05`～`1e-05`，折半合计 0 次。
- gamma_attn：32 分量，最大绝对误差 `1.118016257e-08`，步长 `1e-05`～`1e-05`，折半合计 0 次。
- beta_attn：32 分量，最大绝对误差 `1.904283664e-08`，步长 `1e-05`～`1e-05`，折半合计 0 次。
- W_ff1：2048 分量，最大绝对误差 `1.546642636e-08`，步长 `1e-05`～`1e-05`，折半合计 0 次。
- b_ff1：64 分量，最大绝对误差 `5.257703613e-09`，步长 `1e-05`～`1e-05`，折半合计 0 次。
- W_ff2：2048 分量，最大绝对误差 `2.083317695e-08`，步长 `1e-05`～`1e-05`，折半合计 0 次。
- b_ff2：32 分量，最大绝对误差 `1.479967437e-08`，步长 `1e-05`～`1e-05`，折半合计 0 次。
- gamma_ffn：32 分量，最大绝对误差 `9.866312226e-09`，步长 `1e-05`～`1e-05`，折半合计 0 次。
- beta_ffn：32 分量，最大绝对误差 `9.370481252e-09`，步长 `1e-05`～`1e-05`，折半合计 0 次。
- W_cls：320 分量，最大绝对误差 `1.142849937e-07`，步长 `1e-05`～`1e-05`，折半合计 0 次。
- b_cls：10 分量，最大绝对误差 `2.329527515e-08`，步长 `1e-05`～`1e-05`，折半合计 0 次。
- X：784 分量，最大绝对误差 `1.826357732e-09`，步长 `1e-05`～`1e-05`，折半合计 0 次。

参考L2范数不超过1e-12时相对L2误差记为null，依靠绝对误差判据；没有把近零参考误记为相对误差0。每个分量的实际GPU梯度、CPU差分梯度、实际步长与折半次数均保存在JSON。

另一张训练图前向损失 `2.9217672348`，前向/反向均可复用，新梯度形状、类型、设备、有限性通过，参数及其缓存未改变。本样本没有重复完整差分，未据此报告数据集准确率。

实际片段输出：

```text
新梯度数组数量: 20
新参数梯度标量总数: 9802
W_in 的新梯度: (28, 32)
W_Q 的新梯度: (32, 32)
gamma_ffn 的新梯度: (32,)
d_X_new: (28, 28)
W_in 新梯度的实际设备: <CUDA Device 0>
d_X_new 的实际设备: <CUDA Device 0>
```

[本步代码](#本步代码) · [配套前向函数来源](#配套前向函数来源) · [实际GPU、独立参考完整源码和逐分量证据](2026-10-07_transformer_backward_function_validation_v044.json)。相对链接已核验，本步无新增图。

## 与前版变化、限制及下一步

v043整理完整前向函数并生成更新后新缓存；本版新增完整反向函数，读取该缓存，返回20组新参数梯度和新像素梯度。新参数0，20数组仍9802标量；无第二次SGD、完整训练循环、测试评估或检查点保存。主模型和历史记录保留，追加项目记录索引；同步仅复制本版记录与证据，不重跑模型。

当前验证限于首图与更新一次后的参数状态，另一训练图仅函数复用检查。不能从单图损失下降或全分量梯度通过推断训练集/测试集准确率。缓存与params必须配对且在反向完成前保持不变；后续更新应使用grads_new而不是旧grads。下一步用前向/反向函数及手写更新循环逐步组织GPU训练，再记录实际训练损失、测试准确率与局限。
