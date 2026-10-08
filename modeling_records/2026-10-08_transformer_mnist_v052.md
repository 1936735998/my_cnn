# MNIST 手写 Transformer 建模记录

日期：2026-10-08（Asia/Shanghai）；版本：v052。
阶段：独立文件加载真实训练存档并在GPU验证1000张测试图。本步真实执行完整测试评估，没有训练；不是仅文档更新。

## 问题范围、输入来源与处理（事实）

用户不希望每次重新执行训练，准备单独新建文件加载训练结果验证。本版提供一个完整独立脚本：初始化已有CUDA运行环境，读取存档及配置，只读取原前1000测试图片，手写Transformer前向和稳定CE，固定参数评估并打印前5图预测。脚本不依赖此前交互变量，不导入或执行原训练主文件，不读取训练数组，没有反向传播、SGD或随机初始化。用户自行新建文件粘贴；核验只执行workspace中的精确代码副本，没有编辑用户主文件或生成项目根的新验证文件。

实际存档：F:\PythonProjects\deep_learning\outputs\checkpoints\transformer_mnist_20261008_115104_418995.npz。文件50006 bytes，SHA256 349eff0650c37163be783b58d4f39d256b796091884c624428f131d2a320f218；allow_pickle=False读取26个非object成员，其中20参数数组共9802标量，位置编码28×32及5项配置。全部20参数float32内容哈希、shape与v048训练后快照一致，固定位置编码哈希相同。配置num_heads=4、ln_eps=float32(1e-5)、pixel_divisor=255.0、format_version=1及成员dtype通过核对。原文件保持字节不变；记录附件为逐字节复制，未重新序列化存档。

原数据：C:\Users\19367\.keras\datasets\mnist.npz，SHA256 731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1。只读取x_test前1000图和y_test前1000标签；图像转GPU float32后依配置除255一次，标签CPU np.int64。数据数组哈希与v048相同；没有新增清洗、增强、划分或训练集读取。这里所谓测试集范围仅原前1000图，不是MNIST全部10000测试图。

原主文件：F:\PythonProjects\deep_learning\transformer_mnist.py，SHA256 7177a3632cf49ab8829c8744aa13d6ca1f65cbb3ee8a3cf7de4da42c37170aec。本次只读其文件哈希用于保护，连AST函数也没有抽取执行。所有v048～v051相关旧Markdown及JSON在核验前后哈希不变。本次推理参数来自存档，绝不通过执行旧主文件重现训练结果。

## 假设、变量与公式

结构保持28行token、每行28输入特征、d_model32、4头各8维、前馈64、单post-LN编码器、固定位置编码、均值池化、10分类。输入X为已经归一化的GPU float32 28×28灰度数组。图像朝向和行顺序与原MNIST一致；文件只处理此既定输入结构。标签仅用于外部评估损失与准确率，infer_one_image函数不接收target，也不返回反向缓存。

~~~text
E = X @ W_in + b_in + position_encoding
A_h = softmax(Q_h @ K_h.T / sqrt(8))
Z = concat(A_h @ V_h) @ W_O + b_O
N = LayerNorm(E + Z)
F = ReLU(N @ W_ff1 + b_ff1) @ W_ff2 + b_ff2
encoder_output = LayerNorm(N + F)
logits = mean(encoder_output, axis=0) @ W_cls + b_cls
s = logits - max(logits)
P = exp(s) / sum(exp(s))
L = log(sum(exp(s))) - s[target]
mean_loss = GPU float32逐图累计L / float32(1000)
accuracy = GPU int64正确计数转float32 / float32(1000)
~~~

LayerNorm每行32特征上计算均值和中心化方差，加存档ln_eps后归一化，再乘gamma加beta。两个LayerNorm参数独立。注意力行softmax通过减最大值稳定计算，分类CE直接使用logits，避免先计算低概率再log。网络所有参数与特征数组、损失及准确率标量位于GPU0；标签与循环索引在CPU，打印时将少量指标同步为Python值。

## 方法选择理由（判断）

独立验证需要实际保存的权重与必要配置，不能import包含顶层训练循环的旧文件，否则会重训。本版将前向所需数学完整放入新文件；层归一化提为小函数，保留原算子顺序。CUDA启动只通过已有torch包位置找到CUDA DLL，不导入torch网络或使用自动求导。用户继续使用现有项目.venv解释器，无新依赖或重新安装环境。

只固定评估测试图；不重新计算训练集，不随机选图，不改变参数，不使用测试集调学习率或选择轮次。每图Python循环保持目前教学结构，属于功能验证，不是GPU吞吐性能基准。标签用于准确率与监督损失是验证所需信息；实际无标签推理可单独调用infer_one_image并argmax概率。

## 完整独立代码

[可独立运行的源码](transformer_validation_v052.py)；以下为精确实际执行代码，UTF-8源码SHA256 a212945fa470d6570dae0bdab9821cc9c66402ff9e2a0d6eb4f0215fa38b0315，原文件字节SHA256 a212945fa470d6570dae0bdab9821cc9c66402ff9e2a0d6eb4f0215fa38b0315。

~~~python
import ctypes
import os
from importlib.util import find_spec
from pathlib import Path

import numpy as np


# 1. GPU 环境：复用当前项目中已有的 CUDA DLL。
torch_spec = find_spec("torch")
if torch_spec is None or torch_spec.origin is None:
    raise RuntimeError("请使用 deep_learning 项目的 .venv Python 解释器运行")

cuda_library_path = Path(torch_spec.origin).parent / "lib"
cuda_dll_handle = os.add_dll_directory(str(cuda_library_path))
cuda_library_handles = [
    ctypes.WinDLL(str(cuda_library_path / name))
    for name in (
        "nvrtc64_130_0.dll",
        "cublasLt64_13.dll",
        "cublas64_13.dll",
        "curand64_10.dll",
    )
]

import cupy as cp

cp.cuda.Device(0).use()


# 2. 读取之前训练好的参数，不依赖训练脚本中的变量。
checkpoint_path = Path(
    "F:/PythonProjects/deep_learning/outputs/checkpoints/"
    "transformer_mnist_20261008_115104_418995.npz"
)

with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
    if int(checkpoint["format_version"].item()) != 1:
        raise ValueError("当前代码支持第 1 版参数文件")

    parameter_names = checkpoint["parameter_names"].tolist()
    params = {
        name: cp.asarray(checkpoint[name], dtype=cp.float32)
        for name in parameter_names
    }
    position_encoding = cp.asarray(
        checkpoint["position_encoding"], dtype=cp.float32
    )
    num_heads = int(checkpoint["num_heads"].item())
    ln_eps = cp.float32(checkpoint["ln_eps"].item())
    pixel_divisor = float(checkpoint["pixel_divisor"].item())


# 3. 只读取测试数据，并按训练时的规则归一化一次。
data_path = Path.home() / ".keras" / "datasets" / "mnist.npz"
test_limit = 1000

with np.load(data_path, allow_pickle=False) as data:
    x_test = cp.asarray(
        data["x_test"][:test_limit], dtype=cp.float32
    ) / pixel_divisor
    test_labels = np.asarray(data["y_test"][:test_limit], dtype=np.int64)


# 4. 手写 LayerNorm，保持之前的计算顺序。
def layer_norm(values, gamma, beta, eps):
    mean = cp.mean(values, axis=-1, keepdims=True)
    centered = values - mean
    variance = cp.mean(centered ** 2, axis=-1, keepdims=True)
    inv_std = cp.float32(1.0) / cp.sqrt(variance + eps)
    normalized = centered * inv_std
    return normalized * gamma + beta


# 5. 手写 Transformer 前向：返回 logits 和概率，不需要真实标签。
def infer_one_image(X, params, position_encoding, num_heads, ln_eps):
    seq_len = X.shape[0]
    d_model = params["W_in"].shape[1]
    head_dim = d_model // num_heads

    embedding = cp.dot(X, params["W_in"]) + params["b_in"]
    encoder_input = embedding + position_encoding

    Q = cp.dot(encoder_input, params["W_Q"]) + params["b_Q"]
    K = cp.dot(encoder_input, params["W_K"]) + params["b_K"]
    V = cp.dot(encoder_input, params["W_V"]) + params["b_V"]
    Q_heads = Q.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)
    K_heads = K.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)
    V_heads = V.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)

    scores = cp.matmul(Q_heads, K_heads.transpose(0, 2, 1))
    scores = scores / cp.sqrt(cp.float32(head_dim))
    shifted_scores = scores - cp.max(scores, axis=-1, keepdims=True)
    exp_scores = cp.exp(shifted_scores)
    weights = exp_scores / cp.sum(exp_scores, axis=-1, keepdims=True)

    head_output = cp.matmul(weights, V_heads)
    concatenated = head_output.transpose(1, 0, 2).reshape(seq_len, d_model)
    attention_output = cp.dot(concatenated, params["W_O"]) + params["b_O"]

    attention_norm = layer_norm(
        encoder_input + attention_output,
        params["gamma_attn"], params["beta_attn"], ln_eps
    )
    hidden_linear = cp.dot(attention_norm, params["W_ff1"]) + params["b_ff1"]
    hidden = cp.maximum(hidden_linear, cp.float32(0.0))
    ff_output = cp.dot(hidden, params["W_ff2"]) + params["b_ff2"]
    encoder_output = layer_norm(
        attention_norm + ff_output,
        params["gamma_ffn"], params["beta_ffn"], ln_eps
    )

    image_features = cp.mean(encoder_output, axis=0)
    logits = cp.dot(image_features, params["W_cls"]) + params["b_cls"]
    shifted_logits = logits - cp.max(logits)
    exp_logits = cp.exp(shifted_logits)
    probabilities = exp_logits / cp.sum(exp_logits)
    return logits, probabilities


# 6. 固定参数验证：标签仅用于计算损失和判断预测是否正确。
num_test = x_test.shape[0]
loss_sum = cp.zeros((), dtype=cp.float32)
correct_sum = cp.zeros((), dtype=cp.int64)
preview_results = []

for index in range(num_test):
    target = int(test_labels[index])
    logits, probabilities = infer_one_image(
        x_test[index], params, position_encoding, num_heads, ln_eps
    )

    # 与之前训练/评估使用相同的稳定交叉熵公式。
    shifted_logits = logits - cp.max(logits)
    loss = cp.log(cp.sum(cp.exp(shifted_logits))) - shifted_logits[target]
    prediction = cp.argmax(probabilities)

    loss_sum += loss
    correct_sum += (prediction == target)

    if index < 5:
        predicted_digit = int(prediction)
        preview_results.append((
            index, target, predicted_digit, float(probabilities[predicted_digit])
        ))

mean_loss = loss_sum / cp.float32(num_test)
accuracy = correct_sum.astype(cp.float32) / cp.float32(num_test)

gpu_name = cp.cuda.runtime.getDeviceProperties(0)["name"]
if isinstance(gpu_name, bytes):
    gpu_name = gpu_name.decode("utf-8")

print("读取的参数文件:", checkpoint_path)
print("GPU 型号:", gpu_name)
print("参数数组数量:", len(params))
print("参数标量总数:", sum(value.size for value in params.values()))
print("W_in 的实际设备:", params["W_in"].device)
print("W_in 的数据类型:", params["W_in"].dtype)
print("测试集样本数:", num_test)
print("测试集平均损失:", float(mean_loss))
print("预测正确数量:", int(correct_sum))
print(f"测试集准确率: {float(accuracy) * 100:.2f}%")

for image_index, true_digit, predicted_digit, probability in preview_results:
    print(
        f"图片 {image_index} | 真实: {true_digit} | 预测: {predicted_digit} | "
        f"预测类别概率: {probability:.6f}"
    )
~~~

## 实际执行、验证与结果（事实）

解释器 F:\PythonProjects\deep_learning\.venv\Scripts\python.exe，Python 3.13.15、NumPy 2.5.3、CuPy 14.2.0；GPU0 NVIDIA GeForce RTX 5060，可见GPU数1。独立进程仅runpy.run_path上述新文件，未运行/导入原主文件、旧前向定义或旧训练循环。

精确代码真实输出：

~~~text
读取的参数文件: F:\PythonProjects\deep_learning\outputs\checkpoints\transformer_mnist_20261008_115104_418995.npz
GPU 型号: NVIDIA GeForce RTX 5060
参数数组数量: 20
参数标量总数: 9802
W_in 的实际设备: <CUDA Device 0>
W_in 的数据类型: float32
测试集样本数: 1000
测试集平均损失: 0.4118737280368805
预测正确数量: 875
测试集准确率: 87.50%
图片 0 | 真实: 7 | 预测: 7 | 预测类别概率: 0.994331
图片 1 | 真实: 2 | 预测: 2 | 预测类别概率: 0.485387
图片 2 | 真实: 1 | 预测: 1 | 预测类别概率: 0.973488
图片 3 | 真实: 0 | 预测: 0 | 预测类别概率: 0.990463
图片 4 | 真实: 4 | 预测: 4 | 预测类别概率: 0.993720
~~~

1000图固定评估平均损失0.4118737280368805、正确875/1000、准确率87.50%。这些数值是本次新执行所得，与v048原固定测试结果相同，不是复制历史指标作为新结果。全部1000图的单图loss、prediction及真实类别概率逐项与v048已保存GPU日志精确一致，最大损失/真实类概率绝对差均0.0。测试索引0、1、999全部10类概率另与v051已保存GPU结果逐位一致，最大差0.0。前5图预览直接复用该次循环输出，无新增前向。

独立observer实际计数：infer_one_image 1000次，layer_norm 2000次，backward_one_image 0次，train_one_image 0次，新增更新0次。每个infer返回时后台从其返回logits再算一次稳定CE，用于保存并对照单图日志；共1000次后台CE计算，没有再次调用网络。CPU NumPy对这1000损失按相同顺序float32累计，得到loss_sum=411.87371826171875、mean_loss=0.4118737280368805；独立整数正确数875、float32准确率0.875，与最终GPU标量精确相同。

所有20参数与位置编码在开始/结束之间内容SHA256、对象身份、GPU存储地址不变；params字典对象及键顺序不变，参数全部有限、GPU0 float32。图像和CPU标签内容、身份、存储地址同样不变。原用户存档、主文件、原数据、v048～v051旧记录哈希不变。静态AST确认没有训练集标识、旧main导入、随机生成、反向或参数更新，参数由checkpoint成员cp.asarray恢复。profile观察器没有改动网络返回或变量，结束恢复原profile。

JSON保存精确新文件源码/哈希、原文件保护哈希、存档26成员、20参数GPU状态、数据/配置、1000图10类logits和概率及loss/prediction、对照范围、三个10概率样本对照、实际调用计数、CPU指标重算与stdout。[实际执行证据](2026-10-08_transformer_standalone_validation_v052.json) · [存档字节副本](checkpoints/transformer_mnist_20261008_115104_418995.npz) · [独立代码](transformer_validation_v052.py)。相对链接已核验，本步不新增图。

## 与前版变化、未解限制及下一步

v051是在已有脚本变量上定义无标签单图预测，后台只比较3图。v052改为完整独立文件，从实际存档恢复参数及配置、只读取测试数据，并新执行1000图固定评估。训练数学、参数值、数据划分与预处理没有改变。保留旧记录，追加项目索引；同步复制既有证据/源码/存档附件，不再执行推理或训练。

核验进程独立于用户终端内存，不声称读取用户终端对象。本步只覆盖固定checkpoint、同一GPU及前1000原测试图片；完整10类概率历史对照仅索引0/1/999，不声称1000图全部概率有独立CPU数学参考。没有新的CPUfloat64网络执行或有限差分，因为本次变化是加载和独立组织前向，旧数学已有核验。代码不是通用损坏存档诊断器，使用者需保证文件存在、结构兼容并选择项目.venv Python。CUDA DLL名称与当前环境对应，其他环境可能需调整启动段。

87.50%只描述本次1000图，不代表全MNIST基准、最优模型、概率校准或外部手写图片表现。训练权重此前仅来自原前5000训练图、3轮当前SGD教学训练。自带图片需要与训练一致的28×28灰度、前景/背景和归一化处理，不能仅把任意图片路径交给当前函数。

下一步可在独立文件中筛选预测错误的测试图片，或另写外部图像预处理及无标签预测；均可复用现有checkpoint，无需重训。
