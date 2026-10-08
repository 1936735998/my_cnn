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