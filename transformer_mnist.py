import ctypes
import os
import sys
from importlib.util import find_spec
from pathlib import Path

import numpy as np


# ==================================================
# GPU 环境准备
# 复用项目已有的 CUDA 库，不使用 PyTorch 计算模型
# ==================================================

torch_spec = find_spec("torch")

if torch_spec is None:
    raise RuntimeError("请检查是否选择了项目的 .venv 解释器")

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


# ==================================================
# 检查计算设备
# ==================================================

cp.cuda.Device(0).use()

gpu_name = cp.cuda.runtime.getDeviceProperties(0)["name"]

if isinstance(gpu_name, bytes):
    gpu_name = gpu_name.decode("utf-8")

print("Python:", sys.version.split()[0])
print("NumPy:", np.__version__)
print("CuPy:", cp.__version__)
print("GPU 数量:", cp.cuda.runtime.getDeviceCount())
print("GPU 型号:", gpu_name)


# ==================================================
# MNIST 数据
# 读取你已经下载的本地缓存
# ==================================================

data_path = Path.home() / ".keras" / "datasets" / "mnist.npz"

if not data_path.exists():
    raise FileNotFoundError(f"没有找到 MNIST 缓存: {data_path}")

with np.load(data_path, allow_pickle=False) as data:

    x_train = cp.asarray(
        data["x_train"][:5000],
        dtype=cp.float32
    ) / 255.0

    y_train = cp.asarray(
        data["y_train"][:5000],
        dtype=cp.int64
    )

    x_test = cp.asarray(
        data["x_test"][:1000],
        dtype=cp.float32
    ) / 255.0

    y_test = cp.asarray(
        data["y_test"][:1000],
        dtype=cp.int64
    )


# ==================================================
# 检查序列形状
# ==================================================

sequence = x_train[0]
label = y_train[0]

print("x_train:", x_train.shape)
print("y_train:", y_train.shape)
print("x_test:", x_test.shape)
print("y_test:", y_test.shape)

print("一张图片:", sequence.shape)
print("一行像素:", sequence[0].shape)

print("像素范围:", float(x_train.min()), float(x_train.max()))
print("第一张图片的标签:", int(label))

print("训练输入的实际设备:", x_train.device)
print("测试输入的实际设备:", x_test.device)
# ==================================================
# 输入投影参数初始化
# ==================================================

cp.random.seed(42)#生成随机数的种子

input_size = 28
d_model = 32

W_in = cp.random.randn(input_size, d_model).astype(cp.float32) * 0.1
b_in = cp.zeros(d_model, dtype=cp.float32)

print("W_in:", W_in.shape)
print("b_in:", b_in.shape)


# ==================================================
# 将每行的 28 个像素映射成 32 个特征
# ==================================================

X = x_train[0]
embedding = cp.dot(X, W_in) + b_in

print("X:", X.shape)
print("embedding:", embedding.shape)
print("一行映射后的特征:", embedding[0].shape)
print("W_in 的实际设备:", W_in.device)
print("embedding 的实际设备:", embedding.device)
# ==================================================
# 正弦位置编码
# ==================================================

seq_len = embedding.shape[0]

# 28 个行位置，编号为 0～27
position = cp.arange(seq_len, dtype=cp.float32).reshape(seq_len, 1)#生成一个28行1列的矩阵，里面的值是0到27

# 选择第 0、2、4、...、30 个特征位置
dimension_index = cp.arange(0, d_model, 2, dtype=cp.float32)#生成一个16行1列的矩阵，里面的值是0到30，步长为2

angles = position / (10000 ** (dimension_index / d_model))#生成角度

position_encoding = cp.zeros((seq_len, d_model), dtype=cp.float32)#生成一个28行32列的矩阵，里面的值都是0

position_encoding[:, 0::2] = cp.sin(angles)
position_encoding[:, 1::2] = cp.cos(angles)


# ==================================================
# 输入特征 + 位置信息
# ==================================================

encoder_input = embedding + position_encoding

print("position_encoding:", position_encoding.shape)
print("encoder_input:", encoder_input.shape)
print("第 0 行位置编码的前 6 个值:", position_encoding[0, :6])
print("第 1 行位置编码的前 6 个值:", position_encoding[1, :6])
print("encoder_input 的实际设备:", encoder_input.device)
# ==================================================
# Q、K、V 的投影参数 Q当前位置用于寻找相关信息的特征，K各个位置用于与 Q 匹配的特征，V最终被取出、汇总的信息
# ==================================================

W_Q = cp.random.randn(d_model, d_model).astype(cp.float32) * 0.1
W_K = cp.random.randn(d_model, d_model).astype(cp.float32) * 0.1
W_V = cp.random.randn(d_model, d_model).astype(cp.float32) * 0.1

b_Q = cp.zeros(d_model, dtype=cp.float32)
b_K = cp.zeros(d_model, dtype=cp.float32)
b_V = cp.zeros(d_model, dtype=cp.float32)

print("W_Q:", W_Q.shape)
print("W_K:", W_K.shape)
print("W_V:", W_V.shape)


# ==================================================
# 同一份输入，生成三种特征表示
# ==================================================

Q = cp.dot(encoder_input, W_Q) + b_Q
K = cp.dot(encoder_input, W_K) + b_K
V = cp.dot(encoder_input, W_V) + b_V

print("Q:", Q.shape)
print("K:", K.shape)
print("V:", V.shape)

print("Q 的实际设备:", Q.device)
print("K 的实际设备:", K.device)
print("V 的实际设备:", V.device)
# ==================================================
# 拆分为 4 个注意力头
# ==================================================

num_heads = 4
head_dim = d_model // num_heads

Q_heads = Q.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)
K_heads = K.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)
V_heads = V.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)

print("每个头的特征数:", head_dim)
print("Q_heads:", Q_heads.shape)
print("K_heads:", K_heads.shape)
print("V_heads:", V_heads.shape)
print("第 0 个头的 Q:", Q_heads[0].shape)
print("Q_heads 的实际设备:", Q_heads.device)
# ==================================================
# 计算每个头的匹配分数
# ==================================================

# (4, 28, 8) → (4, 8, 28)
# 保留头的维度，交换最后两个维度
K_transposed = K_heads.transpose(0, 2, 1)

# 每个头分别进行矩阵乘法
# (4, 28, 8) @ (4, 8, 28) → (4, 28, 28)
attention_scores = cp.matmul(Q_heads, K_transposed)#Q*K

# 每个头有 8 个特征，所以除以 sqrt(8)
attention_scores = attention_scores / cp.sqrt(cp.float32(head_dim))

print("K_transposed:", K_transposed.shape)
print("attention_scores:", attention_scores.shape)
print("第 0 个头的分数矩阵:", attention_scores[0].shape)
print("attention_scores 的实际设备:", attention_scores.device)
# ==================================================
# 稳定 softmax：把匹配分数转换成注意力权重
# ==================================================

# 每一行减去该行的最大分数，避免 exp 溢出
scores_shifted = attention_scores - cp.max(
    attention_scores, axis=-1, keepdims=True
)

exp_scores = cp.exp(scores_shifted)

# 对每个查询位置对应的 28 个键位置归一化
attention_weights = exp_scores / cp.sum(
    exp_scores, axis=-1, keepdims=True
)

weight_sums = cp.sum(attention_weights, axis=-1)

print("attention_weights:", attention_weights.shape)
print("每行权重和的形状:", weight_sums.shape)
print("第 0 个头前 5 行的权重和:", weight_sums[0, :5])
print("attention_weights 的实际设备:", attention_weights.device)
# ==================================================
# 用注意力权重对 V 加权求和
# ==================================================

# (4, 28, 28) @ (4, 28, 8) → (4, 28, 8)
head_output = cp.matmul(attention_weights, V_heads)

print("head_output:", head_output.shape)
print("第 0 个头的输出:", head_output[0].shape)
print("head_output 的实际设备:", head_output.device)
# ==================================================
# 合并四个头的输出
# ==================================================

# (4, 28, 8) → (28, 4, 8)
# 把同一行的四个头放在一起
head_output_by_position = head_output.transpose(1, 0, 2)

# (28, 4, 8) → (28, 32)
# 将每行的 4 × 8 个特征拼接起来
concatenated_heads = head_output_by_position.reshape(seq_len, d_model)

print("head_output_by_position:", head_output_by_position.shape)
print("concatenated_heads:", concatenated_heads.shape)
print("一行拼接后的特征:", concatenated_heads[0].shape)
print("concatenated_heads 的实际设备:", concatenated_heads.device)
# ==================================================
# 多头注意力的输出投影
# ==================================================

# 沿用前面的随机数状态，这里不用重新设置种子
W_O = cp.random.randn(d_model, d_model).astype(cp.float32) * 0.1
b_O = cp.zeros(d_model, dtype=cp.float32)

# (28, 32) @ (32, 32) + (32,) → (28, 32)
attention_output = cp.dot(concatenated_heads, W_O) + b_O

print("W_O:", W_O.shape)
print("b_O:", b_O.shape)
print("attention_output:", attention_output.shape)
print("W_O 的实际设备:", W_O.device)
print("attention_output 的实际设备:", attention_output.device)
# ==================================================
# 注意力子层的残差连接
# ==================================================

# (28, 32) + (28, 32) → (28, 32)
attention_residual = encoder_input + attention_output
#残差连接，encoder_input是输入特征+位置信息
#attention_output是多头注意力的输出投影,是使用encoder_input计算得到的，所以两者形状相同，可以相加
print("attention_residual:", attention_residual.shape)
print("一行残差相加后的特征:", attention_residual[0].shape)
print("attention_residual 的实际设备:", attention_residual.device)
# ==================================================
# 注意力残差之后的 LayerNorm
# ==================================================

gamma_attn = cp.ones(d_model, dtype=cp.float32)
beta_attn = cp.zeros(d_model, dtype=cp.float32)
ln_eps = cp.float32(1e-5)

# 每一行的 32 个特征分别计算均值和方差
mean_attn = cp.mean(attention_residual, axis=-1, keepdims=True)#均值
centered_attn = attention_residual - mean_attn
var_attn = cp.mean(centered_attn ** 2, axis=-1, keepdims=True)#方差

# epsilon 放在开平方内部，避免零方差导致除零
inv_std_attn = cp.float32(1.0) / cp.sqrt(var_attn + ln_eps)
normalized_attn = centered_attn * inv_std_attn  # 标准化：x_hat = (x - μ) / sqrt(σ² + ε)

# 可训练的逐特征缩放和偏移
attention_norm = normalized_attn * gamma_attn + beta_attn

print("gamma_attn:", gamma_attn.shape)
print("beta_attn:", beta_attn.shape)
print("mean_attn:", mean_attn.shape)
print("var_attn:", var_attn.shape)
print("attention_norm:", attention_norm.shape)
print("第 0 行输出的均值:", float(cp.mean(attention_norm[0])))
print("第 0 行输出的方差:", float(cp.var(attention_norm[0])))
print("attention_norm 的实际设备:", attention_norm.device)
# ==================================================
# 前馈网络第一层：32 → 64，再经过 ReLU
# ==================================================

d_ff = 64

W_ff1 = cp.random.randn(d_model, d_ff).astype(cp.float32) * 0.1
b_ff1 = cp.zeros(d_ff, dtype=cp.float32)

# 对全部 28 行使用同一组参数
# (28, 32) @ (32, 64) + (64,) → (28, 64)
ff_hidden_linear = cp.dot(attention_norm, W_ff1) + b_ff1

# ReLU：正数保留，负数变为 0
ff_hidden = cp.maximum(ff_hidden_linear, cp.float32(0.0))

print("W_ff1:", W_ff1.shape)
print("b_ff1:", b_ff1.shape)
print("ff_hidden_linear:", ff_hidden_linear.shape)
print("ff_hidden:", ff_hidden.shape)
print("ff_hidden 的实际设备:", ff_hidden.device)
# ==================================================
# 前馈网络第二层：64 → 32
# ==================================================

W_ff2 = cp.random.randn(d_ff, d_model).astype(cp.float32) * 0.1
b_ff2 = cp.zeros(d_model, dtype=cp.float32)

# 对全部 28 行使用同一组参数
# (28, 64) @ (64, 32) + (32,) → (28, 32)
ff_output = cp.dot(ff_hidden, W_ff2) + b_ff2

print("W_ff2:", W_ff2.shape)
print("b_ff2:", b_ff2.shape)
print("ff_output:", ff_output.shape)
print("ff_output 的实际设备:", ff_output.device)
# ==================================================
# 前馈子层的残差连接
# ==================================================

# (28, 32) + (28, 32) → (28, 32)
ff_residual = attention_norm + ff_output

print("ff_residual:", ff_residual.shape)
print("一行残差相加后的特征:", ff_residual[0].shape)
print("ff_residual 的实际设备:", ff_residual.device)
# ==================================================
# 前馈残差之后的第二次 LayerNorm
# ==================================================

# 与第一次 LayerNorm 分别使用自己的可训练参数
gamma_ffn = cp.ones(d_model, dtype=cp.float32)
beta_ffn = cp.zeros(d_model, dtype=cp.float32)

# 继续对每行的 32 个特征计算均值和方差
mean_ffn = cp.mean(ff_residual, axis=-1, keepdims=True)
centered_ffn = ff_residual - mean_ffn
var_ffn = cp.mean(centered_ffn ** 2, axis=-1, keepdims=True)

# ln_eps 沿用第一次 LayerNorm 设置的固定常数
inv_std_ffn = cp.float32(1.0) / cp.sqrt(var_ffn + ln_eps)
normalized_ffn = centered_ffn * inv_std_ffn
encoder_output = normalized_ffn * gamma_ffn + beta_ffn

print("gamma_ffn:", gamma_ffn.shape)
print("beta_ffn:", beta_ffn.shape)
print("mean_ffn:", mean_ffn.shape)
print("var_ffn:", var_ffn.shape)
print("encoder_output:", encoder_output.shape)
print("第 0 行输出的均值:", float(cp.mean(encoder_output[0])))
print("第 0 行输出的方差:", float(cp.var(encoder_output[0])))
print("encoder_output 的实际设备:", encoder_output.device)
# ==================================================
# 平均池化：汇总整张图片的特征
# ==================================================

# 沿 28 个行位置取平均，保留 32 个特征
# (28, 32) → (32,)
image_features = cp.mean(encoder_output, axis=0)

print("encoder_output:", encoder_output.shape)
print("image_features:", image_features.shape)
print("image_features 的实际设备:", image_features.device)
# ==================================================
# 分类头：32 个图片特征 → 10 个分类分数
# ==================================================

num_classes = 10

W_cls = cp.random.randn(d_model, num_classes).astype(cp.float32) * 0.1
b_cls = cp.zeros(num_classes, dtype=cp.float32)

# (32,) @ (32, 10) + (10,) → (10,)
logits = cp.dot(image_features, W_cls) + b_cls

print("W_cls:", W_cls.shape)
print("b_cls:", b_cls.shape)
print("logits:", logits.shape)
print("logits 的实际设备:", logits.device)
# ==================================================
# 稳定 softmax：分类分数 → 分类概率
# ==================================================

# 沿最后一维的 10 个类别，减去最大分数
shifted_logits = logits - cp.max(logits, axis=-1, keepdims=True)
exp_logits = cp.exp(shifted_logits)

# 每个类别的指数值除以全部类别的指数值之和
probabilities = exp_logits / cp.sum(exp_logits, axis=-1, keepdims=True)

print("probabilities:", probabilities.shape)
print("10 个分类概率:", probabilities)
print("概率之和:", float(cp.sum(probabilities)))
print("probabilities 的实际设备:", probabilities.device)
# ==================================================
# 真实标签、稳定交叉熵与当前预测
# ==================================================

# 当前输入是 x_train[0]，读取对应的真实标签
target = int(y_train[0])

# 等价于 -log(probabilities[target]) 的稳定计算
log_normalizer = cp.log(cp.sum(exp_logits))
loss = log_normalizer - shifted_logits[target]

predicted_class = int(cp.argmax(probabilities))

print("真实标签:", target)
print("当前预测:", predicted_class)
print("真实类别的概率:", float(probabilities[target]))
print("交叉熵损失:", float(loss))
print("loss 的形状:", loss.shape)
print("loss 的实际设备:", loss.device)
# ==================================================
# 反向第一步：损失对 logits 的梯度
# ==================================================

# 单独保存梯度，保留原来的 probabilities
d_logits = probabilities.copy()
d_logits[target] -= cp.float32(1.0)

print("d_logits:", d_logits.shape)
print("10 个分数的梯度:", d_logits)
print("梯度之和:", float(cp.sum(d_logits)))
print("d_logits 的实际设备:", d_logits.device)
# ==================================================
# 反向第二步：分类头
# ==================================================

# 外积：(32,) 与 (10,) → (32, 10)
dW_cls = cp.outer(image_features, d_logits)

# 偏置直接接收对应类别的梯度
db_cls = d_logits.copy()

# (32, 10) @ (10,) → (32,)
# 使用本次前向计算时的 W_cls
d_image_features = cp.dot(W_cls, d_logits)

print("dW_cls:", dW_cls.shape)
print("db_cls:", db_cls.shape)
print("d_image_features:", d_image_features.shape)
print("dW_cls 的实际设备:", dW_cls.device)
print("db_cls 的实际设备:", db_cls.device)
print("d_image_features 的实际设备:", d_image_features.device)
# ==================================================
# 反向第三步：平均池化
# ==================================================

# 每行接收图片特征梯度的 1/28
# (32,) → (28, 32)
d_encoder_output = cp.broadcast_to(
    d_image_features / cp.float32(seq_len),
    encoder_output.shape
).copy()

print("d_encoder_output:", d_encoder_output.shape)
print("第 0 行的梯度形状:", d_encoder_output[0].shape)
print("首尾两行梯度是否相同:", bool(cp.all(
    d_encoder_output[0] == d_encoder_output[-1]
)))
print("各行梯度之和与上游梯度的最大差值:", float(cp.max(cp.abs(
    cp.sum(d_encoder_output, axis=0) - d_image_features
))))
print("d_encoder_output 的实际设备:", d_encoder_output.device)
# ==================================================
# 反向第四步：第二次 LayerNorm
# ==================================================

# gamma、beta 在 28 行之间共享，参数梯度沿位置轴求和
dgamma_ffn = cp.sum(d_encoder_output * normalized_ffn, axis=0)
dbeta_ffn = cp.sum(d_encoder_output, axis=0)

# 先反向通过缩放：encoder_output = normalized_ffn * gamma_ffn + beta_ffn
d_normalized_ffn = d_encoder_output * gamma_ffn

# 每一行分别在 32 个特征内取平均，保留 (28, 1) 便于广播
mean_d_normalized_ffn = cp.mean(
    d_normalized_ffn, axis=-1, keepdims=True
)
mean_d_normalized_times_x_ffn = cp.mean(
    d_normalized_ffn * normalized_ffn, axis=-1, keepdims=True
)

# 继续反向通过标准化，得到对 ff_residual 的梯度
d_ff_residual = inv_std_ffn * (
    d_normalized_ffn
    - mean_d_normalized_ffn
    - normalized_ffn * mean_d_normalized_times_x_ffn
)

print("dgamma_ffn:", dgamma_ffn.shape)
print("dbeta_ffn:", dbeta_ffn.shape)
print("d_normalized_ffn:", d_normalized_ffn.shape)
print("mean_d_normalized_ffn:", mean_d_normalized_ffn.shape)
print("mean_d_normalized_times_x_ffn:", mean_d_normalized_times_x_ffn.shape)
print("d_ff_residual:", d_ff_residual.shape)
print("dgamma_ffn 的实际设备:", dgamma_ffn.device)
print("dbeta_ffn 的实际设备:", dbeta_ffn.device)
print("d_ff_residual 的实际设备:", d_ff_residual.device)
# ==================================================
# 反向第五步：前馈残差连接
# ==================================================

# 直连支路：先保存传给 attention_norm 的这部分梯度
d_attention_norm_skip = d_ff_residual.copy()

# 前馈支路：传给 ff_output，接下来继续反向经过前馈网络
d_ff_output = d_ff_residual.copy()

print("d_attention_norm_skip:", d_attention_norm_skip.shape)
print("d_ff_output:", d_ff_output.shape)
print("d_attention_norm_skip 的实际设备:", d_attention_norm_skip.device)
print("d_ff_output 的实际设备:", d_ff_output.device)
# ==================================================
# 反向第六步：前馈网络第二层
# ==================================================

# (64, 28) @ (28, 32) → (64, 32)
# 累加 28 行对共享权重的梯度贡献
dW_ff2 = cp.dot(ff_hidden.T, d_ff_output)

# 每个输出特征的偏置梯度，沿 28 行求和
db_ff2 = cp.sum(d_ff_output, axis=0)

# (28, 32) @ (32, 64) → (28, 64)
# 使用本次前向计算时的 W_ff2
d_ff_hidden = cp.dot(d_ff_output, W_ff2.T)

print("dW_ff2:", dW_ff2.shape)
print("db_ff2:", db_ff2.shape)
print("d_ff_hidden:", d_ff_hidden.shape)
print("dW_ff2 的实际设备:", dW_ff2.device)
print("db_ff2 的实际设备:", db_ff2.device)
print("d_ff_hidden 的实际设备:", d_ff_hidden.device)
# ==================================================
# 反向第七步：ReLU
# ==================================================

# 正输入处为 True，负输入和零输入处为 False
relu_mask = ff_hidden_linear > cp.float32(0.0)

# True 相当于乘 1，False 相当于乘 0
d_ff_hidden_linear = d_ff_hidden * relu_mask

print("relu_mask:", relu_mask.shape)
print("d_ff_hidden_linear:", d_ff_hidden_linear.shape)
print("负数或零处的梯度是否全为 0:", bool(cp.all(
    d_ff_hidden_linear[~relu_mask] == cp.float32(0.0)
)))
print("正数处的梯度是否保持不变:", bool(cp.all(
    d_ff_hidden_linear[relu_mask] == d_ff_hidden[relu_mask]
)))
print("d_ff_hidden_linear 的实际设备:", d_ff_hidden_linear.device)
# ==================================================
# 反向第八步：前馈网络第一层
# ==================================================

# (32, 28) @ (28, 64) → (32, 64)
# 累加 28 行对共享权重的梯度贡献
dW_ff1 = cp.dot(attention_norm.T, d_ff_hidden_linear)

# 每个隐藏特征的偏置梯度，沿 28 行求和
db_ff1 = cp.sum(d_ff_hidden_linear, axis=0)

# (28, 64) @ (64, 32) → (28, 32)
# 使用本次前向的 W_ff1，得到前馈支路返回的梯度
d_attention_norm_from_ffn = cp.dot(d_ff_hidden_linear, W_ff1.T)

print("dW_ff1:", dW_ff1.shape)
print("db_ff1:", db_ff1.shape)
print("d_attention_norm_from_ffn:", d_attention_norm_from_ffn.shape)
print("dW_ff1 的实际设备:", dW_ff1.device)
print("db_ff1 的实际设备:", db_ff1.device)
print("d_attention_norm_from_ffn 的实际设备:", d_attention_norm_from_ffn.device)
# ==================================================
# 反向第九步：汇合前馈与直连路径的梯度
# ==================================================

# attention_norm 同时进入两条路径，总梯度是两份贡献之和
d_attention_norm = d_attention_norm_skip + d_attention_norm_from_ffn

print("d_attention_norm_skip:", d_attention_norm_skip.shape)
print("d_attention_norm_from_ffn:", d_attention_norm_from_ffn.shape)
print("d_attention_norm:", d_attention_norm.shape)
print("d_attention_norm 的实际设备:", d_attention_norm.device)
# ==================================================
# 反向第十步：第一次 LayerNorm
# ==================================================

# gamma、beta 在 28 行之间共享，参数梯度沿位置轴求和
dgamma_attn = cp.sum(d_attention_norm * normalized_attn, axis=0)
dbeta_attn = cp.sum(d_attention_norm, axis=0)

# 反向通过缩放：attention_norm = normalized_attn * gamma_attn + beta_attn
d_normalized_attn = d_attention_norm * gamma_attn

# 每行在 32 个特征内取平均，保留 (28, 1) 便于广播
mean_d_normalized_attn = cp.mean(
    d_normalized_attn, axis=-1, keepdims=True
)
mean_d_normalized_times_x_attn = cp.mean(
    d_normalized_attn * normalized_attn, axis=-1, keepdims=True
)

# 反向通过标准化，得到对 attention_residual 的梯度
d_attention_residual = inv_std_attn * (
    d_normalized_attn
    - mean_d_normalized_attn
    - normalized_attn * mean_d_normalized_times_x_attn
)

print("dgamma_attn:", dgamma_attn.shape)
print("dbeta_attn:", dbeta_attn.shape)
print("d_normalized_attn:", d_normalized_attn.shape)
print("mean_d_normalized_attn:", mean_d_normalized_attn.shape)
print("mean_d_normalized_times_x_attn:", mean_d_normalized_times_x_attn.shape)
print("d_attention_residual:", d_attention_residual.shape)
print("dgamma_attn 的实际设备:", dgamma_attn.device)
print("dbeta_attn 的实际设备:", dbeta_attn.device)
print("d_attention_residual 的实际设备:", d_attention_residual.device)
# ==================================================
# 反向第十一步：注意力残差连接
# ==================================================

# 直连支路：先保存传给 encoder_input 的这部分梯度
d_encoder_input_skip = d_attention_residual.copy()

# 注意力支路：传给 attention_output，接下来继续反向经过注意力
d_attention_output = d_attention_residual.copy()

print("d_encoder_input_skip:", d_encoder_input_skip.shape)
print("d_attention_output:", d_attention_output.shape)
print("d_encoder_input_skip 的实际设备:", d_encoder_input_skip.device)
print("d_attention_output 的实际设备:", d_attention_output.device)
# ==================================================
# 反向第十二步：注意力输出投影
# ==================================================

# (32, 28) @ (28, 32) → (32, 32)
# 累加 28 行对共享权重的梯度贡献
dW_O = cp.dot(concatenated_heads.T, d_attention_output)

# 输出偏置在 28 行之间共享，沿位置轴求和
db_O = cp.sum(d_attention_output, axis=0)

# (28, 32) @ (32, 32) → (28, 32)
# 使用本次前向的 W_O，把梯度传回拼接后的多头特征
d_concatenated_heads = cp.dot(d_attention_output, W_O.T)

print("dW_O:", dW_O.shape)
print("db_O:", db_O.shape)
print("d_concatenated_heads:", d_concatenated_heads.shape)
print("dW_O 的实际设备:", dW_O.device)
print("db_O 的实际设备:", db_O.device)
print("d_concatenated_heads 的实际设备:", d_concatenated_heads.device)
# ==================================================
# 反向第十三步：逆转多头拼接
# ==================================================

# 每个位置的 32 个特征，重新分成 4 个头，每个头 8 个特征
# (28, 32) → (28, 4, 8)
d_head_output_by_position = d_concatenated_heads.reshape(
    seq_len, num_heads, head_dim
)

# 把头的轴移到最前面：(28, 4, 8) → (4, 28, 8)
d_head_output = d_head_output_by_position.transpose(1, 0, 2).copy()

print("d_head_output_by_position:", d_head_output_by_position.shape)
print("d_head_output:", d_head_output.shape)
print("第 0 个头的输出梯度:", d_head_output[0].shape)
print("d_head_output 的实际设备:", d_head_output.device)
# ==================================================
# 反向第十四步：注意力加权求和
# ==================================================

# 每个头分别计算：(28, 8) @ (8, 28) → (28, 28)
# 头轴保持不变，只交换 V 的位置轴和特征轴
d_attention_weights = cp.matmul(
    d_head_output,
    V_heads.transpose(0, 2, 1)
)

# 每个头分别计算：(28, 28) @ (28, 8) → (28, 8)
# 累加所有查询位置对 V 的梯度贡献
d_V_heads = cp.matmul(
    attention_weights.transpose(0, 2, 1),
    d_head_output
)

print("d_attention_weights:", d_attention_weights.shape)
print("d_V_heads:", d_V_heads.shape)
print("d_attention_weights 的实际设备:", d_attention_weights.device)
print("d_V_heads 的实际设备:", d_V_heads.device)
# ==================================================
# 反向第十五步：注意力 softmax
# ==================================================

# 对每个头、每个查询行，沿 28 个键位置计算加权和
# (4, 28, 28) → (4, 28, 1)
weighted_gradient_sum = cp.sum(
    d_attention_weights * attention_weights,
    axis=-1,
    keepdims=True
)

# softmax 反向：每个权重乘以对应上游梯度与该行加权和之差
d_attention_scores = attention_weights * (
    d_attention_weights - weighted_gradient_sum
)

print("weighted_gradient_sum:", weighted_gradient_sum.shape)
print("d_attention_scores:", d_attention_scores.shape)
print("第 0 个头前 5 行的梯度和:", cp.sum(
    d_attention_scores, axis=-1
)[0, :5])
print("d_attention_scores 的实际设备:", d_attention_scores.device)
# ==================================================
# 反向第十六步：缩放 QK 分数
# ==================================================

# 前向除以 sqrt(head_dim)，反向也先除以同一个常数
attention_scale = cp.sqrt(cp.float32(head_dim))
d_qk_product = d_attention_scores / attention_scale

# 每个头分别计算：(28, 28) @ (28, 8) → (28, 8)
d_Q_heads = cp.matmul(d_qk_product, K_heads)

# 交换分数梯度的查询轴和键轴，累加各查询对 K 的贡献
# 每个头分别计算：(28, 28) @ (28, 8) → (28, 8)
d_K_heads = cp.matmul(
    d_qk_product.transpose(0, 2, 1),
    Q_heads
)

print("attention_scale:", float(attention_scale))
print("d_qk_product:", d_qk_product.shape)
print("d_Q_heads:", d_Q_heads.shape)
print("d_K_heads:", d_K_heads.shape)
print("d_Q_heads 的实际设备:", d_Q_heads.device)
print("d_K_heads 的实际设备:", d_K_heads.device)
# ==================================================
# 反向第十七步：逆转 Q、K、V 的拆头
# ==================================================

# 先恢复位置轴在前，再拼回每个位置的 32 个特征
# (4, 28, 8) → (28, 4, 8) → (28, 32)
d_Q = d_Q_heads.transpose(1, 0, 2).reshape(seq_len, d_model).copy()
d_K = d_K_heads.transpose(1, 0, 2).reshape(seq_len, d_model).copy()
d_V = d_V_heads.transpose(1, 0, 2).reshape(seq_len, d_model).copy()

print("d_Q:", d_Q.shape)
print("d_K:", d_K.shape)
print("d_V:", d_V.shape)
print("d_Q 的实际设备:", d_Q.device)
print("d_K 的实际设备:", d_K.device)
print("d_V 的实际设备:", d_V.device)
# ==================================================
# 反向第十八步：Q、K、V 的线性投影
# ==================================================

# Q 投影：共享参数梯度，以及 Q 路径返回的输入梯度
dW_Q = cp.dot(encoder_input.T, d_Q)
db_Q = cp.sum(d_Q, axis=0)
d_encoder_input_from_Q = cp.dot(d_Q, W_Q.T)

# K 投影
dW_K = cp.dot(encoder_input.T, d_K)
db_K = cp.sum(d_K, axis=0)
d_encoder_input_from_K = cp.dot(d_K, W_K.T)

# V 投影
dW_V = cp.dot(encoder_input.T, d_V)
db_V = cp.sum(d_V, axis=0)
d_encoder_input_from_V = cp.dot(d_V, W_V.T)

print("dW_Q:", dW_Q.shape)
print("db_Q:", db_Q.shape)
print("d_encoder_input_from_Q:", d_encoder_input_from_Q.shape)
print("dW_K:", dW_K.shape)
print("db_K:", db_K.shape)
print("d_encoder_input_from_K:", d_encoder_input_from_K.shape)
print("dW_V:", dW_V.shape)
print("db_V:", db_V.shape)
print("d_encoder_input_from_V:", d_encoder_input_from_V.shape)
print("Q 路径的实际设备:", d_encoder_input_from_Q.device)
print("K 路径的实际设备:", d_encoder_input_from_K.device)
print("V 路径的实际设备:", d_encoder_input_from_V.device)
# ==================================================
# 反向第十九步：汇合 encoder_input 的四条路径
# ==================================================

# 同一个输入通过直连、Q、K、V 四条路径影响损失
d_encoder_input = (
    d_encoder_input_skip
    + d_encoder_input_from_Q
    + d_encoder_input_from_K
    + d_encoder_input_from_V
)

print("d_encoder_input:", d_encoder_input.shape)
print("一行输入特征的梯度:", d_encoder_input[0].shape)
print("d_encoder_input 的实际设备:", d_encoder_input.device)
# ==================================================
# 反向第二十步：从 encoder_input 传回 embedding
# ==================================================

# encoder_input = embedding + position_encoding
# 位置编码固定，梯度直接传回输入映射的输出
d_embedding = d_encoder_input.copy()

print("d_embedding:", d_embedding.shape)
print("一行映射特征的梯度:", d_embedding[0].shape)
print("d_embedding 的实际设备:", d_embedding.device)
# ==================================================
# 反向第二十一步：输入线性映射
# ==================================================

# embedding = X @ W_in + b_in
dW_in = cp.dot(X.T, d_embedding)
db_in = cp.sum(d_embedding, axis=0)
d_X = cp.dot(d_embedding, W_in.T)

print("dW_in:", dW_in.shape)
print("db_in:", db_in.shape)
print("d_X:", d_X.shape)
print("dW_in 的实际设备:", dW_in.device)
print("db_in 的实际设备:", db_in.device)
print("d_X 的实际设备:", d_X.device)
# ==================================================
# 参数更新第一步：手写 SGD
# ==================================================

learning_rate = cp.float32(0.01)

# 保存原参数数组的引用，不复制参数
params = {
    "W_in": W_in, "b_in": b_in,
    "W_Q": W_Q, "b_Q": b_Q,
    "W_K": W_K, "b_K": b_K,
    "W_V": W_V, "b_V": b_V,
    "W_O": W_O, "b_O": b_O,
    "gamma_attn": gamma_attn, "beta_attn": beta_attn,
    "W_ff1": W_ff1, "b_ff1": b_ff1,
    "W_ff2": W_ff2, "b_ff2": b_ff2,
    "gamma_ffn": gamma_ffn, "beta_ffn": beta_ffn,
    "W_cls": W_cls, "b_cls": b_cls,
}

# 每个参数对应本次反向传播得到的梯度
grads = {
    "W_in": dW_in, "b_in": db_in,
    "W_Q": dW_Q, "b_Q": db_Q,
    "W_K": dW_K, "b_K": db_K,
    "W_V": dW_V, "b_V": db_V,
    "W_O": dW_O, "b_O": db_O,
    "gamma_attn": dgamma_attn, "beta_attn": dbeta_attn,
    "W_ff1": dW_ff1, "b_ff1": db_ff1,
    "W_ff2": dW_ff2, "b_ff2": db_ff2,
    "gamma_ffn": dgamma_ffn, "beta_ffn": dbeta_ffn,
    "W_cls": dW_cls, "b_cls": db_cls,
}

# 留下更新前的损失；参数变化不会自动改变旧 loss
loss_before_update = loss.copy()

# 原地更新：参数 = 参数 - 学习率 × 梯度
for name in params:
    params[name] -= learning_rate * grads[name]

print("学习率:", f"{float(learning_rate):.2f}")
print("已更新的参数数组数量:", len(params))
print("参数标量总数:", sum(parameter.size for parameter in params.values()))
print("更新前的损失:", float(loss_before_update))
print("W_in 的实际设备:", W_in.device)
print("W_cls 的实际设备:", W_cls.device)
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
# ==================================================
# 训练准备：单张图片的完整训练步骤
# ==================================================

def train_one_image(
    X, target, params, position_encoding, num_heads, ln_eps, learning_rate
):
    # 1. 用当前参数做一次新的前向传播
    loss_before, probabilities_before, cache = forward_one_image(
        X, target, params, position_encoding, num_heads, ln_eps
    )

    # 2. 根据这次前向的缓存，计算新的梯度
    grads, _ = backward_one_image(params, cache)

    # 3. 所有梯度计算完毕后，统一更新参数
    for name in params:
        params[name] -= learning_rate * grads[name]

    # 返回本步更新前的损失与概率
    return loss_before, probabilities_before


# 索引 1 对应第二张训练图片
sample_index = 1
X_step = x_train[sample_index]
target_step = int(y_train[sample_index])

loss_before_step, probabilities_before_step = train_one_image(
    X_step, target_step, params, position_encoding,
    num_heads, ln_eps, learning_rate
)

# 再做一次前向，观察这一步更新后的结果
loss_after_step, probabilities_after_step, cache_step = forward_one_image(
    X_step, target_step, params, position_encoding, num_heads, ln_eps
)

print("本次训练图片索引:", sample_index)
print("真实标签:", target_step)
print("本步更新前的损失:", float(loss_before_step))
print("本步更新后的损失:", float(loss_after_step))
print("损失减少量:", float(loss_before_step - loss_after_step))
print("更新前的预测:", int(cp.argmax(probabilities_before_step)))
print("更新后的预测:", int(cp.argmax(probabilities_after_step)))
print("训练损失的实际设备:", loss_before_step.device)
print("W_in 的实际设备:", params["W_in"].device)
# ==================================================
# 第一次多图训练：前 128 张图片，训练 1 轮
# ==================================================

num_train = min(128, x_train.shape[0])
num_epochs = 1

# 标签只复制一次，方便 Python 循环取出整数
train_labels_cpu = cp.asnumpy(y_train[:num_train])

for epoch in range(num_epochs):
    # 每轮打乱训练顺序，每张图片恰好使用一次
    order = cp.asnumpy(cp.random.permutation(num_train))

    loss_sum = cp.zeros((), dtype=cp.float32)
    correct_sum = cp.zeros((), dtype=cp.int64)

    for step, index in enumerate(order, start=1):
        index = int(index)
        target_i = int(train_labels_cpu[index])

        loss_i, probabilities_i = train_one_image(
            x_train[index], target_i, params, position_encoding,
            num_heads, ln_eps, learning_rate
        )

        # 累计每次更新前的损失和预测结果
        loss_sum += loss_i
        correct_sum += (cp.argmax(probabilities_i) == target_i)

        if step % 32 == 0 or step == num_train:
            mean_loss = float(loss_sum / cp.float32(step))
            online_accuracy = float(
                correct_sum.astype(cp.float32) / cp.float32(step)
            ) * 100.0

            print(
                f"第 {epoch + 1}/{num_epochs} 轮 | 已训练 {step}/{num_train} | "
                f"在线平均损失: {mean_loss:.4f} | "
                f"在线准确率: {online_accuracy:.2f}%"
            )

print("本段新增更新次数:", num_train * num_epochs)
print("本轮在线预测正确数量:", int(correct_sum))
print("W_in 的实际设备:", params["W_in"].device)
print("W_cls 的实际设备:", params["W_cls"].device)
print("W_in 的数据类型:", params["W_in"].dtype)
# ==================================================
# 固定参数评估：只做前向传播
# ==================================================

def evaluate_images(
    images, labels, params, position_encoding, num_heads, ln_eps
):
    num_images = images.shape[0]
    labels_cpu = cp.asnumpy(labels)

    loss_sum = cp.zeros((), dtype=cp.float32)
    correct_sum = cp.zeros((), dtype=cp.int64)

    for index in range(num_images):
        target_i = int(labels_cpu[index])
        loss_i, probabilities_i, _ = forward_one_image(
            images[index], target_i, params,
            position_encoding, num_heads, ln_eps
        )

        loss_sum += loss_i
        correct_sum += (cp.argmax(probabilities_i) == target_i)

    mean_loss = loss_sum / cp.float32(num_images)
    accuracy = correct_sum.astype(cp.float32) / cp.float32(num_images)
    return mean_loss, accuracy


# 评估刚才训练过的前 128 张图片
train_eval_loss, train_eval_accuracy = evaluate_images(
    x_train[:num_train], y_train[:num_train], params,
    position_encoding, num_heads, ln_eps
)

# 评估已经加载的 1000 张测试图片
test_eval_loss, test_eval_accuracy = evaluate_images(
    x_test, y_test, params, position_encoding, num_heads, ln_eps
)

print("训练子集样本数:", num_train)
print("训练子集平均损失:", float(train_eval_loss))
print("训练子集准确率:", f"{float(train_eval_accuracy) * 100.0:.2f}%")
print("测试集样本数:", x_test.shape[0])
print("测试集平均损失:", float(test_eval_loss))
print("测试集准确率:", f"{float(test_eval_accuracy) * 100.0:.2f}%")
print("评估损失的实际设备:", test_eval_loss.device)
print("评估准确率的数据类型:", test_eval_accuracy.dtype)
# 延续当前参数，用已加载的全部 5,000 张训练图片训练 3 轮。
num_train_full = x_train.shape[0]
num_epochs_full = 3
train_labels_cpu_full = cp.asnumpy(y_train)
train_history_full = []

for epoch_full in range(num_epochs_full):
    # 每轮打乱顺序；每张训练图片在本轮恰好使用一次。
    order_full = cp.asnumpy(cp.random.permutation(num_train_full))
    loss_sum_full = cp.zeros((), dtype=cp.float32)
    correct_sum_full = cp.zeros((), dtype=cp.int64)

    for step_full, index_full in enumerate(order_full, start=1):
        index_full = int(index_full)
        target_full = int(train_labels_cpu_full[index_full])

        # 内部依次完成：前向传播、手写反向传播、一次 SGD 更新。
        # 返回的是本次更新之前的损失和概率。
        loss_i_full, probabilities_i_full = train_one_image(
            x_train[index_full], target_full, params,
            position_encoding, num_heads, ln_eps, learning_rate
        )

        loss_sum_full += loss_i_full
        correct_sum_full += (cp.argmax(probabilities_i_full) == target_full)

        if step_full % 500 == 0 or step_full == num_train_full:
            mean_loss_full = float(loss_sum_full / cp.float32(step_full))
            online_accuracy_full = float(
                correct_sum_full.astype(cp.float32) / cp.float32(step_full)
            ) * 100

            print(
                f"第 {epoch_full + 1}/{num_epochs_full} 轮 | "
                f"已训练 {step_full}/{num_train_full} | "
                f"在线平均损失: {mean_loss_full:.4f} | "
                f"在线准确率: {online_accuracy_full:.2f}%"
            )

    train_history_full.append({
        "epoch": epoch_full + 1,
        "online_mean_loss": float(loss_sum_full / cp.float32(num_train_full)),
        "online_accuracy": float(
            correct_sum_full.astype(cp.float32) / cp.float32(num_train_full)
        ),
    })

print("本段新增更新次数:", num_train_full * num_epochs_full)

# 训练结束后，使用同一组最终参数分别评估训练集与测试集。
full_train_eval_loss, full_train_eval_accuracy = evaluate_images(
    x_train, y_train, params, position_encoding, num_heads, ln_eps
)
full_test_eval_loss, full_test_eval_accuracy = evaluate_images(
    x_test, y_test, params, position_encoding, num_heads, ln_eps
)

print("完整训练集样本数:", x_train.shape[0])
print("完整训练集平均损失:", float(full_train_eval_loss))
print(f"完整训练集准确率: {float(full_train_eval_accuracy) * 100:.2f}%")
print("测试集样本数:", x_test.shape[0])
print("测试集平均损失:", float(full_test_eval_loss))
print(f"测试集准确率: {float(full_test_eval_accuracy) * 100:.2f}%")
print("W_in 的实际设备:", params["W_in"].device)
print("W_cls 的实际设备:", params["W_cls"].device)
print("W_in 的数据类型:", params["W_in"].dtype)
from datetime import datetime
from pathlib import Path

# 放在当前 Python 脚本所在项目的 outputs/checkpoints 目录。
checkpoint_dir = Path(__file__).resolve().parent / "outputs" / "checkpoints"
checkpoint_dir.mkdir(parents=True, exist_ok=True)

checkpoint_stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
checkpoint_path = checkpoint_dir / f"transformer_mnist_{checkpoint_stamp}.npz"

# 复制一份到 CPU 用于写文件；params 中的原数组继续留在 GPU。
checkpoint_arrays = {
    name: cp.asnumpy(value) for name, value in params.items()
}

# 同时保存以后推理需要的配置，全部使用普通数组。
checkpoint_arrays["parameter_names"] = np.asarray(list(params), dtype=np.str_)
checkpoint_arrays["position_encoding"] = cp.asnumpy(position_encoding)
checkpoint_arrays["num_heads"] = np.asarray(num_heads, dtype=np.int32)
checkpoint_arrays["ln_eps"] = np.asarray(float(ln_eps), dtype=np.float32)
checkpoint_arrays["pixel_divisor"] = np.asarray(255.0, dtype=np.float32)
checkpoint_arrays["format_version"] = np.asarray(1, dtype=np.int32)

# xb 表示新建文件；如果同名文件已存在，就拒绝覆盖。
with checkpoint_path.open("xb") as checkpoint_file:
    np.savez(checkpoint_file, **checkpoint_arrays)

print("参数已保存到:", checkpoint_path)
print("保存的参数数组数量:", len(params))
print("保存的参数标量总数:", sum(value.size for value in params.values()))
print("保存的位置编码形状:", checkpoint_arrays["position_encoding"].shape)
print("当前 W_in 的实际设备:", params["W_in"].device)
print("当前 W_cls 的实际设备:", params["W_cls"].device)

from pathlib import Path

# 明确读取你刚才保存的这个文件。
load_checkpoint_path = Path(
    "F:/PythonProjects/deep_learning/outputs/checkpoints/"
    "transformer_mnist_20261008_115104_418995.npz"
)

# 新加载的数组放在 GPU 0。
cp.cuda.Device(0).use()

with np.load(load_checkpoint_path, allow_pickle=False) as saved_checkpoint:
    if int(saved_checkpoint["format_version"].item()) != 1:
        raise ValueError("当前代码支持第 1 版参数文件")

    loaded_parameter_names = saved_checkpoint["parameter_names"].tolist()
    loaded_params = {
        name: cp.asarray(saved_checkpoint[name], dtype=cp.float32)
        for name in loaded_parameter_names
    }
    loaded_position_encoding = cp.asarray(
        saved_checkpoint["position_encoding"], dtype=cp.float32
    )
    loaded_num_heads = int(saved_checkpoint["num_heads"].item())
    loaded_ln_eps = cp.float32(saved_checkpoint["ln_eps"].item())
    loaded_pixel_divisor = float(saved_checkpoint["pixel_divisor"].item())

# 比较全部参数和固定位置编码。
loaded_parameters_match = all(
    bool(cp.array_equal(params[name], loaded_params[name]))
    for name in params
)
loaded_positions_match = bool(
    cp.array_equal(position_encoding, loaded_position_encoding)
)

# x_test 已经除过 255，这里直接使用，不再重复归一化。
load_sample_index = 0
load_target = int(y_test[load_sample_index])

load_loss_original, load_probabilities_original, load_cache_original = forward_one_image(
    x_test[load_sample_index], load_target, params,
    position_encoding, num_heads, ln_eps
)
load_loss_restored, load_probabilities_restored, load_cache_restored = forward_one_image(
    x_test[load_sample_index], load_target, loaded_params,
    loaded_position_encoding, loaded_num_heads, loaded_ln_eps
)

print("读取的文件:", load_checkpoint_path)
print("加载的参数数组数量:", len(loaded_params))
print("加载的参数标量总数:", sum(value.size for value in loaded_params.values()))
print("加载的位置编码形状:", loaded_position_encoding.shape)
print("加载的 W_in 的实际设备:", loaded_params["W_in"].device)
print("加载的 W_in 的数据类型:", loaded_params["W_in"].dtype)
print("全部参数是否与当前模型相同:", loaded_parameters_match)
print("位置编码是否相同:", loaded_positions_match)
print("真实标签:", load_target)
print("当前模型预测:", int(cp.argmax(load_probabilities_original)))
print("加载后模型预测:", int(cp.argmax(load_probabilities_restored)))
print("当前模型损失:", float(load_loss_original))
print("加载后模型损失:", float(load_loss_restored))
print("概率最大绝对差:", float(cp.max(cp.abs(
    load_probabilities_original - load_probabilities_restored
))))