import platform
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import set_random_seed


def get_cpu_name():
    """获取 CPU 型号；Windows 从系统硬件信息读取。"""
    name = platform.processor() or "未获取到 CPU 型号"
    if sys.platform == "win32":
        import winreg
        try:
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                name = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return name


def get_variable_device(variable):
    """兼容 Keras 包装变量和 TensorFlow 变量。"""
    value = getattr(variable, "value", variable)
    if callable(value):
        value = value()
    return value.device


# 固定随机种子，便于重复运行示例。
set_random_seed(42)

physical_devices = tf.config.list_physical_devices()
visible_gpus = tf.config.get_visible_devices("GPU")
training_device = "/GPU:0" if visible_gpus else "/CPU:0"
cpu_model = get_cpu_name()

print("\n========== 本次训练设备信息 ==========", flush=True)
print(f"Python 解释器：{sys.executable}", flush=True)
print(f"TensorFlow 版本：{tf.__version__}", flush=True)
print(f"CPU 型号：{cpu_model}", flush=True)
print("TensorFlow 检测到的物理设备：", flush=True)
for device in physical_devices:
    print(f"  {device.name}（{device.device_type}）", flush=True)
print(f"TensorFlow 可见 GPU 数量：{len(visible_gpus)}", flush=True)
if visible_gpus:
    gpu_details = tf.config.experimental.get_device_details(visible_gpus[0])
    gpu_model = gpu_details.get("device_name", "TensorFlow 未提供 GPU 型号")
    print(f"所选 GPU 型号：{gpu_model}", flush=True)
print(f"本次训练指定的主设备：{training_device}", flush=True)

# 某些辅助运算只能在 CPU 执行，允许 TensorFlow 为它们选择合适设备。
tf.config.set_soft_device_placement(True)
# 如需逐个查看运算的实际设备，可将 False 改为 True（输出较多）。
tf.debugging.set_log_device_placement(False)

input_dim = 10
num_classes = 3

# 每个样本有 10 个特征，且只属于 0、1、2 中的一个类别。
X_train = np.random.rand(100, input_dim).astype(np.float32)
y_train = np.random.randint(0, num_classes, size=100)

with tf.device(training_device):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])

    # 整数类别标签对应 sparse_categorical_crossentropy。
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=2)
    probabilities = model.predict(X_train, verbose=0)

actual_weight_devices = sorted({
    get_variable_device(variable) for variable in model.trainable_variables
})
predicted_classes = np.argmax(probabilities, axis=1)

print("\n100 个样本的分类结果（类别编号：0、1、2）：")
for i, (predicted, label) in enumerate(zip(predicted_classes, y_train), start=1):
    print(f"样本 {i:03d}：预测类别 = {predicted}，训练标签 = {label}")

print("\n========== 本次训练设备汇总 ==========", flush=True)
print(f"指定主设备：{training_device}", flush=True)
print("模型权重实际所在设备：" + ", ".join(actual_weight_devices), flush=True)
if visible_gpus:
    print(f"所选 GPU 型号：{gpu_model}", flush=True)
else:
    print(f"物理 CPU 型号：{cpu_model}", flush=True)
