"""Keras 3 残差网络示例：使用已安装的 PyTorch 后端，优先在 GPU 训练。"""

import os

# 必须在导入 keras 之前选择后端；这里只设置当前 Python 进程。
os.environ["KERAS_BACKEND"] = "torch"

import json
import platform
import random
import sys
from datetime import datetime
from pathlib import Path

import keras
import numpy as np
import torch
from keras import Input, Model
from keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv2D,
    Dense,
    GlobalAveragePooling2D,
)


def get_cpu_name():
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


def residual_block(inputs, filters, stride=1, name=None):
    """残差块：输出为 ReLU(F(inputs) + shortcut(inputs))。"""
    def layer_name(suffix):
        return f"{name}_{suffix}" if name else None

    shortcut = inputs

    # 主分支：两个 3×3 卷积；第二次激活放在残差相加之后。
    x = Conv2D(
        filters, (3, 3), strides=stride, padding="same", use_bias=False,
        name=layer_name("conv1"),
    )(inputs)
    x = BatchNormalization(name=layer_name("bn1"))(x)
    x = Activation("relu", name=layer_name("relu1"))(x)

    x = Conv2D(
        filters, (3, 3), padding="same", use_bias=False,
        name=layer_name("conv2"),
    )(x)
    x = BatchNormalization(name=layer_name("bn2"))(x)

    # Add 要求两条分支形状相同；必要时用 1×1 卷积调整通道和尺寸。
    if inputs.shape[-1] != filters or stride != 1:
        shortcut = Conv2D(
            filters, (1, 1), strides=stride, padding="same", use_bias=False,
            name=layer_name("shortcut_conv"),
        )(shortcut)
        shortcut = BatchNormalization(name=layer_name("shortcut_bn"))(shortcut)

    x = Add(name=layer_name("add"))([x, shortcut])
    return Activation("relu", name=layer_name("out"))(x)


def build_model(input_shape=(32, 32, 3), num_classes=3):
    inputs = Input(shape=input_shape, name="image")
    x = Conv2D(16, (3, 3), padding="same", use_bias=False, name="stem_conv")(inputs)
    x = BatchNormalization(name="stem_bn")(x)
    x = Activation("relu", name="stem_relu")(x)

    x = residual_block(x, filters=16, name="block1")
    x = residual_block(x, filters=32, stride=2, name="block2")
    x = GlobalAveragePooling2D(name="average_pool")(x)
    outputs = Dense(num_classes, activation="softmax", name="classifier")(x)
    return Model(inputs=inputs, outputs=outputs, name="small_resnet")


def main():
    # 这是随机数据演示；实际应用请替换为有真实标签的图像。
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cpu_name = get_cpu_name()
    gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]

    print("\n========== 本次训练设备信息 ==========", flush=True)
    print(f"Python 解释器：{sys.executable}", flush=True)
    print(f"Keras 版本：{keras.__version__}，后端：{keras.backend.backend()}", flush=True)
    print(f"PyTorch 版本：{torch.__version__}，CUDA 构建版本：{torch.version.cuda}", flush=True)
    print(f"物理 CPU 型号：{cpu_name}", flush=True)
    print(f"PyTorch 可见 GPU：{gpu_names or '未检测到'}", flush=True)
    print(f"本次训练主设备：{device}", flush=True)

    rng = np.random.default_rng(seed)
    X_train = rng.random((100, 32, 32, 3), dtype=np.float32)
    y_train = rng.integers(0, 3, size=100, dtype=np.int64)

    model = build_model()
    model.to(device)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    history = model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=2)
    probabilities = model.predict(X_train, batch_size=16, verbose=0)
    predicted_classes = np.argmax(probabilities, axis=1)

    # 验证输出格式；这些检查不代表模型具有真实预测能力。
    assert probabilities.shape == (100, 3)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)
    actual_devices = sorted({str(variable.value.device) for variable in model.trainable_variables})

    print("\n100 个样本的分类结果（类别编号：0、1、2）：")
    for i, (predicted, label) in enumerate(zip(predicted_classes, y_train), start=1):
        print(f"样本 {i:03d}：预测类别 = {predicted}，训练标签 = {label}")

    print("\n========== 本次训练设备汇总 ==========", flush=True)
    print(f"模型参数实际所在设备：{', '.join(actual_devices)}", flush=True)
    if device.type == "cuda":
        print(f"本次使用的物理 GPU：{torch.cuda.get_device_name(device)}", flush=True)
    else:
        print(f"本次使用的物理 CPU：{cpu_name}", flush=True)

    result = {
        "script_version": "v001",
        "time": datetime.now().astimezone().isoformat(),
        "python_executable": sys.executable,
        "keras_version": keras.__version__,
        "backend": keras.backend.backend(),
        "torch_version": torch.__version__,
        "cuda_build_version": torch.version.cuda,
        "selected_device": str(device),
        "actual_parameter_devices": actual_devices,
        "cpu_name": cpu_name,
        "visible_gpu_names": gpu_names,
        "seed": seed,
        "data_origin": "independent random demo images and labels",
        "input_shape": list(X_train.shape),
        "probabilities_shape": list(probabilities.shape),
        "software_output_checks": "passed",
        "model_parameter_count": model.count_params(),
        "epochs_completed": len(history.history["loss"]),
        "training_history": {key: [float(v) for v in values] for key, values in history.history.items()},
        "post_training_accuracy_on_training_samples": float(np.mean(predicted_classes == y_train)),
        "predicted_class_counts": np.bincount(predicted_classes, minlength=3).tolist(),
        "training_labels": y_train.tolist(),
        "predicted_classes": predicted_classes.tolist(),
        "probabilities": probabilities.tolist(),
        "generalization_validation": "not performed",
    }
    result_file = Path(__file__).resolve().parent / (
        "resnet_run_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json"
    )
    with result_file.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(f"本次结果已保存：{result_file}", flush=True)


if __name__ == "__main__":
    main()
