"""Keras 3 LSTM 序列三分类示例：使用已安装的 PyTorch 后端，优先在 GPU 训练。"""

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
from keras import Input, Sequential
from keras.layers import LSTM, Dense


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


def lstm_model(input_shape=(20, 10), num_classes=3):
    """每个样本有 20 个时间步，每步 10 个特征；输出整段序列的类别。"""
    return Sequential([
        Input(shape=input_shape, name="sequence"),
        # 默认 return_sequences=False：每个样本输出最后时间步的 128 维表示。
        LSTM(units=128, name="lstm"),
        Dense(units=num_classes, activation="softmax", name="classifier"),
    ], name="lstm_classifier")


def main():
    # 这是随机序列演示；实际应用请替换为有真实标签的时间序列。
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
    # LSTM 输入顺序：(样本数, 时间步数, 每步特征数)。
    X_train = rng.random((100, 20, 10), dtype=np.float32)
    y_train = rng.integers(0, 3, size=100, dtype=np.int64)

    model = lstm_model(input_shape=X_train.shape[1:], num_classes=3)
    model.to(device)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    initial_kernel = model.trainable_variables[0].value.detach().clone()
    history = model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=2)
    weight_update_verified = not torch.equal(initial_kernel, model.trainable_variables[0].value.detach())
    assert weight_update_verified, "训练后 LSTM 输入权重未更新"
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
        "data_origin": "independent random demo sequences and labels",
        "input_shape": list(X_train.shape),
        "probabilities_shape": list(probabilities.shape),
        "software_output_checks": "passed",
        "weight_update_verified": weight_update_verified,
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
        "lstm_run_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json"
    )
    with result_file.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(f"本次结果已保存：{result_file}", flush=True)


if __name__ == "__main__":
    main()
