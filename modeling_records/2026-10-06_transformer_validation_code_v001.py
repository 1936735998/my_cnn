"""独立执行教学快照，验证真实数据与 CuPy GPU 后端；不执行模型训练。"""
import hashlib
import json
import platform
import runpy
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

here = Path(__file__).resolve().parent
source_name = "2026-10-06_transformer_step01_code_v001.py"
out_dir = here if (here / source_name).exists() else here.parent / "outputs" / "transformer_v001"
source = out_dir / source_name
namespace = runpy.run_path(str(source))
import cupy as cp
data_path = namespace["data_path"]

checks = {}
with np.load(data_path, allow_pickle=False) as data:
    raw_shapes = {name: list(data[name].shape) for name in data.files}
    raw_dtypes = {name: str(data[name].dtype) for name in data.files}
    for prefix, count in [("train", 5000), ("test", 1000)]:
        x = namespace[f"x_{prefix}"]
        y = namespace[f"y_{prefix}"]
        expected = data[f"x_{prefix}"][:count].astype(np.float32) / 255.0
        assert isinstance(x, cp.ndarray) and x.device.id == 0
        assert isinstance(y, cp.ndarray) and y.device.id == 0
        np.testing.assert_allclose(cp.asnumpy(x), expected, rtol=1e-7, atol=1e-7)
        np.testing.assert_array_equal(cp.asnumpy(y), data[f"y_{prefix}"][:count])
        assert x.shape == (count, 28, 28) and y.shape == (count,)
        assert bool(cp.all(cp.isfinite(x)))
        assert float(x.min()) >= 0 and float(x.max()) <= 1
        assert bool(cp.all((y >= 0) & (y < 10)))
        checks[prefix] = {
            "x_shape": list(x.shape), "y_shape": list(y.shape),
            "x_dtype": str(x.dtype), "y_dtype": str(y.dtype),
            "x_device": str(x.device), "y_device": str(y.device),
            "min": float(x.min()), "max": float(x.max()),
            "first_label": int(y[0]), "all_finite": True,
            "normalization_matches_cpu": True, "labels_preserved": True,
        }

# 用真实 GPU 运算核验矩阵乘法、广播、softmax 与手写梯度更新。
# 这是后端验证，不是 Transformer 前向或训练。
a = cp.ones((4, 4), dtype=cp.float32)
matrix = a @ a
np.testing.assert_array_equal(cp.asnumpy(matrix), np.full((4, 4), 4.0))
scores = cp.asarray([[1.0, 2.0, 3.0]], dtype=cp.float32)
exp_scores = cp.exp(scores - scores.max(axis=-1, keepdims=True))
probabilities = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
np.testing.assert_allclose(cp.asnumpy(probabilities.sum(axis=-1)), [1.0], atol=1e-6)
w = cp.asarray([1.0], dtype=cp.float32)
loss_before = 0.5 * (2 * w - 6) ** 2
gradient = (2 * w - 6) * 2
w_new = w - 0.1 * gradient
loss_after = 0.5 * (2 * w_new - 6) ** 2
assert float(loss_after[0]) < float(loss_before[0])
cp.cuda.runtime.deviceSynchronize()

gpu_props = cp.cuda.runtime.getDeviceProperties(0)
gpu_name = gpu_props["name"]
if isinstance(gpu_name, bytes):
    gpu_name = gpu_name.decode("utf-8")
hardware = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
    check=True, capture_output=True, text=True,
).stdout.strip()

result = {
    "date": "2026-10-06", "timezone": "Asia/Shanghai", "version": "v001",
    "recorded_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
    "scope": "GPU 数据准备与基础运算验证；未初始化或训练 Transformer",
    "python": sys.version.split()[0], "executable": sys.executable,
    "numpy": np.__version__, "cupy": cp.__version__,
    "reused_cuda_dll_directory": str(namespace["cuda_library_path"]),
    "cpu_identifier": platform.processor(),
    "hardware_report": hardware,
    "cupy_visible_gpu_count": cp.cuda.runtime.getDeviceCount(),
    "selected_device": 0, "gpu_name": gpu_name,
    "compute_capability": [gpu_props["major"], gpu_props["minor"]],
    "cuda_runtime_version": cp.cuda.runtime.runtimeGetVersion(),
    "cuda_driver_version": cp.cuda.runtime.driverGetVersion(),
    "source_file": source.name,
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "data_file": str(data_path),
    "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
    "raw_shapes": raw_shapes, "raw_dtypes": raw_dtypes,
    "checks": checks,
    "backend_smoke_check": {
        "device": str(matrix.device), "matmul_4x4_value": float(matrix[0, 0]),
        "softmax_row_sum": float(probabilities.sum()),
        "scalar_gradient": float(gradient[0]), "updated_scalar": float(w_new[0]),
        "scalar_loss_before": float(loss_before[0]), "scalar_loss_after": float(loss_after[0]),
        "is_transformer_training": False,
    },
    "transformer_parameters_initialized": False,
    "transformer_forward_executed": False,
    "transformer_training_executed": False,
}
result_path = out_dir / "2026-10-06_transformer_data_gpu_validation_v001.json"
result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print("真实 GPU 数据准备和基础计算检查通过:", result_path)
