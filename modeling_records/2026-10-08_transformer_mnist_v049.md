# MNIST 手写 Transformer 建模记录

日期：2026-10-08（Asia/Shanghai）；版本：v049。
阶段：实际保存训练后的参数，并读取到GPU核验。本版没有新增训练，没有重新评估训练全集或测试全集；不是仅文档更新。

## 问题范围、输入来源与处理（事实）

用户报告v048新增15000次更新后，5000张训练图片loss 0.34107816219329834、准确率89.24%，1000张测试图片loss 0.4118737280368805、准确率87.50%，参数为GPU0 float32。本版继续教学：将当前20参数数组、固定位置编码及推理配置保存为普通NumPy npz文件。代码仅贴聊天，用户自行接到脚本末尾，不编辑用户主文件。加载及预测对照是本版后台核验，下一步再逐项教学加载。

核验使用v048真实训练证据中全部20项 `parameter_snapshots[name].after`。逐项转换float32，检查形状和 `after_sha256` 后复制到GPU0，精确恢复实际训练后的9802参数标量。没有重跑主文件、随机初始化、重seed、15000步训练或6000图评估。恢复的是先前独立核验进程已记录的实际参数，不是读取用户当前终端内存。

v048证据SHA256 `7a0d42e1e0e7a65e2693c37a14fe853ff9330d961801a94f5dc60a81d115ef12`。主文件 `F:\PythonProjects\deep_learning\transformer_mnist.py` 当前SHA256 `0adb35937a5543f38218aaa45230a516fa053a238df20f5bf6e2c241fa9b77b7`。只通过AST执行原CUDA库启动段（到import cupy）、原位置编码六条语句、头数与LN epsilon赋值和原forward函数定义；forward源码逐字等于v048已验证版本。本版执行前后主文件及v048证据均未改变。

输入为本地MNIST `C:\Users\19367\.keras\datasets\mnist.npz`，文件SHA256 `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`；使用原前1000张测试图，以GPU float32除255、标签int64，两个数组哈希均精确匹配v048。无新增清洗、增强或划分。用原语句构造28×32位置编码，哈希 `5ff07736b815787e93c5b28bac57b784603e51c397187bb46de95ded4e0363bb` 精确匹配v048。训练集未重新加载、未参与计算。本版推理只使用测试索引0、真实标签7。

## 假设、变量与公式

模型沿用28行token、输入28、d_model32、4头各8维、前馈64、单post-LN编码器、固定位置编码、均值池化、10分类。20参数数组共9802标量。位置编码28×32 float32；num_heads=4 int32；LN epsilon=float32(1e-5)；pixel_divisor=255 float32；format_version=1 int32；parameter_names为Unicode字符串数组。20参数加6项配置共26个npz成员，没有object数组。

```text
CPU_parameter[name] = asnumpy(GPU_parameter[name])
np.savez(file, parameter_arrays + inference_configuration)
GPU_loaded[name] = cp.asarray(np.load(file, allow_pickle=False)[name])
要求：每项原参数和加载参数的shape、dtype、bytes完全相同。
要求：同一图下loss_loaded = loss_original，P_loaded = P_original。
预测类别 = argmax(P)。本版不计算梯度、不做SGD。
```

文件通过Path(__file__).resolve().parent定位当前脚本所在项目；保存到outputs/checkpoints，文件名带日期、时分秒及微秒。以xb创建，已有同名文件时拒绝覆盖。CPU副本用于持久化，原参数继续留在GPU，不用CPU副本替换params。

## 方法选择理由（判断）

普通npz可直接用NumPy读取，适合当前少量数组及配置，不依赖框架checkpoint对象。显式保存位置编码和像素除数可在后续推理时恢复相同输入处理，保存头数与epsilon可恢复相同计算配置。保存参数名列表便于逐项加载。本版先教保存，再教加载，保持一步一段可执行代码的节奏。

日期与微秒降低命名碰撞概率，xb进一步防止覆盖。这种文件用于已定义forward函数的推理参数恢复；它不包含完整训练恢复状态，不保证接续训练时随机顺序逐位相同。

## 本步代码

```python
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
```

## 实际执行、核验与结果（事实）

解释器 `F:\PythonProjects\deep_learning\.venv\Scripts\python.exe`，Python 3.13.15、NumPy 2.5.3、CuPy 14.2.0；GPU0 NVIDIA GeForce RTX 5060，可见GPU数 1。精确聊天片段真实写入 `F:\PythonProjects\deep_learning\outputs\checkpoints\transformer_mnist_20261008_112805_701931.npz`，文件 50006 bytes，SHA256 `349eff0650c37163be783b58d4f39d256b796091884c624428f131d2a320f218`。

真实输出：

```text
参数已保存到: F:\PythonProjects\deep_learning\outputs\checkpoints\transformer_mnist_20261008_112805_701931.npz
保存的参数数组数量: 20
保存的参数标量总数: 9802
保存的位置编码形状: (28, 32)
当前 W_in 的实际设备: <CUDA Device 0>
当前 W_cls 的实际设备: <CUDA Device 0>
```

实际np.load以allow_pickle=False读取26个成员。全部成员是非object普通数组；20参数与位置编码的字节、形状、dtype完全保真，所有元数据的dtype和值核对通过。另用cp.asarray创建独立GPU参数和位置编码，原params对象及各参数身份、存储地址、内容保留，未覆盖原参数。测试数据和位置编码同样保持内容、身份、存储地址。

后台只执行2次单图forward：原参数一次、加载参数一次；未调用backward或SGD。本次测试第0张图片真实标签 7，原预测 7，加载后预测 7；原loss 0.005684994161128998，加载loss 0.005684994161128998，loss和全部10类概率逐位完全相等。与v048该样本日志核对预测、loss及真实类别概率，在rtol=2e-6、atol=2e-6下通过；是否与历史值逐位相同在JSON单独记录。本版往返检查范围是1张图片，未声称6000图重新评估或CPU参考差分验证。

静态检查本版执行的AST不引用cp.random，同时公共随机生成、seed与随机状态API在核验期间设禁止调用guard，调用数0，之后原函数恢复。没有为核验取得或初始化随机状态。文件不保存随机数状态；保存操作不包含随机采样或参数更新。

v048的训练准确率89.24%及测试准确率87.50%属于历史结果引用，本版没有重新计算这些全集指标。本版实际结果是参数持久化保真和单图GPU读取推理一致。JSON保存精确代码及哈希、主文件/数据/v048证据哈希、forward/位置编码/配置原AST源码、全部成员哈希与dtype、真实文件路径/大小/哈希、原参数保护检查、单图完整概率与loss对照、历史对照、环境和stdout。

[本步代码](#本步代码) · [实际保存、读取及GPU对照证据](2026-10-08_transformer_checkpoint_validation_v049.json) · [训练后参数存档副本](checkpoints/transformer_mnist_20261008_112805_701931.npz)。存档副本与项目outputs/checkpoints实际文件字节相同；相对链接已核验，无新增图。

## 与前版变化、未解限制及下一步

v048完成5000张图片三轮训练及最终固定评估，但参数只在进程内。本版将其20参数、位置编码及推理配置真正保存到npz并完成GPU往返读写核验，不新增优化步骤。旧版本保留；项目记录索引追加v049，同步阶段只复制已生成Markdown、JSON和存档证据副本，不执行模型。

核验恢复自v048真实参数快照，与用户终端内存隔离；用户执行聊天片段保存的是其当时params中的值。训练范围仍是原前5000张、测试成绩为原前1000张，不能称为完整MNIST基准。文件不保存Python函数、自动梯度、数据集、训练轮次进度或随机数状态；加载需已有兼容forward函数，完整继续训练可能需要另外记录训练进度与随机状态。文件只包含推理参数和配置，不用作当前用户主脚本的替代文件。

下一步教学用allow_pickle=False读取参数、恢复GPU数组，再对同一图片比较保存前后预测。后续可检查预测错误样本或学习手写批量计算。
