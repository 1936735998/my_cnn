# MNIST 手写 Transformer 建模记录

日期：2026-10-08（Asia/Shanghai）；版本：v050。
阶段：实际读取用户已保存的参数文件到GPU，并执行两次单图前向作比较。本步没有新增训练，没有重新评估训练全集或测试全集；不是仅文档更新。

## 问题范围、输入来源与处理（事实）

用户报告v049参数保存到 F:\PythonProjects\deep_learning\outputs\checkpoints\transformer_mnist_20261008_115104_418995.npz，20参数数组、9802标量，固定位置编码28×32，原参数继续在GPU0。当前教学步骤读取用户明确报告的这个实际文件，恢复独立的GPU参数、位置编码与推理配置，再对同一图比较当前模型及加载模型。代码贴聊天，用户自行接在现有脚本后面；核验没有编辑或运行用户主文件。

用户实际npz在核验前存在，大小50006 bytes，SHA256 349eff0650c37163be783b58d4f39d256b796091884c624428f131d2a320f218。实际以allow_pickle=False打开26个成员，逐项核对20个参数的shape、float32 dtype及字节SHA256，全部与v048真实训练后的after快照一致；位置编码与v048哈希一致，6项推理配置的值及dtype通过。没有重新保存原用户文件，没有生成另一个npz冒充用户保存结果。记录中的存档附件是该实际文件的逐字节只读复制。

用于对照的原params在独立进程中从v048真实GPU训练证据的20项parameter_snapshots[name].after恢复；每项转换float32并核对after_sha256后复制GPU。此原模型恢复与用户文件读取彼此独立，没有随机初始化或训练。核验无法读取用户终端内存，不能将本进程的params称为用户终端变量；它与用户实际文件全部参数逐字节相同是本次检查结果。

v048证据SHA256 7a0d42e1e0e7a65e2693c37a14fe853ff9330d961801a94f5dc60a81d115ef12；原主文件 F:\PythonProjects\deep_learning\transformer_mnist.py SHA256 21c209e8f710f229f895436090d19e80d10f3a6aed6d1e2e6f43bb22102a5154。只通过AST执行主文件从开头到import cupy的CUDA库启动段、6条原位置编码语句、头数和LN epsilon赋值、原forward_one_image定义。forward源码逐字等于v048已验证版本。执行前后主文件、用户npz、v048/v049有关旧记录SHA256保持一致。

原MNIST数据 C:\Users\19367\.keras\datasets\mnist.npz，SHA256 731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1；只加载前1000张测试图，以GPU float32除255，标签int64。数组哈希精确匹配v048。测试输入已经归一化，本片段直接使用x_test，不再除255；没有新增清洗、增强或划分。训练集未重新加载或参与计算。本步只推理测试索引0，真实标签7。

## 假设、变量与公式

模型沿用28行token、输入28、d_model32、4头各8维、前馈64、单post-LN编码器、固定位置编码、均值池化、10分类。20参数数组共9802标量。保存文件内另有位置编码28×32 float32、num_heads=4 int32、LN epsilon=float32(1e-5)、pixel_divisor=255 float32、format_version=1 int32、Unicode parameter_names列表，共26个非object成员。当前兼容性假设是文件来自此前同一结构、forward定义和预处理规则。

~~~text
CPU_arrays = np.load(actual_user_checkpoint, allow_pickle=False)
loaded_params[name] = cp.asarray(CPU_arrays[name], dtype=cp.float32)
loaded_position_encoding = cp.asarray(CPU_arrays["position_encoding"], dtype=cp.float32)
X_test 已经归一化，无第二次像素除法。
比较：params[name] == loaded_params[name]；PE == loaded_PE。
原loss, 原P, 原cache = forward(X, original_params, original_configuration)
加载loss, 加载P, 加载cache = forward(X, loaded_params, loaded_configuration)
预测 = argmax(P)；概率最大绝对差 = max(abs(原P - 加载P))。
本步不计算梯度，不执行SGD。
~~~

loaded_params及loaded_position_encoding是独立GPU数组。原params、位置编码和测试数据的字典/数组身份、存储地址和内容保留，加载后的数组不覆盖原参数。

## 方法选择理由（判断）

普通npz与NumPy/CuPy数组直接兼容，适合本教学模型的少量参数。显式读取文件里的头数、epsilon和位置编码，使恢复模型使用保存时的推理配置。allow_pickle=False可直接加载本文件的普通数值与Unicode数组。先保留原模型，再用独立loaded_params对同一图进行检查，能在调用前后比较全部参数，并验证完整概率向量和损失一致。

沿用一段可执行代码的教学节奏，本步没有改变网络结构、训练算法或优化器。读取存档后进行推理应复用已经定义的forward_one_image；参数文件没有Python函数实现。

## 本步代码

~~~python
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
~~~

## 实际执行、验证与结果（事实）

解释器 F:\PythonProjects\deep_learning\.venv\Scripts\python.exe，Python 3.13.15、NumPy 2.5.3、CuPy 14.2.0；GPU0 NVIDIA GeForce RTX 5060，可见GPU数量1。精确聊天片段在独立核验进程中读取了用户实际文件，没有写回或替代该文件。读取前后的SHA256均为349eff0650c37163be783b58d4f39d256b796091884c624428f131d2a320f218。

实际输出：

~~~text
读取的文件: F:\PythonProjects\deep_learning\outputs\checkpoints\transformer_mnist_20261008_115104_418995.npz
加载的参数数组数量: 20
加载的参数标量总数: 9802
加载的位置编码形状: (28, 32)
加载的 W_in 的实际设备: <CUDA Device 0>
加载的 W_in 的数据类型: float32
全部参数是否与当前模型相同: True
位置编码是否相同: True
真实标签: 7
当前模型预测: 7
加载后模型预测: 7
当前模型损失: 0.005684994161128998
加载后模型损失: 0.005684994161128998
概率最大绝对差: 0.0
~~~

全部20个加载参数是float32、有限值、GPU0数组，形状与原参数一致。20项参数和固定位置编码逐字节相同；加载后的数组均使用独立存储，原params及其20项参数的对象身份、存储地址和内容保留。原位置编码及测试数据也没有改变。

本版实际执行2次forward：相同测试第0图用原params一次、用loaded_params一次。真实标签7，原预测7，加载后预测7。原loss=0.005684994161128998，加载loss=0.005684994161128998，全部10类概率及loss逐位完全相同，概率最大绝对差0.0；真实类别概率0.9943311214447021。这些单图值也与v048该样本日志逐位一致。未调用backward或SGD，新增参数更新0，未运行主文件或重放此前训练。

本步执行的AST没有cp.random引用，公共随机生成、seed和随机状态API在核验期间加禁止调用guard，调用数0，结束时恢复原函数。没有获取、初始化、保存或恢复随机状态。没有进行新的有限差分或多图/全集评估。

训练准确率89.24%及测试准确率87.50%为v048历史固定评估结果，本步没有重新计算。实际验证范围是用户文件26个成员的全部内容与配置、20组独立GPU参数、1张测试图的两次完整前向；不能把1张图的正确预测当作重新验证1000图准确率。

JSON保存用户文件路径/大小/哈希与成员信息、原权重恢复来源及逐项SHA256、精确聊天代码/哈希、原forward/环境启动/位置编码/配置源码、原数据哈希、原数组保护结果、完整10类概率与loss对照、历史样本对照、环境及stdout。[本步代码](#本步代码) · [用户参数文件读取及GPU推理证据](2026-10-08_transformer_checkpoint_load_validation_v050.json) · [实际用户存档字节副本](checkpoints/transformer_mnist_20261008_115104_418995.npz)。相对链接已核验，无新增图。

## 与前版变化、未解限制及下一步

v049教学保存推理参数，用户已实际保存并报告文件路径。本步教学读取这个指定文件，恢复GPU参数与配置，并同图比较加载前后结果，不增加优化步骤或改变模型。保留所有旧记录，索引追加v050。同步阶段只复制已生成Markdown、JSON和用户存档字节副本，不执行模型，不重新保存用户原文件。

后台原params来自v048真实核验快照，与用户终端内存隔离；本步已确认用户实际存档与该快照逐项字节相同。用户执行片段时，若其当前params仍是保存时参数，两个模型比较应一致；此条件不能推广到之后又修改了原params的会话。

存档只含推理参数和配置，没有Python函数、数据集、训练进度、优化器状态或随机数状态。loaded_pixel_divisor本步只读取并记录，因x_test已处理过，不再次使用；后续读取新原始图像时需依保存的除数处理一次。模型仍只训练原前5000张、历史测试指标仅覆盖原前1000张，不能称为完整MNIST基准或收敛最优值。

下一步可教学使用loaded_params进行独立单图预测、查看10类概率和预测错误样本。后续独立运行推理需准备同样的forward函数及原始输入处理，不必重训练。
