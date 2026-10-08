# MNIST 手写 Transformer 建模记录

日期：2026-10-07（Asia/Shanghai）；版本：v005。
阶段：Q/K/V 多头拆分与逆变换验证。未计算注意力分数、softmax、损失、梯度或训练。

## 范围与当前输入状态

用户要求继续逐段手写教学；代码直接贴在聊天里。本轮第 5 步将 Q/K/V 各拆成 4 个头，每头 8 维，维持基础 CuPy 数组运算与后续手写反向、参数更新的计划。不修改用户主 `F:\PythonProjects\deep_learning\transformer_mnist.py`，不另行交付 Python 代码文件。

本轮实际检查的保存文件包含第 1～3 步，尚无上一条第 4 步的 Q/K/V 定义。助手已说明须先追加保存 v004 代码，再追加本步。不能把磁盘文件状态等同于用户编辑器里未保存的内容，也不能断言用户从未运行过第 4 步。

## 输入来源与处理（事实）

复用本地 MNIST 缓存 `C:\Users\19367\.keras\datasets\mnist.npz`，SHA256 `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`，与前版一致。训练集前 5000 图、测试集前 1000 图；像素 CuPy float32 除以 255，标签 int64，未新增下载、清洗、划分或数据增强。

验证在独立 Python 进程执行保存的前三步，再从 v004 验证记录取出当时逐字代码临时生成 Q/K/V；仅用于助手验证，未写回用户主文件。Q/K/V 均来自首图的 encoder_input `(28,32)`；初始化沿用 seed=42 后的连续随机流和 0.1 尺度，不执行优化更新，也不声称读取了用户实时进程的参数。

## 假设、变量与公式

序列长度 T=28、特征宽度 D=32、头数 H=4，每头特征 d=D/H=8。当前宽度可被头数整除。对 Q、K、V 分别执行相同的重排：

```text
(行, 总特征)       (28,32)
→ reshape: (行, 头, 每头特征) (28,4,8)
→ transpose(1,0,2): (头, 行, 每头特征) (4,28,8)
```

对应关系：`Q_heads[h,i,j] = Q[i,h*8+j]`，K/V 相同。每个头完整保留图片的 28 行，使用该行投影结果中的 8 个特征；这些是前一步学习式投影的特征，不是 8 个原始像素。四个头用于后续从不同投影子空间计算注意力，参照 [Transformer 原论文第 3.2.2 节](https://arxiv.org/html/1706.03762v7)。当前仅做形状重排，尚无注意力计算或学到的关系。

本步新增训练参数 0；逆变换为 `heads.transpose(1,0,2).reshape(seq_len,d_model)`，应恢复投影的全部数值和顺序。

## 本步代码

须放在上一条 Q/K/V 生成代码之后：

```python
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
```

## 方法选择理由（判断）

先解释特征拆分与轴顺序，再计算注意力分数，便于用户逐段核对维度。保留所有 28 行，之后每个头可比较任意两行。4 头和 8 维沿用教学配置，未证明最优。此步没有调用现成多头注意力层或自动求导。

## 实际执行与验证（事实）

通过 `F:\PythonProjects\deep_learning\.venv\Scripts\python.exe` 在 PowerShell 独立核验，未声称在用户 VS Code 终端点击运行。Python 3.13.15、NumPy 2.5.3、CuPy 14.2.0；可见 GPU 1 张，指定 GPU 0，型号 NVIDIA GeForce RTX 5060。本轮没有调整环境依赖。

Q_heads、K_heads、V_heads 均为 `(4,28,8)`、有限 float32、设备 `<CUDA Device 0>`；任一头为 `(28,8)`。把各个头与原投影相应的连续 8 维列切片逐元素比较，12 个头切片全部精确相同；对三组结果执行逆变换，全部精确恢复原 Q/K/V。原投影数据未改变，CUDA 同步完成。

当前保存文件是否包含 Q/K/V：`False`；验证进程是否临时补充 v004 片段：`True`。用户主文件执行前后 SHA256 一致：`2beb3767e6fe89aff84b52cc1a149c41ec84c222ae3e44a0f617dc0b6ac97681`。聊天片段 SHA256：`d9ebfd49426b7939ce31b03aaf3f941d1b39c00fa515f58c312da4e5738fde36`。

[本步代码](#本步代码) · [真实多头拆分验证结果](2026-10-07_transformer_head_split_validation_v005.json)。JSON 保存逐字代码片段、来源、输入准备边界、配置、形状、设备、逐头映射和逆变换结果及控制台输出。

## 相比上一版的变化

v004 生成完整 Q/K/V；v005 新增三个投影的多头重排与无损逆变换验证，数值与元素数量保持不变，新增参数为 0。保留此前全部版本与证据；项目 modeling_records/index.md 和本交付索引新增 2026-10-07 v005。本轮未重新训练模型。

## 未解决限制与下一步

尚未计算 QKᵀ、缩放分数、softmax 权重、加权 V、多头合并输出投影、残差、LayerNorm、前馈或分类头。优化更新次数为 0，没有训练损失、预测准确率或完整模型验证。维度重排本身不能证明多个头已经学习到不同关系，也不构成 GPU 训练或提速证据。

用户须先保存前一步 Q/K/V 片段，才能在其程序中使用本段。以后改变头数或宽度时须满足 d_model 可被 num_heads 整除，并检查元素顺序。下一步按每头维度 8 计算缩放点积注意力，分数形状预计 `(4,28,28)`；缩放使用 sqrt(8)。后续仍逐段手写梯度并做独立数值验证。
