# 手写 LSTM 第 7 步：保存各时间步的前向结果

日期：2026-10-06；版本：v007。
前置：用户已完成分类层梯度。此步只保存前向缓存，不执行新的 LSTM 反向或更新。

## 为什么要保存

当前循环中 combined、c_prev、f、i_gate、g、o、c、h 每一步都会重新赋值；循环结束后直接读取这些变量得到最后一步的值。
反向传播要从第 28 行向第 1 行计算，各步导数需要当时的门值、状态和拼接输入。因此用列表 cache 为每步保存一条字典记录，包含 8 个数组。

## 第一处：完整图片循环前初始化

找到“LSTM：读完整张图片并计算类别分数”部分。将 for 之前的状态初始化写成：

```python
h = np.zeros(hidden_size)
c = np.zeros(hidden_size)

cache = []
```

紧接着仍是原来的 for t in range(len(sequence))。cache=[] 放在循环外，一张图片开始时执行一次，不放入单步演示，也不放在循环里。

## 第二处：循环内保存

将上述 for 循环最后的 c/h 更新两句替换为下面这段，保持四个空格的循环内缩进：

```python
    c = f * c_prev + i_gate * g
    h = o * np.tanh(c)

    cache.append({
        "combined": combined.copy(),
        "c_prev": c_prev.copy(),
        "f": f.copy(),
        "i_gate": i_gate.copy(),
        "g": g.copy(),
        "o": o.copy(),
        "c": c.copy(),
        "h": h.copy()
    })
```

cache.append 每处理一行执行一次，把该步字典追加到列表。键是字符串，值是对应 NumPy 数组的副本。copy 保存独立快照；当前重新赋值不会原地改写旧数组，使用副本也便于防止以后原地修改影响缓存。

完整图片循环之后的 output、loss 和输出层梯度代码继续使用原逻辑。

## 第三处：文件末尾检查

在文件末尾追加，不缩进：

```python
print("缓存的时间步数:", len(cache))
print("第一步 combined:", cache[0]["combined"].shape)
print("最后一步 c:", cache[-1]["c"].shape)
print("最后一步 h:", cache[-1]["h"].shape)
```

实际预期输出：

```text
缓存的时间步数: 28
第一步 combined: (92,)
最后一步 c: (64,)
最后一步 h: (64,)
```

## 怎样取回数据

cache[0] 是第一步记录；cache[-1] 是最后一步记录；cache[27] 也是本张图片最后一步记录。
cache[0]["f"] 取回第一步的遗忘门；cache[-1]["h"] 取回最后一步的隐藏状态。
combined 前 28 个分量是当前行，后 64 个分量是旧 h，因此本方案无需额外保存 h_prev。c_prev 单独保存，用于求遗忘门的梯度。sigmoid/tanh 的导数可用门或候选输出计算，本阶段无需额外保存线性输入。
h 虽非门梯度必须字段，保留它便于检查状态连续性和取回末步分类输入。

## 实际验证与下一步

已在内存中将这两处插入实际用户代码，独立执行未缓存/已缓存两份首图前向与输出梯度；未修改用户主文件。
缓存长度 28，每步 8 个键、形状与有限值正确；各行输入、前一步 h/c 连续性、零初始状态、独立副本检查通过。末步缓存 h/c 与实际末态一致；原始 output、loss、输出层梯度及参数逐元素一致，loss 仍为 0.08934437001590566。
Python 3.13.15，NumPy 2.5.3，CPU float64；没有 BPTT、优化更新、训练或测试评价。

[建模记录](2026-10-06_lstm_mnist_v007.md) · [实际缓存验证 JSON](2026-10-06_lstm_cache_validation_v007.json) · [上一课](lstm_from_scratch_lesson_06.md)。

下一步取最后一条记录，计算损失对输出门与细胞状态的梯度，再逐步传播到其余门和上一时间步。
