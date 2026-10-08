# 手写 LSTM 第 10 步：本步参数贡献与前一状态梯度

日期：2026-10-06；版本：v010。
前置：用户已保存末步四种 d_z 与 d_c。本步只处理第28步，名字 _step 明确是该时间步贡献，不是整张图片共享参数的总梯度。

## 追加代码

```python
# ==================================================
# 最后一个时间步：参数梯度与上一时刻状态的梯度
# ==================================================

combined_t = step["combined"]

# 四组参数在本时间步的梯度贡献
d_W_f_step = np.outer(combined_t, d_z_f)
d_b_f_step = d_z_f.copy()

d_W_i_step = np.outer(combined_t, d_z_i)
d_b_i_step = d_z_i.copy()

d_W_g_step = np.outer(combined_t, d_z_g)
d_b_g_step = d_z_g.copy()

d_W_o_step = np.outer(combined_t, d_z_o)
d_b_o_step = d_z_o.copy()

# 四条门计算路径共同传回 combined
d_combined = (
    np.dot(d_z_f, W_f.T)
    + np.dot(d_z_i, W_i.T)
    + np.dot(d_z_g, W_g.T)
    + np.dot(d_z_o, W_o.T)
)

# combined = [当前输入, 上一步 h]
d_h_prev = d_combined[input_size:]

# 旧记忆通过 f * c_prev 传到当前 c
d_c_prev = d_c * f_t

print("d_W_f_step:", d_W_f_step.shape)
print("d_W_i_step:", d_W_i_step.shape)
print("d_W_g_step:", d_W_g_step.shape)
print("d_W_o_step:", d_W_o_step.shape)
print("d_b_f_step:", d_b_f_step.shape)

print("d_combined:", d_combined.shape)
print("d_h_prev:", d_h_prev.shape)
print("d_c_prev:", d_c_prev.shape)

print("d_h_prev 前 5 个值:", d_h_prev[:5])
print("d_c_prev 前 5 个值:", d_c_prev[:5])
```

## 参数梯度与输出层的对应

四组线性映射都是 z=combined W+b。因而本步 d_W=outer(combined,d_z)，d_b=d_z。
combined=(92,)，d_z=(64,)，外积得到 (92,64)。四组偏置梯度均为 (64,)。
_step 贡献后续需对28步逐项累加；共享权重在逆序计算期间保持不变，全部梯度完成后才更新。

## 四路合并与两个状态路径

combined 同时参与四个门计算，所以它的总局部梯度是四个 d_z @ W.T 的和，形状 (92,)。
拼接顺序是 [x,h_prev]，前28个分量对应本行输入，后64个分量对应旧隐藏状态。d_combined[input_size:] 因而就是 d_h_prev。
旧细胞状态通过 c=f*c_prev+i*g 的直接记忆路径进入当前状态，独立局部导数 d_c_prev=d_c*f。它不会被 d_combined 的切片取代。
此处将旧 h 和旧 c 作为当前步接口的独立输入求偏导；它们在上一时间步内部的依赖关系将在下一轮反向继续处理。

## 实际结果与验证

四个本步权重梯度均为 (92,64)，四个偏置梯度均为 (64,)；d_combined=(92,)，d_h_prev/d_c_prev=(64,)。
首五项实测：
- d_h_prev [3.69180080e-3, -1.91982617e-3, 2.84663684e-3, -1.12339956e-3, -1.92319936e-3]；
- d_c_prev [-5.51668161e-5, 2.81036869e-3, 3.79443374e-3, 2.10712999e-4, -6.76813118e-4]。

项目 Python 3.13.15、NumPy 2.5.3，CPU float64。中央差分 epsilon=1e-6、绝对容差1e-8；固定缓存当前 combined/c_prev，只重算当前步和分类/MSE。每组 W 抽查6项（含输入区零值及隐藏区非零贡献），四组偏置各64项、combined全部92项、c_prev全部64项检查通过；隐藏切片也与数值导数后64项一致。最大误差约2.18e-11。
这些检查仅验证本时间步局部贡献，未扰动此前28步中的同一共享参数来声称完整序列总梯度已核验。
原参数、h、末步缓存保持原值，助手未编辑主 Python 文件，无优化更新、训练或测试评价。

[建模记录](2026-10-06_lstm_mnist_v010.md) · [真实本步参数/状态梯度 JSON](2026-10-06_lstm_step_parameter_state_gradient_validation_v010.json) · [上一课](lstm_from_scratch_lesson_09.md)。

下一步从 cache[-1] 到 cache[0] 逆序执行这些公式，传递前一状态梯度，并累加各步对共享参数的贡献。
