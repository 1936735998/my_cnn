import numpy as np

def relu(x):

    return np.maximum(0,x)

# ======================
# 1. 输入
# ======================

# 假设输入只有3个数字

x = np.array(
    [
        1,
        2,
        3
    ]
)



# ======================
# 2. 初始化权重
# ======================

# 输入3个
# 输出2个

W = np.random.randn(3,2)


# 偏置

b = np.zeros(2)



print("权重:")
print(W)



# ======================
# 3. Forward
# ======================

y = np.dot(x,W)+b


print("输出:")
print(y)
y=relu(y)
print(y)
target=np.array([0,1])


loss=np.mean(
    (y-target)**2
)


print(loss)