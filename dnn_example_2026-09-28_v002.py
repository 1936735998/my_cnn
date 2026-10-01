import numpy as np
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import set_random_seed

# 固定随机种子，便于重复运行示例。
set_random_seed(42)

input_dim = 10
num_classes = 3

# 每个样本有 10 个特征，且只属于 0、1、2 中的一个类别。
X_train = np.random.rand(100, input_dim).astype(np.float32)
y_train = np.random.randint(0, num_classes, size=100)

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
# 输出这 100 个训练样本各自的预测类别。
probabilities = model.predict(X_train, verbose=0)
predicted_classes = np.argmax(probabilities, axis=1)

print("\n100 个样本的分类结果（类别编号：0、1、2）：")
for i, (predicted, label) in enumerate(zip(predicted_classes, y_train), start=1):
    print(f"样本 {i:03d}：预测类别 = {predicted}，训练标签 = {label}")
