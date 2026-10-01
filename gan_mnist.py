import torch #主要负责张量，自动求导等基础部分
import torch.nn as nn  # 神经网络模块
import torch.optim as optim  #优化器模块
from torch.utils.data import DataLoader #负责分批，打乱数据
from torchvision import datasets, transforms #datasets负责数据集的下载，transforms负责预处理等
import matplotlib.pyplot as plt #画图工具


# =========================
# 1. 定义生成器 Generator
# =========================
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),

            nn.Linear(128, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# =========================
# 2. 定义判别器 Discriminator
# =========================
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 128), #y=xW^T+b
            nn.ReLU(),

            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# =========================
# 3. 超参数
# =========================
input_dim = 100       # 随机噪声维度
output_dim = 784      # MNIST: 28 * 28
batch_size = 64
learning_rate = 0.0002
num_epochs = 20


# =========================
# 4. 设备
# =========================
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("当前使用设备：", device)


# =========================
# 5. 加载 MNIST 数据集
# =========================
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    transform=transform,
    download=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)


# =========================
# 6. 创建 Generator / Discriminator
# =========================
gen = Generator(input_dim, output_dim).to(device)
disc = Discriminator(output_dim).to(device)


# =========================
# 7. 损失函数
# =========================
criterion = nn.BCELoss()#二元交叉熵损失函数


# =========================
# 8. 优化器
# =========================
optimizer_G = optim.Adam(
    gen.parameters(),
    lr=learning_rate
)

optimizer_D = optim.Adam(
    disc.parameters(),
    lr=learning_rate
)


# =========================
# 9. 开始训练
# =========================
for epoch in range(num_epochs):

    for batch_idx, (real_images, _) in enumerate(train_loader):

        # -------------------------
        # 准备真实图片
        # -------------------------
        real_images = real_images.view(-1, 784).to(device)

        batch_size_current = real_images.size(0)

        # 真实标签 = 1
        real_labels = torch.ones(
            batch_size_current,
            1,
            device=device
        )

        # 假图片标签 = 0
        fake_labels = torch.zeros(
            batch_size_current,
            1,
            device=device
        )

        # =================================
        # 一、训练判别器 Discriminator
        # =================================

        # 清空梯度
        optimizer_D.zero_grad()

        # ---- 真实图片 ----
        real_output = disc(real_images)

        real_loss = criterion(
            real_output,
            real_labels
        )

        # ---- 生成假图片 ----
        noise = torch.randn(
            batch_size_current,
            input_dim,
            device=device
        )

        fake_images = gen(noise)

        fake_output = disc(
            fake_images.detach()
        )

        fake_loss = criterion(
            fake_output,
            fake_labels
        )

        # 判别器总损失
        loss_D = real_loss + fake_loss

        # 反向传播
        loss_D.backward()

        # 更新判别器参数
        optimizer_D.step()


        # =================================
        # 二、训练生成器 Generator
        # =================================

        optimizer_G.zero_grad()

        # 再生成一批假图片
        noise = torch.randn(
            batch_size_current,
            input_dim,
            device=device
        )

        fake_images = gen(noise)

        # 让判别器判断
        fake_output = disc(fake_images)

        # 生成器希望判别器认为这些是真图片
        loss_G = criterion(
            fake_output,
            real_labels
        )

        # 反向传播
        loss_G.backward()

        # 更新生成器
        optimizer_G.step()


        # =========================
        # 打印训练信息
        # =========================
        if batch_idx % 200 == 0:

            print(
                f"Epoch [{epoch + 1}/{num_epochs}] "
                f"Batch [{batch_idx}/{len(train_loader)}] "
                f"Loss_D: {loss_D.item():.4f} "
                f"Loss_G: {loss_G.item():.4f}"
            )


# =========================
# 10. 训练完成
# =========================
print("训练完成！")


# =========================
# 11. 生成图片
# =========================
with torch.no_grad():

    noise = torch.randn(
        16,
        input_dim,
        device=device
    )

    generated_images = gen(noise)

    generated_images = generated_images.view(
        -1, 28, 28
    ).cpu()


# =========================
# 12. 显示生成结果
# =========================
fig, axes = plt.subplots(4, 4, figsize=(8, 8))

for i, ax in enumerate(axes.flatten()):

    ax.imshow(
        generated_images[i],
        cmap="gray"
    )

    ax.axis("off")

plt.tight_layout()
plt.show()