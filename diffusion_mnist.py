import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# =========================================================
# 1. Diffusion 模型
# =========================================================

class DiffusionModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_timesteps):
        super(DiffusionModel, self).__init__()

        self.num_timesteps = num_timesteps

        # 每一个时间步 t 使用一个简单的网络
        self.noise_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim)
            )
            for _ in range(num_timesteps)
        ])

    def forward(self, x, t):

        # t 是当前时间步
        # 例如 t = 100，就使用第 100 个网络
        noise_prediction = self.noise_predictors[t](x)

        return noise_prediction


# =========================================================
# 2. 参数
# =========================================================

input_dim = 784       # 28 × 28
hidden_dim = 128
num_timesteps = 1000

batch_size = 64
learning_rate = 1e-3
epochs = 5


# =========================================================
# 3. 加载 MNIST
# =========================================================

transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True
)


# =========================================================
# 4. 初始化模型
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = DiffusionModel(
    input_dim,
    hidden_dim,
    num_timesteps
).to(device)


# =========================================================
# 5. 损失函数和优化器
# =========================================================

criterion = nn.MSELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=learning_rate
)


# =========================================================
# 6. 定义 Diffusion 的噪声调度
# =========================================================

beta = torch.linspace(
    1e-4,
    0.02,
    num_timesteps
).to(device)

alpha = 1.0 - beta

alpha_bar = torch.cumprod(
    alpha,
    dim=0
)


# =========================================================
# 7. 给图片加噪声
# =========================================================

def add_noise(x, t):

    """
    x: 原始图片
    t: 时间步

    返回：
    noisy_x: 加噪后的图片
    noise: 真正加入的噪声
    """

    noise = torch.randn_like(x)

    # 取当前时间步对应的 alpha_bar
    sqrt_alpha_bar = torch.sqrt(
        alpha_bar[t]
    ).unsqueeze(1)

    sqrt_one_minus_alpha_bar = torch.sqrt(
        1 - alpha_bar[t]
    ).unsqueeze(1)

    # DDPM 的核心公式
    noisy_x = (
        sqrt_alpha_bar * x
        +
        sqrt_one_minus_alpha_bar * noise
    )

    return noisy_x, noise


# =========================================================
# 8. 训练
# =========================================================

for epoch in range(epochs):

    for batch_idx, (images, labels) in enumerate(dataloader):

        # -------------------------------------------------
        # 原始图片
        # -------------------------------------------------

        images = images.view(
            images.size(0),
            -1
        ).to(device)

        # -------------------------------------------------
        # 随机选择时间步
        # -------------------------------------------------

        t = torch.randint(
            0,
            num_timesteps,
            (images.size(0),),
            device=device
        )

        # -------------------------------------------------
        # 给图片添加噪声
        # -------------------------------------------------

        noisy_images, noise = add_noise(
            images,
            t
        )

        # -------------------------------------------------
        # 模型预测噪声
        # -------------------------------------------------

        # 这里因为一个 batch 中每张图片的 t
        # 可能不同，所以逐个时间步处理

        noise_predictions = torch.zeros_like(
            noise
        )

        for timestep in range(num_timesteps):

            mask = (t == timestep)

            if mask.any():

                noise_predictions[mask] = model(
                    noisy_images[mask],
                    timestep
                )

        # -------------------------------------------------
        # 计算 Loss
        # -------------------------------------------------

        loss = criterion(
            noise_predictions,
            noise
        )

        # -------------------------------------------------
        # 反向传播
        # -------------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # -------------------------------------------------
        # 打印
        # -------------------------------------------------

        if batch_idx % 200 == 0:

            print(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"Batch [{batch_idx}/{len(dataloader)}] "
                f"Loss: {loss.item():.4f}"
            )

x = torch.randn(1, 784).to(device)