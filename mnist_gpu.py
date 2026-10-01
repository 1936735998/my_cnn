import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# 定义生成器和判别器网络结构
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
class Discriminator(nn.Module):
def __init__(self, input_dim):
super(Discriminator, self).__init__()
self.model = nn.Sequential(
nn.Linear(input_dim, 128),
nn.ReLU(),
nn.Linear(128, 1),
nn.Sigmoid()
)
def forward(self, x):
return self.model(x)
# 实例化生成器和判别器对象
input_dim = 100 # 输入维度可根据实际需求调整
output_dim = 784 # 对于MNIST数据集，输出维度为28*28=784
gen = Generator(input_dim, output_dim)
disc = Discriminator(output_dim)
# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer_G = optim.Adam(gen.parameters(), lr=0.0002)
optimizer_D = optim.Adam(disc.parameters(), lr=0.0002)