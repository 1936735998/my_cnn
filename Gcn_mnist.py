import torch
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv


# ============================================================
# 1. 检查训练设备
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("当前使用设备：", device)

if torch.cuda.is_available():
    print("GPU名称：", torch.cuda.get_device_name(0))


# ============================================================
# 2. 加载 Cora 数据集
# ============================================================

dataset = Planetoid(
    root="./data/Cora",
    name="Cora"
)

print("数据集名称：", dataset.name)
print("图数量：", len(dataset))
print("节点数量：", dataset[0].num_nodes)
print("节点特征数量：", dataset.num_features)
print("类别数量：", dataset.num_classes)


# ============================================================
# 3. 获取 Cora 图数据
# ============================================================

data = dataset[0].to(device)

print("节点特征 shape：", data.x.shape)
print("边索引 shape：", data.edge_index.shape)


# ============================================================
# 4. 定义 GCN 模型
# ============================================================

class GNN(torch.nn.Module):

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels
    ):
        super(GNN, self).__init__()

        # 第一层 GCN
        self.conv1 = GCNConv(
            in_channels,
            hidden_channels
        )

        # 第二层 GCN
        self.conv2 = GCNConv(
            hidden_channels,
            out_channels
        )

    def forward(self, data):

        # 获取节点特征和图结构
        x = data.x
        edge_index = data.edge_index

        # 第一层图卷积
        x = self.conv1(x, edge_index)

        # 激活函数
        x = F.relu(x)

        # Dropout
        x = F.dropout(
            x,
            p=0.5,
            training=self.training
        )

        # 第二层图卷积
        x = self.conv2(x, edge_index)

        # 输出 log 概率
        return F.log_softmax(x, dim=1)


# ============================================================
# 5. 设置超参数
# ============================================================

num_epochs = 200
learning_rate = 0.01
hidden_channels = 16

out_channels = dataset.num_classes


# ============================================================
# 6. 创建模型
# ============================================================

model = GNN(
    in_channels=dataset.num_features,
    hidden_channels=hidden_channels,
    out_channels=out_channels
).to(device)


# ============================================================
# 7. 定义优化器
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
    weight_decay=5e-4#权重衰减，用于减少过拟合
)


# ============================================================
# 8. 开始训练
# ============================================================

model.train()

for epoch in range(1, num_epochs + 1):

    # --------------------------------------------------------
    # 清零梯度
    # --------------------------------------------------------

    optimizer.zero_grad()

    # --------------------------------------------------------
    # 前向传播
    # --------------------------------------------------------

    out = model(data)

    # --------------------------------------------------------
    # 只使用训练节点计算 Loss
    # --------------------------------------------------------

    loss = F.nll_loss(
        out[data.train_mask],
        data.y[data.train_mask]
    )

    # --------------------------------------------------------
    # 反向传播
    # --------------------------------------------------------

    loss.backward()

    # --------------------------------------------------------
    # 更新模型参数
    # --------------------------------------------------------

    optimizer.step()

    # --------------------------------------------------------
    # 每 50 个 epoch 输出一次
    # --------------------------------------------------------

    if epoch % 50 == 0:

        print(
            f"Epoch [{epoch}/{num_epochs}] "
            f"Loss: {loss.item():.4f}"
        )


# ============================================================
# 9. 模型评估
# ============================================================

model.eval()

with torch.no_grad():

    out = model(data)

    # 得到预测类别
    pred = out.argmax(dim=1)

    # 测试集准确率
    correct = (
        pred[data.test_mask]
        == data.y[data.test_mask]
    ).sum()

    accuracy = (
        correct.float()
        / data.test_mask.sum()
    )

print("--------------------------------")
print("训练完成！")
print("最终测试集准确率：", accuracy.item())
print("--------------------------------")