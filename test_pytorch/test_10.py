import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 1. 准备数据
torch.manual_seed(42)
X = torch.randn(100, 2)  # 100 个样本，2 个特征
true_w = torch.tensor([2.0, 3.0])
true_b = 4.0
Y = X @ true_w + true_b + torch.randn(100) * 0.1

# 2. 定义模型
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel()

# 3. 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 4. 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()

    predictions = model(X)
    loss = criterion(predictions.squeeze(), Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. 评估模型
print(f'\n训练后的权重: {model.linear.weight.data.numpy()}')
print(f'训练后的偏置: {model.linear.bias.data.numpy()}')
print(f'真实权重: {true_w.numpy()}')
print(f'真实偏置: {true_b}')