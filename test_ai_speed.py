import torch
import torch.nn as nn
import time

# ======================
# 真实一点的小模型：分类网络
# ======================
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)  # 模拟10分类
        )

    def forward(self, x):
        return self.layers(x)


# ======================
# 设备：自动用 Mac GPU (MPS)
# ======================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"使用设备: {device}")

model = SimpleNet().to(device)

# 优化器、损失函数（真实训练标配）
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# ======================
# 模拟一批训练数据（不用真实数据集）
# ======================
batch_size = 512
epochs = 150

# 数据搬到 GPU
x = torch.randn(batch_size, 128).to(device)
y = torch.randint(0, 10, (batch_size,)).to(device)

# ======================
# 开始训练
# ======================
print("\n开始真实 AI 训练...\n")
start_time = time.time()

for epoch in range(epochs):
    optimizer.zero_grad()
    pred = model(x)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 30 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

end_time = time.time()

print("\n================================")
print(f"训练总耗时：{end_time - start_time:.2f} 秒")
print(f"最终损失值: {loss.item():.4f}")
print("✅ 训练完成！你的 Mac 很强")
print("================================")