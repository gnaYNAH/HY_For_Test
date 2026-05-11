# ===================== 导入所需库 =====================
import torch
import torch.nn as nn
import torch.optim as optim
# 学习率余弦退火调度器
from torch.optim.lr_scheduler import CosineAnnealingLR
# 加载MNIST数据集与数据预处理
from torchvision import datasets, transforms
# 数据加载器
from torch.utils.data import DataLoader
# 进度条美化训练过程
from tqdm import tqdm
# 绘图可选，如需看损失曲线可开启
import matplotlib.pyplot as plt

# ===================== 适配Mac设备（Apple Silicon MPS优先） =====================
# Mac M1/M2/M3 自动用MPS加速，没有则用CPU
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("✅ 使用 Mac MPS GPU 加速")
else:
    DEVICE = torch.device("cpu")
    print("⚠️ MPS不可用，使用 CPU 运行")

# ===================== 超参数全局配置 =====================
EPOCHS = 100           # 训练总轮数
BATCH_SIZE = 32        # 每批次样本数
LR = 1e-3               # 初始学习率

# ===================== 1. 补全：定义 SimpleNet 简单卷积网络 =====================
class SimpleNet(nn.Module):
    """简易CNN卷积网络，用于MNIST手写数字分类"""
    def __init__(self):
        super(SimpleNet, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        # 全连接层
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # 前向传播
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        # 展平
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ===================== 2. 数据预处理 & 加载MNIST数据集 =====================
# 数据归一化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载训练集、测试集
train_dataset = datasets.MNIST(
    root="./data", train=True, download=False, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=False, transform=transform
)

# 构造数据加载器
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False
)

# ===================== 3. 模型、损失函数、优化器初始化 =====================
# 创建模型并移到Mac可用设备上
model = SimpleNet().to(DEVICE)

# 多分类任务交叉熵损失
criterion = nn.CrossEntropyLoss()

# AdamW优化器：带权重衰减，防止过拟合
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

# 余弦退火学习率：训练过程自动降低学习率
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

# ===================== 4. 训练主循环 =====================
best_acc = 0.0  # 记录最佳准确率
# 保存每轮损失，后续可绘图
train_loss_list = []
test_acc_list = []

print("=" * 60)
print("开始训练 MNIST 分类模型")
print("=" * 60)

for epoch in range(EPOCHS):
    # ========== 训练阶段 ==========
    model.train()  # 切换到训练模式
    total_loss = 0.0
    train_correct = 0

    # 加入进度条，美化命令行输出
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", ncols=100)
    for inputs, labels in pbar:
        # 数据移到设备
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        # 清空梯度：set_to_none=True 更省内存，Mac友好
        optimizer.zero_grad(set_to_none=True)

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播求梯度
        loss.backward()

        # 梯度裁剪：防止深层网络梯度爆炸
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 更新模型参数
        optimizer.step()

        # 累计损失、正确样本数
        total_loss += loss.item()
        # argmax(1) 取出预测类别
        train_correct += (outputs.argmax(1) == labels).sum().item()

    # 每轮结束更新学习率
    scheduler.step()

    # 计算训练集平均损失、准确率
    avg_loss = total_loss / len(train_loader)
    train_acc = train_correct / len(train_loader.dataset)
    current_lr = scheduler.get_last_lr()[0]

    train_loss_list.append(avg_loss)

    # ========== 测试集评估（不参与训练，看真实泛化能力） ==========
    model.eval()  # 切换评估模式
    test_correct = 0
    with torch.no_grad():  # 关闭梯度，节省Mac内存
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs)
            test_correct += (outputs.argmax(1) == labels).sum().item()

    test_acc = test_correct / len(test_loader.dataset)
    test_acc_list.append(test_acc)

    # 打印本轮日志
    print(f"Loss: {avg_loss:.4f} | 训练准确率: {train_acc:.4f} | "
          f"测试准确率: {test_acc:.4f} | 学习率: {current_lr:.6f}")

    # 保存测试集最佳模型
    if test_acc > best_acc:
        best_acc = test_acc
        # 保存完整断点：模型权重、优化器、学习率调度器、最佳精度
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc,
        }, 'best_model.pth')
        print(f"📌 已保存当前最佳模型，最佳测试准确率: {best_acc:.4f}")

print("=" * 60)
print(f"训练全部完成！全局最佳测试准确率: {best_acc:.4f}")
print("=" * 60)

# ===================== 可选：绘制训练损失 & 测试准确率曲线 =====================
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

plt.figure(figsize=(12, 5))
plt.subplot(1,2,1)
plt.plot(train_loss_list, label="训练损失", color="#e74c3c")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("训练损失变化")
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1,2,2)
plt.plot(test_acc_list, label="测试准确率", color="#27ae60")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("测试准确率变化")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()