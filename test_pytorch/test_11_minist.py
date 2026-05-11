# 导入PyTorch核心库
import torch
# 导入神经网络模块
import torch.nn as nn
# 导入神经网络函数接口
import torch.nn.functional as F
# 导入优化器模块
import torch.optim as optim
# 导入torchvision用于数据集和数据预处理
from torchvision import datasets, transforms
# 导入绘图库用于结果可视化
import matplotlib.pyplot as plt

# -------------------------- 1. 数据加载与预处理 --------------------------
# 定义数据预处理流程：组合多个预处理操作
transform = transforms.Compose([
    transforms.ToTensor(),  # 将PIL图像或numpy数组转换为张量，并归一化到[0,1]
    transforms.Normalize((0.5,), (0.5,))  # 标准化：将数据从[0,1]归一化到[-1,1]
])

# 加载MNIST训练数据集
# root：数据集存放路径；train=True：加载训练集；transform：应用预处理；download=True：自动下载
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

# 加载MNIST测试数据集
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# 训练集数据加载器：批量加载、打乱数据
# batch_size=64：每批次64张图片；shuffle=True：打乱数据顺序，提升训练效果
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# 测试集数据加载器：批量加载、不打乱
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# -------------------------- 2. 定义CNN卷积神经网络模型 --------------------------
# 继承nn.Module，构建自定义卷积网络
class SimpleCNN(nn.Module):
    # 初始化网络层
    def __init__(self):
        # 调用父类初始化方法
        super(SimpleCNN, self).__init__()
        
        # 第一层卷积层
        # in_channels=1：输入是灰度图，单通道；out_channels=32：输出32个特征图
        # kernel_size=3：卷积核大小3×3；stride=1：步长1；padding=1：填充1圈，保持尺寸不变
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        
        # 第二层卷积层
        # 输入32通道（上一层输出），输出64通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # 第一层全连接层
        # 输入维度：64个特征图 × 7×7（经过两次池化后尺寸）；输出128维
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        
        # 第二层全连接层（输出层）
        # 输入128维，输出10维（对应0-9共10个数字类别）
        self.fc2 = nn.Linear(128, 10)

    # 前向传播（数据流向）
    def forward(self, x):
        # 卷积1 → ReLU激活 → 最大池化（2×2）
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # 池化后尺寸减半：28→14
        
        # 卷积2 → ReLU激活 → 最大池化
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # 池化后尺寸再减半：14→7
        
        # 展平：将特征图拉成一维向量，-1表示自动计算batch维度
        x = x.view(-1, 64 * 7 * 7)
        
        # 全连接层1 + ReLU激活
        x = F.relu(self.fc1(x))
        
        # 全连接层2（输出层，无激活，CrossEntropyLoss自带softmax）
        x = self.fc2(x)
        
        return x

# 创建模型实例
model = SimpleCNN()

# -------------------------- 3. 定义损失函数与优化器 --------------------------
# 多分类任务使用交叉熵损失（自带LogSoftmax+NLLLoss）
criterion = nn.CrossEntropyLoss()

# 随机梯度下降优化器
# lr=0.01：学习率；momentum=0.9：动量，加速收敛
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# -------------------------- 4. 模型训练 --------------------------
# 训练轮次：整个数据集训练5遍
num_epochs = 5

# 设置模型为训练模式（启用Dropout、BatchNorm等训练特有层）
model.train()

# 循环训练每一轮
for epoch in range(num_epochs):
    total_loss = 0  # 累计损失
    
    # 遍历训练集的每一个批次
    for images, labels in train_loader:
        outputs = model(images)  # 前向传播：输入图片，得到预测输出
        loss = criterion(outputs, labels)  # 计算预测值与真实标签的损失
        
        optimizer.zero_grad()  # 清空上一步的梯度，避免累积
        loss.backward()        # 反向传播：计算梯度
        optimizer.step()       # 更新网络参数
        
        total_loss += loss.item()  # 累加当前批次损失
    
    # 打印本轮平均损失
    print(f"Epoch [{epoch+1}/{num_epochs}], 平均损失: {total_loss / len(train_loader):.4f}")

# -------------------------- 5. 模型测试 --------------------------
# 设置模型为评估模式（关闭Dropout、BatchNorm等）
model.eval()

correct = 0  # 预测正确的样本数
total = 0    # 总样本数

# 测试时关闭梯度计算，节省内存、加速计算
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)  # 前向传播
        # 取输出最大值的索引作为预测类别
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)  # 累加批次样本数
        # 统计预测正确的数量
        correct += (predicted == labels).sum().item()

# 计算测试集准确率
accuracy = 100 * correct / total
print(f"测试集准确率: {accuracy:.2f}%")

# -------------------------- 6. 测试结果可视化 --------------------------
# 从测试集取一个批次数据
dataiter = iter(test_loader)
images, labels = next(dataiter)

# 模型预测
outputs = model(images)
_, predictions = torch.max(outputs, 1)

# 创建1行6列的子图，显示6张图片
fig, axes = plt.subplots(1, 6, figsize=(12, 4))

# 循环绘制图片
for i in range(6):
    axes[i].imshow(images[i][0], cmap='gray')  # 显示灰度图
    axes[i].set_title(f"真实标签: {labels[i]}\n预测: {predictions[i]}")
    axes[i].axis('off')  # 关闭坐标轴

plt.show()