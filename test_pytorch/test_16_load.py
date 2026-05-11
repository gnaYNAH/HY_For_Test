# 导入PyTorch基础库
import torch
# 导入神经网络模块
import torch.nn as nn
# 导入优化器模块
import torch.optim as optim

# ===================== 适配Mac设备（Apple Silicon 专属优化） =====================
# 优先使用MPS GPU加速，无MPS则自动切CPU，内存占用更低
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("✅ Mac 检测到MPS加速，使用GPU训练")
else:
    DEVICE = torch.device("cpu")
    print("⚠️ 未检测到MPS，使用CPU训练")

# ===================== 定义极简神经网络模型 =====================
class SimpleModel(nn.Module):
    """
    极简单层全连接网络
    输入维度：10
    输出分类：2分类
    """
    def __init__(self):
        # 继承父类初始化
        super(SimpleModel, self).__init__()
        # 全连接层：10个输入特征 → 2个输出类别
        self.fc = nn.Linear(in_features=10, out_features=2)

    def forward(self, x):
        """前向传播逻辑"""
        # 数据过全连接层，返回预测logits
        return self.fc(x)

# ===================== 模型、优化器、损失函数初始化 =====================
# 创建模型并移到Mac最优设备上
model = SimpleModel().to(DEVICE)

# 随机梯度下降优化器
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 交叉熵损失：用于2分类任务
criterion = nn.CrossEntropyLoss()

# ===================== 模拟训练流程 =====================
# 总共训练5个轮次
for epoch in range(5):
    # 切换为训练模式（启用Dropout/BatchNorm等训练专属逻辑）
    model.train()

    # 模拟构造一批假训练数据
    # inputs: [批次32, 特征维度10] 随机正态分布
    inputs = torch.randn(32, 10).to(DEVICE)
    # labels: 32个样本，标签只能是0或1
    labels = torch.randint(0, 2, (32,)).to(DEVICE)

    # 清空历史梯度，节省Mac内存
    optimizer.zero_grad(set_to_none=True)

    # 前向传播：模型预测
    outputs = model(inputs)

    # 计算预测与真实标签的损失
    loss = criterion(outputs, labels)

    # 反向传播：计算梯度
    loss.backward()

    # 根据梯度更新模型参数
    optimizer.step()

    # ===================== 每2个Epoch保存一次训练断点 =====================
    if epoch % 2 == 0:
        # 打包断点：轮数、模型权重、优化器状态、当前损失
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }
        # 按轮数命名保存，不覆盖历史断点
        torch.save(checkpoint, f'checkpoint_epoch{epoch}.pth')
        print(f'📌 已保存第 {epoch} 轮训练断点')

# ===================== 训练结束：保存最后一轮完整模型权重 =====================
torch.save(model.state_dict(), 'final_model.pth')
print("\n✅ 训练结束，已保存最后一轮模型：final_model.pth")

# ===================== 演示：加载模型 + 推理预测 =====================
# 1. 新建空模型结构
loaded_model = SimpleModel().to(DEVICE)
# 2. 加载保存好的权重参数
loaded_model.load_state_dict(torch.load('final_model.pth', map_location=DEVICE))
# 3. 切换为评估模式（关闭训练专属层，推理更省内存）
loaded_model.eval()

# 构造单条测试输入
test_input = torch.randn(1, 10).to(DEVICE)

<<<<<<< HEAD
# 推理时关闭梯度计算，极致节省Mac内存
=======
# 推理时关闭梯度计算，极致节省Mac内存，，，，，，
>>>>>>> f5d10e9 (init: 初始化项目，提交所有测试代码1)
with torch.no_grad():
    test_output = loaded_model(test_input)

print(f'\n🧪 模型推理输出：{test_output}')