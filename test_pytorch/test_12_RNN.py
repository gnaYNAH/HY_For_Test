# ===================== 导入所需库 =====================
import torch                # PyTorch 核心库，张量与模型计算
import torch.nn as nn       # 神经网络层
import torch.optim as optim # 优化器
import matplotlib.pyplot as plt  # 绘图
import numpy as np          # 数值计算、独热编码
from tqdm import tqdm       # 命令行动态进度条

# ===================== 适配中文绘图（自动兼容 Win/Mac） =====================
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# ===================== 1. 构建字符词典 & 序列任务 =====================
# 字符池
char_set = list("hello")
# 字符转索引、索引转字符
char_to_idx = {c: i for i, c in enumerate(char_set)}
idx_to_char = {i: c for i, c in enumerate(char_set)}

# 任务：输入 hello  ->  输出 elloh
input_str = "hello"
target_str = "elloh"

# 字符序列转为数字索引
input_data = [char_to_idx[c] for c in input_str]
target_data = [char_to_idx[c] for c in target_str]

# ===================== 2. 数据预处理：独热编码 =====================
# 生成独热编码
input_one_hot = np.eye(len(char_set))[input_data]

# 转为 PyTorch 张量
inputs = torch.tensor(input_one_hot, dtype=torch.float32)
targets = torch.tensor(target_data, dtype=torch.long)

# ===================== 3. 超参数设置 =====================
input_size = len(char_set)    # 输入特征维度
hidden_size = 16              # 加大隐藏层，拟合更好
output_size = len(char_set)   # 输出类别数
num_epochs = 300              # 训练轮数适当增加
learning_rate = 0.05          # 学习率微调

# ===================== 4. 定义 RNN 模型 =====================
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        # RNN层：batch_first=True → [batch, seq_len, feature]
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        # 全连接层：隐藏层 → 输出类别
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden

# 初始化模型
model = RNNModel(input_size, hidden_size, output_size)

# ===================== 5. 损失函数 & 优化器 =====================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ===================== 6. 训练过程（带进度条+实时Loss） =====================
losses = []
hidden = None  # 初始隐藏状态

print("=" * 50)
print(f"输入序列：{input_str}")
print(f"目标序列：{target_str}")
print("=" * 50)

# tqdm 进度条，实时展示 loss
pbar = tqdm(range(num_epochs), desc="训练中", ncols=90)
for epoch in pbar:
    optimizer.zero_grad()

    # 增加batch维度送入模型
    outputs, hidden = model(inputs.unsqueeze(0), hidden)
    # 截断梯度，防止RNN梯度累积爆炸
    hidden = hidden.detach()

    # 计算损失
    loss = criterion(outputs.view(-1, output_size), targets)

    # 反向传播 + 参数更新
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    # 进度条右侧实时刷新当前损失
    pbar.set_postfix(loss=f"{loss.item():.6f}")

print("=" * 50)
print("训练全部完成！")

# ===================== 7. 模型测试 & 结果精细化输出 =====================
with torch.no_grad():
    test_hidden = None
    test_output, _ = model(inputs.unsqueeze(0), test_hidden)
    predicted = torch.argmax(test_output, dim=2).squeeze().numpy()

    # 数字序列转回字符
    input_seq = ''.join([idx_to_char[i] for i in input_data])
    target_seq = ''.join([idx_to_char[i] for i in target_data])
    pred_seq = ''.join([idx_to_char[i] for i in predicted])

print("\n【预测结果对比】")
print(f"输入序列：{input_seq}")
print(f"目标序列：{target_seq}")
print(f"预测序列：{pred_seq}")

# 判断是否完全预测正确
if pred_seq == target_seq:
    print("✅ 模型完全学会序列转换！")
else:
    print("❌ 未完全拟合，可增大 hidden_size 或增加训练轮数")

# ===================== 8. 美化损失曲线 =====================
plt.figure(figsize=(10, 5))
plt.plot(losses, color='#2E86AB', linewidth=2, label='训练损失')
plt.xlabel('训练轮数 Epoch')
plt.ylabel('损失值 Loss')
plt.title('RNN 序列预测 训练损失变化曲线')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()