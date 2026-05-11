# 导入PyTorch核心库
import torch
# 导入神经网络模块
import torch.nn as nn
# 导入优化器
import torch.optim as optim
# 导入绘图库，用于训练损失可视化
import matplotlib.pyplot as plt
# 导入进度条库，命令行展示训练进度
from tqdm import tqdm

# 关闭多余警告
import warnings
warnings.filterwarnings("ignore")

# ===================== 定义标准Transformer模型类（彻底修复维度） =====================
class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, max_seq_len=1000):
        super(TransformerModel, self).__init__()

        # 词嵌入层：单词索引 -> 稠密向量
        self.embedding = nn.Embedding(input_dim, model_dim)

        # 可学习位置编码 [max_seq_len, model_dim] 改成这种写法最稳
        self.positional_encoding = nn.Parameter(torch.randn(max_seq_len, model_dim))

        # Transformer 开启 batch_first=True 统一维度 [batch, seq_len, dim]
        self.transformer = nn.Transformer(
            d_model=model_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            batch_first=True,
            dropout=0.1
        )

        # 输出全连接层
        self.fc = nn.Linear(model_dim, output_dim)

    def forward(self, src, tgt):
        # src/tgt : [batch, seq_len]
        batch_size = src.size(0)
        src_seq_len = src.size(1)
        tgt_seq_len = tgt.size(1)

        # 词嵌入
        src_emb = self.embedding(src)  # [batch, src_seq_len, model_dim]
        tgt_emb = self.embedding(tgt)  # [batch, tgt_seq_len, model_dim]

        # 位置编码截取 + 广播适配batch维度，彻底解决维度不匹配
        src_pe = self.positional_encoding[:src_seq_len, :].unsqueeze(0)  # [1, seq_len, dim]
        tgt_pe = self.positional_encoding[:tgt_seq_len, :].unsqueeze(0)

        src_emb = src_emb + src_pe
        tgt_emb = tgt_emb + tgt_pe

        # Transformer 前向
        trans_out = self.transformer(src_emb, tgt_emb)  # [batch, tgt_seq_len, model_dim]

        # 映射到词汇表
        output = self.fc(trans_out)
        return output

# ===================== 超参数配置 =====================
input_dim = 10000     # 词汇表大小
model_dim = 512       # 模型维度
num_heads = 8         # 多头注意力头数
num_layers = 2        # 层数减小，更快收敛
output_dim = 10000    # 输出词汇表
num_epochs = 80       # 训练轮数
lr = 0.001            # 学习率
batch_size = 32       # 批次大小
src_len = 10          # 源序列长度
tgt_len = 15          # 目标序列长度

# ===================== 初始化模型、损失函数、优化器 =====================
model = TransformerModel(input_dim, model_dim, num_heads, num_layers, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# ===================== 构造模拟数据 [batch, seq_len] =====================
src = torch.randint(0, input_dim, (batch_size, src_len))
tgt = torch.randint(0, input_dim, (batch_size, tgt_len))

# 保存损失
loss_list = []

# ===================== 循环训练 + 进度条 =====================
print("========== Transformer 模型开始训练 ==========")
for epoch in tqdm(range(num_epochs), desc="训练进度", ncols=80):
    optimizer.zero_grad()

    output = model(src, tgt)

    # 展平维度计算损失
    loss = criterion(output.reshape(-1, output_dim), tgt.reshape(-1))

    loss.backward()
    optimizer.step()

    loss_list.append(loss.item())

print("========== 训练完成 ==========")

# ===================== 绘制损失曲线 =====================
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

plt.figure(figsize=(10, 5))
plt.plot(loss_list, color='#2E86AB', linewidth=2, label='训练损失 Loss')
plt.xlabel('训练轮数 Epoch')
plt.ylabel('损失值 Loss')
plt.title('Transformer 训练损失变化曲线')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()