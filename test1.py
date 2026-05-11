# 测试3：最简单的人工智能 - 线性回归预测
from sklearn.linear_model import LinearRegression
import numpy as np

# 准备数据
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# 训练模型
model = LinearRegression()
model.fit(X, y)

# 预测
predict = model.predict([[6]])
print("预测结果：6 对应输出 =", predict[0])
print("🎉 AI 模型运行成功！环境完全正常")