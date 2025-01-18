import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

# 假设P1是你提到的时序序列，这里我们使用随机数据来模拟
# 在实际应用中，你应该替换为你的时序序列数据
print(torch.cuda.is_available())
fr_amazon = open('Data/Cell_Phones_and_Accessories/2010\\Struct_KG.txt', 'r')
lines = fr_amazon.readlines()[1:]
for line in lines:
    P_dict = eval(line)
    for review in P_dict.keys():
        p = []
        for i in range(0, len(P_dict[review])):
            p.append(P_dict[review][i][6])
        print(p)
# P1 = torch.tensor([x1, x2, ..., xn])
#
# # 定义高斯分布模型
# class GaussianModel(nn.Module):
#     def __init__(self):
#         super(GaussianModel, self).__init__()
#         self.loc = nn.Parameter(torch.tensor(0.0))  # 均值初始化为0
#         self.scale = nn.Parameter(torch.tensor(1.0))  # 方差初始化为1
#
#     def forward(self):
#         return Normal(self.loc, self.scale)
#
# # 初始化模型
# model = GaussianModel()
#
# # 定义损失函数（负对数似然）
# def negative_log_likelihood(model, x):
#     return -model().log_prob(x).mean()
#
# # 选择优化器
# optimizer = optim.Adam(model.parameters(), lr=0.01)
#
# # 训练模型
# num_epochs = 100
# for epoch in range(num_epochs):
#     # 前向传播
#     loss = negative_log_likelihood(model, P1)
#     # 反向传播和优化
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
#     if (epoch + 1) % 10 == 0:
#         print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')
#
# # 检测异常
# with torch.no_grad():
#     anomalies_scores = -model().log_prob(P1)
#     threshold = torch.percentile(anomalies_scores, 95)
#     anomalies = P1[anomalies_scores > threshold]
#
# # 打印异常值
# print('Anomalies:', anomalies)
