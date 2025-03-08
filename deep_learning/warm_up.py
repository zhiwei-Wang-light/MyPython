# -*- coding: utf-8 -*- #
# ------------------------------------------------------------------
# File Name:        warm_up.py
# Author:           wzw
# Version:          0.1
# Created:          2024/11/7
# Description:      预热+余弦退火
# ------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import math

# 1. 加载鸢尾花数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 2. 数据预处理：标准化特征
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 3. 数据分割：80% 训练集，20% 测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 5. 创建数据集和数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


# 6. 定义神经网络模型
class IrisModel(nn.Module):
    def __init__(self):
        super(IrisModel, self).__init__()
        self.fc1 = nn.Linear(4, 32)  # 输入层 (4个特征)
        self.fc2 = nn.Linear(32, 3)  # 输出层 (3个类别)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # ReLU 激活函数
        x = self.fc2(x)  # 输出层
        return x


# 7. 初始化模型，损失函数和优化器
model = IrisModel()
criterion = nn.CrossEntropyLoss()  # 多分类交叉熵损失
warm_up_iter = 10
lr_max = 0.01
lr_min = 0.0001
t_max = 100
lambda0 = lambda cur_iter: cur_iter / warm_up_iter if cur_iter < warm_up_iter else \
    (lr_min + 0.5 * (lr_max - lr_min) * (
                1.0 + math.cos((cur_iter - warm_up_iter) / (t_max - warm_up_iter) * math.pi))) / lr_max
optimizer = optim.Adam(model.parameters(), lr=lr_max)
# LambdaLR
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda0])
# 8. 训练模型
epochs = 100
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    print(optimizer.param_groups[0]['lr'])
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    scheduler.step()
    # 每10个epoch输出一次损失
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}')

# 9. 测试模型
model.eval()  # 设置模型为评估模式
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy on test data: {accuracy:.2f}%')
