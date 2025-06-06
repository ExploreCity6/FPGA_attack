# import pandas as pd
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.preprocessing import StandardScaler

# # CSV文件路径
# csv_file = 'D:/Users/空城丶/Desktop/fpga-code/data-analysis/CHUNK_SIZE/data 4-10-200-1000/train1.csv'

# # 读取CSV文件
# data = pd.read_csv(csv_file)

# # 分离特征和标签
# X = data.iloc[:, 1:]  # 特征列，从第二列开始
# y = data.iloc[:, 0]   # 第一列是标签列

# # 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 特征缩放
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # 创建带有参数网格的网格搜索对象
# param_grid = {
#     'C': [1, 10, 100, 1000],  # 尝试更大的C值
#     'gamma': [0.001, 0.01, 0.1, 1],  # 尝试不同的gamma值
# }
# grid_search = GridSearchCV(SVC(kernel='rbf', probability=True), param_grid, cv=5)

# # 执行网格搜索
# grid_search.fit(X_train_scaled, y_train)

# # 输出最佳参数
# print("Best parameters:", grid_search.best_params_)

# # 使用最佳参数的模型进行训练
# best_model = grid_search.best_estimator_
# best_model.fit(X_train_scaled, y_train)

# # 预测测试集
# y_pred = best_model.predict(X_test_scaled)

# # 计算准确率
# accuracy = accuracy_score(y_test, y_pred)
# print(classification_report(y_test, y_pred, zero_division=1))  # 设置zero_division=1
# print(f'模型的准确率为: {accuracy:.4f}')

# from sklearn.metrics import precision_score, recall_score, f1_score

# recall = recall_score(y_test, y_pred, average='macro')
# f1 = f1_score(y_test, y_pred, average='macro')

# print(f'召回率: {recall:.4f}')
# print(f'F1分数: {f1:.4f}')


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# 请确保CSV文件的路径正确
csv_file = 'D:/Users/空城丶/Desktop/fpga-code/data-analysis/CHUNK_SIZE/data 4-10-200-1000/train.csv'  # 更改为您的CSV文件路径

# 读取CSV文件
data = pd.read_csv(csv_file)

# 分离特征和标签
X = data.iloc[:, 1:]  # 特征列，从第二列开始
y = data.iloc[:, 0]   # 第一列是标签列

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征缩放
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 创建带有参数网格的网格搜索对象
param_grid = {
    'C': [10],  # 尝试不同的C值
    'gamma': [0.001],  # 尝试不同的gamma值
}
grid_search = GridSearchCV(SVC(kernel='rbf', probability=True), param_grid, cv=5)

# 执行网格搜索
grid_search.fit(X_train_scaled, y_train)

# 输出最佳参数
# print("Best parameters:", grid_search.best_params_)

# 使用最佳参数的模型进行训练
best_model = grid_search.best_estimator_
best_model.fit(X_train_scaled, y_train)

# 预测测试集
y_pred = best_model.predict(X_test_scaled)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred, zero_division=1))  # 设置zero_division=1
print(f'模型的准确率为: {accuracy:.4f}')

# 计算召回率和F1分数
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print(f'召回率: {recall:.4f}')
print(f'F1分数: {f1:.4f}')

# # 打印真实标签和预测标签的对比，按类别分组
# print("Predicted and True labels by class:")
# predicted_true_labels = pd.DataFrame({'True': y_test, 'Predicted': y_pred})
# print(predicted_true_labels.groupby(['True', 'Predicted']).size().unstack().fillna(0))

# # 打印出所有的测试标签和预测结果的标签
# for true, pred in zip(y_test, y_pred):
#     print(f'True: {true}, Predicted: {pred}')


# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
# from sklearn.preprocessing import MinMaxScaler

# # CSV文件路径
# csv_file = 'D:/Users/空城丶/Desktop/fpga-code/data-analysis/CHUNK_SIZE/data 4-10-200-1000/train.csv'


# # 读取CSV文件
# data = pd.read_csv(csv_file)

# # 分离特征和标签
# X = data.iloc[:, 1:]  # 特征列，从第二列开始
# y = data.iloc[:, 0]   # 第一列是标签列

# # 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 创建MinMaxScaler实例
# scaler = MinMaxScaler()

# # 训练数据集归一化
# X_train_scaled = scaler.fit_transform(X_train)

# # 测试数据集归一化
# X_test_scaled = scaler.transform(X_test)

# # 创建随机森林模型
# rf = RandomForestClassifier(n_estimators=100, random_state=42)

# # 训练模型
# rf.fit(X_train_scaled, y_train)

# # 预测测试集
# y_pred = rf.predict(X_test_scaled)

# # 计算准确率
# accuracy = accuracy_score(y_test, y_pred)
# # print(f'模型的准确率为: {accuracy:.4f}')

# # 输出精确率、召回率、F1分数等分类报告
# print("\n分类报告:")
# print(classification_report(y_test, y_pred))

# # # 如果需要单独输出 Precision, Recall, F1 Score
# # precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
# # print(f'精确率 (Precision): {precision:.4f}')
# # print(f'召回率 (Recall): {recall:.4f}')
# # print(f'F1 分数: {f1_score:.4f}')

# # 打印真实标签和预测标签的对比，按类别分组
# print("Predicted and True labels by class:")
# predicted_true_labels = pd.DataFrame({'True': y_test, 'Predicted': y_pred})
# print(predicted_true_labels.groupby(['True', 'Predicted']).size().unstack().fillna(0))




# import torch
# import torch.nn as nn
# import torch.optim as optim
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

# # 加载数据集
# data = pd.read_csv('train.csv')

# # 准备数据：第一列是标签，剩余列是特征
# X = data.iloc[:, 1:].values
# y = data.iloc[:, 0].values

# # 标准化特征
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # 将数据拆分为训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# # 将数据转换为PyTorch张量
# X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
# X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
# y_train_tensor = torch.tensor(y_train, dtype=torch.long)
# y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# # 定义MLP模型
# class MLP(nn.Module):
#     def __init__(self):
#         super(MLP, self).__init__()
#         self.layer1 = nn.Linear(X_train.shape[1], 128)
#         self.layer2 = nn.Linear(128, 64)
#         self.layer3 = nn.Linear(64, 32)
#         self.output = nn.Linear(32, 6)  # 输出层改为6个类别

#     def forward(self, x):
#         x = torch.relu(self.layer1(x))
#         x = torch.relu(self.layer2(x))
#         x = torch.relu(self.layer3(x))
#         x = self.output(x)
#         return x

# # 实例化模型
# model = MLP()

# # 定义损失函数和优化器
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001)

# # 训练模型
# def train_model(model, criterion, optimizer, X_train, y_train, epochs=1500):
#     model.train()
#     for epoch in range(epochs):
#         # 前向传播
#         outputs = model(X_train)
#         loss = criterion(outputs, y_train)

#         # 反向传播和优化
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
 
#         # # 每100轮打印一次损失
#         # if (epoch+1) % 100 == 0:
#         #     print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# # 训练模型1500次
# train_model(model, criterion, optimizer, X_train_tensor, y_train_tensor, epochs=1500)

# # 在测试集上评估模型
# model.eval()
# with torch.no_grad():
#     # 获取预测
#     test_outputs = model(X_test_tensor)
#     _, predicted = torch.max(test_outputs, 1)

# # 计算准确率
# accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
# print(f'模型在测试集上的准确率: {accuracy:.4f}')


# import torch
# import torch.nn as nn
# import torch.optim as optim
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

# # 加载数据集
# data = pd.read_csv('D:/Users/空城丶/Desktop/fpga-code/data-analysis/REPEATS/data 4-20-200-1000/train.csv')

# # 准备数据：第一列是标签，剩余列是特征
# X = data.iloc[:, 1:].values
# y = data.iloc[:, 0].values

# # 标准化特征
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # 将数据拆分为训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# # 将数据转换为PyTorch张量，并调整为适合1D卷积的形状
# X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # 添加一个通道维度
# X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
# y_train_tensor = torch.tensor(y_train, dtype=torch.long)
# y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# # 定义1D卷积神经网络模型
# class ConvNet1D(nn.Module):
#     def __init__(self):
#         super(ConvNet1D, self).__init__()
#         # 1D卷积层：输入通道为1，输出通道为16，卷积核大小为3
#         self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
#         self.batch_norm = nn.BatchNorm1d(16)  # 批归一化层
#         self.fc1 = nn.Linear(16 * X_train.shape[1], 128)  # 全连接层1
#         self.fc2 = nn.Linear(128, 64)  # 全连接层2
#         self.fc3 = nn.Linear(64, 32)  # 全连接层3
#         self.output = nn.Linear(32, 6)  # 输出6个类别
        
#         # 使用Softmax激活函数
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, x):
#         # 卷积层 + 批归一化 + ReLU
#         x = torch.relu(self.batch_norm(self.conv1(x)))
        
#         # 展平
#         x = x.view(x.size(0), -1)  # 扁平化
        
#         # 3层全连接网络
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = torch.relu(self.fc3(x))
        
#         # 输出层，使用Softmax激活函数
#         x = self.softmax(self.output(x))
#         return x

# # 实例化模型
# model = ConvNet1D()

# # 定义损失函数和优化器（Adam优化器）
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # 训练模型
# def train_model(model, criterion, optimizer, X_train, y_train, epochs=1500):
#     model.train()
#     for epoch in range(epochs):
#         # 前向传播
#         outputs = model(X_train)
#         loss = criterion(outputs, y_train)

#         # 反向传播和优化
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         # # 每100轮打印一次损失
#         # if (epoch+1) % 100 == 0:
#         #     print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# # 训练模型1500次
# train_model(model, criterion, optimizer, X_train_tensor, y_train_tensor, epochs=1500)

# # 在测试集上评估模型
# model.eval()
# with torch.no_grad():
#     # 获取预测
#     test_outputs = model(X_test_tensor)
#     _, predicted = torch.max(test_outputs, 1)

# # 计算准确率
# accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
# print(f'模型在测试集上的准确率: {accuracy:.4f}')
