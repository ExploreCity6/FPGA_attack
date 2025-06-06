# import pandas as pd
# from sklearn.model_selection import GridSearchCV
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.preprocessing import StandardScaler

# # 请确保CSV文件的路径正确
# train_csv_file = 'D:/Users/空城丶/Desktop/fpga-code/openwolrd/del four/train no acmv.csv'  # 训练集CSV文件路径
# test_csv_file = 'D:/Users/空城丶/Desktop/fpga-code/openwolrd/del four/test no acmv.csv'    # 测试集CSV文件路径

# # 读取训练集CSV文件
# X_train = pd.read_csv(train_csv_file, header=None).iloc[:, 1:]  # 特征列，从第二列开始
# y_train = pd.read_csv(train_csv_file, header=None).iloc[:, 0]   # 第一列是标签列

# # 读取测试集CSV文件
# X_test = pd.read_csv(test_csv_file, header=None).iloc[:, 1:]  # 特征列，从第二列开始
# y_test = pd.read_csv(test_csv_file, header=None).iloc[:, 0]   # 第一列是标签列

# # 特征缩放
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # 创建带有参数网格的网格搜索对象
# param_grid = {
#     'C': [10],  # 尝试不同的C值
#     'gamma': [0.01],  # 尝试不同的gamma值
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

# #打印分类报告，包括每个类别的准确率
# print(classification_report(y_test, y_pred, zero_division=1))

# # 打印真实标签和预测标签的对比，按类别分组
# print("Predicted and True labels by class:")
# predicted_true_labels = pd.DataFrame({'True': y_test, 'Predicted': y_pred})
# print(predicted_true_labels.groupby(['True', 'Predicted']).size().unstack().fillna(0))


# import pandas as pd
# from sklearn.model_selection import GridSearchCV
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.preprocessing import StandardScaler
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
# from sklearn.preprocessing import MinMaxScaler

# # 请确保CSV文件的路径正确
# train_csv_file = 'D:/Users/空城丶/Desktop/fpga-code/openwolrd/del one/train1.csv'  # 训练集CSV文件路径
# test_csv_file = 'D:/Users/空城丶/Desktop/fpga-code/openwolrd/del one/test1.csv'    # 测试集CSV文件路径

# # 读取训练集CSV文件
# X_train = pd.read_csv(train_csv_file, header=None).iloc[:, 1:]  # 特征列，从第二列开始
# y_train = pd.read_csv(train_csv_file, header=None).iloc[:, 0]   # 第一列是标签列

# # 读取测试集CSV文件
# X_test = pd.read_csv(test_csv_file, header=None).iloc[:, 1:]  # 特征列，从第二列开始
# y_test = pd.read_csv(test_csv_file, header=None).iloc[:, 0]   # 第一列是标签列

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

# # 打印分类报告，包括每个类别的准确率
# print("\n分类报告:")
# print(classification_report(y_test, y_pred, zero_division=1))

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
# data = pd.read_csv('D:/Users/空城丶/Desktop/fpga-code/openwolrd/del one/train1.csv')
# test = pd.read_csv('D:/Users/空城丶/Desktop/fpga-code/openwolrd/del one/test1.csv')
# # 准备数据：第一列是标签，剩余列是特征
# X_train = data.iloc[:, 1:].values
# y_train = data.iloc[:, 0].values
# X_test = test.iloc[:, 1:].values
# y_test = test.iloc[:, 0].values
# # 标准化特征
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.fit_transform(X_test)

# # 将数据转换为PyTorch张量，并调整为适合1D卷积的形状
# X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(1)  # 添加一个通道维度
# X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(1)
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

# # 将预测和真实标签转换为NumPy数组
# y_test_np = y_test_tensor.numpy()
# predicted_np = predicted.numpy()

# # 创建一个DataFrame来存储真实标签和预测标签
# predicted_true_labels = pd.DataFrame({'True': y_test_np, 'Predicted': predicted_np})

# # 打印按类别分组的预测和真实标签对比
# print("Predicted and True labels by class:")
# grouped_labels = predicted_true_labels.groupby(['True', 'Predicted']).size().unstack().fillna(0)
# print(grouped_labels)



# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report
# from sklearn.preprocessing import MinMaxScaler
# import numpy as np

# # 读取CSV文件
# data = pd.read_csv('D:/Users/空城丶/Desktop/fpga-code/openwolrd/del three/train no fmv.csv',header=None)
# test = pd.read_csv('D:/Users/空城丶/Desktop/fpga-code/openwolrd/del three/test no fmv.csv', header=None)

# # 分离特征和标签
# X_train = data.iloc[:, 1:]  # 特征列，从第二列开始
# y_train = data.iloc[:, 0]
# X_test = test.iloc[:, 1:]
# y_test = test.iloc[:, 0]  # 第一列是标签列

# unique_classes = y_train.unique()  # 获取所有已知类型标签

# # 创建MinMaxScaler实例
# scaler = MinMaxScaler()

# # 训练数据集归一化
# X_train_scaled = scaler.fit_transform(X_train)

# # 测试数据集归一化
# X_test_scaled = scaler.transform(X_test)

# # 记录每个二分类器的预测结果
# predictions = pd.DataFrame()

# # 对每个已知类型创建一个二分类器
# for label in unique_classes:
#     # 标记当前类别为正样本（1），其余类别为负样本（0）
#     y_train_binary = np.where(y_train == label, 1, 0)
#     y_test_binary = np.where(y_test == label, 1, 0)

#     # 创建并训练随机森林模型
#     rf = RandomForestClassifier(n_estimators=100, random_state=42)
#     rf.fit(X_train_scaled, y_train_binary)
    
#     # 对测试集进行预测
#     y_pred_binary = rf.predict_proba(X_test_scaled)[:, 1]  # 获取正类的预测概率
#     predictions[label] = y_pred_binary  # 将每个分类器的预测概率存入DataFrame

# # 获取每个样本的预测标签（概率最高的类别）
# predicted_labels = predictions.idxmax(axis=1)

# # 计算准确率和分类报告
# print("\n分类报告:")
# print(classification_report(y_test, predicted_labels, zero_division=1))

# # 打印真实标签和预测标签的对比，按类别分组
# predicted_true_labels = pd.DataFrame({'True': y_test, 'Predicted': predicted_labels})
# print("\nPredicted and True labels by class:")
# print(predicted_true_labels.groupby(['True', 'Predicted']).size().unstack().fillna(0))


# import pandas as pd
# from sklearn.model_selection import GridSearchCV
# from sklearn.svm import SVC
# from sklearn.metrics import classification_report
# from sklearn.preprocessing import StandardScaler
# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.pipeline import Pipeline
# import numpy as np

# # 请确保CSV文件的路径正确
# train_csv_file = 'D:/Users/空城丶/Desktop/fpga-code/openwolrd/del one/train4.csv'
# test_csv_file = 'D:/Users/空城丶/Desktop/fpga-code/openwolrd/del one/test4.csv'

# # 读取训练集CSV文件
# X_train = pd.read_csv(train_csv_file, header=None).iloc[:, 1:]
# y_train = pd.read_csv(train_csv_file, header=None).iloc[:, 0]

# # 读取测试集CSV文件
# X_test = pd.read_csv(test_csv_file, header=None).iloc[:, 1:]
# y_test = pd.read_csv(test_csv_file, header=None).iloc[:, 0]

# # 特征缩放
# scaler = StandardScaler()

# # SVM模型
# svm = SVC(kernel='rbf', probability=True)

# # 参数网格搜索
# param_grid = {'C': [10], 'gamma': [0.01]}

# # 创建带有网格搜索的OneVsRest分类器
# ovo_classifier = OneVsRestClassifier(GridSearchCV(svm, param_grid, cv=5))

# # 使用Pipeline将缩放器和分类器结合
# pipeline = Pipeline([
#     ('scaler', scaler),
#     ('classifier', ovo_classifier)
# ])

# # 训练模型
# pipeline.fit(X_train, y_train)

# # 获取最佳参数
# best_params = [est.best_params_ for est in ovo_classifier.estimators_]
# print("Best parameters for each class classifier:", best_params)

# # 预测并打印分类报告
# y_pred = pipeline.predict(X_test)
# print(classification_report(y_test, y_pred, zero_division=1))


# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report
# from sklearn.preprocessing import MinMaxScaler
# from collections import defaultdict

# # CSV文件路径
# csv_file = 'D:/Users/空城丶/Desktop/fpga-code/data-analysis/CHUNK_SIZE/data 4-10-200-1000/train.csv'

# # 读取CSV文件
# data = pd.read_csv(csv_file)

# # 分离特征和标签
# X = data.iloc[:, 1:]  # 特征列，从第二列开始
# y = data.iloc[:, 0]   # 第一列是标签列

# # 指定多个类型作为未类，例如类型 "0", "1", "2", "4"
# unknown_types = [2, 3, 4, 5]  # 将类型 0, 1, 2, 3 设为未知类型
# print(f"指定的未知类型为: {unknown_types}")

# # 将指定的未知类型标签更改为 "未知"
# y_modified = y.copy()
# y_modified[y_modified.isin(unknown_types)] = -1  # 使用 -1 表示未知类型

# # 创建MinMaxScaler实例
# scaler = MinMaxScaler()
# X_scaled = scaler.fit_transform(X)  # 将整个数据集进行归一化处理

# # 提前划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_modified, test_size=0.2, random_state=42)

# # 初始化存储分类器的字典
# classifiers = {}

# # 逐个已知类型（排除指定的未知类型）进行二分类训练
# for known_type in y.unique():
#     if known_type in unknown_types:
#         continue  # 跳过指定的未知类型

#     # 针对当前分类器的目标类型生成二分类标签：目标类型为1，其他类型（包括未知类型）为0
#     y_train_binary = np.where(y_train == known_type, 1, 0)

#     # 创建并训练随机森林模型
#     clf = RandomForestClassifier(n_estimators=100, random_state=42)
#     clf.fit(X_train, y_train_binary)
#     classifiers[known_type] = clf  # 存储分类器

# # 多分类预测
# y_pred_multi = []
# for sample in X_test:
#     # 字典存储每个分类器的预测得分
#     votes = defaultdict(float)
#     for known_type, clf in classifiers.items():
#         # 获取当前分类器对该样本属于该类的概率
#         prob = clf.predict_proba([sample])[0][1]  # 获取属于正类（1）的概率
#         votes[known_type] = prob  # 将分类器的预测概率添加到投票中

#     # 从投票结果中选择概率最高的类别作为最终预测
#     predicted_class = max(votes, key=votes.get)
#     y_pred_multi.append(predicted_class)

# # 输出整体多分类的分类报告，包括未知类型
# print("\n多分类模型的分类报告(包括未知类型):")
# print(classification_report(y_test, y_pred_multi))


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict

# CSV文件路径
csv_file = 'D:/Users/空城丶/Desktop/fpga-code/compare1/del four/train no cfmv.csv'

# 读取CSV文件
data = pd.read_csv(csv_file)

# 分离特征和标签
X = data.iloc[:, 1:]  # 特征列，从第二列开始
y = data.iloc[:, 0]   # 第一列是标签列

# 指定多个类型作为未知类型，例如类型 "0" 和 "3"
unknown_types = [2, 3, 4, 5]  # 将类型 0 和 3 设为未知类型
print(f"指定的未知类型为: {unknown_types}")

# 将指定的未知类型标签更改为 "未知"
y_modified = y.copy()
y_modified[y_modified.isin(unknown_types)] = -1  # 使用 -1 表示未知类型

# 创建MinMaxScaler实例
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)  # 将整个数据集进行归一化处理

# 提前划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_modified, test_size=0.2, random_state=42)

# 初始化存储分类器的字典
classifiers = {}

# 定义参数网格
param_grid = {
    'C': [10],  # 您可以尝试不同的C值
    'gamma': [0.01],  # 您可以尝试不同的gamma值
}

# 逐个已知类型（排除指定的未知类型）进行二分类训练
for known_type in y.unique():
    if known_type in unknown_types:
        continue  # 跳过指定的未知类型

    # 针对当前分类器的目标类型生成二分类标签：目标类型为1，其他类型（包括未知类型）为0
    y_train_binary = np.where(y_train == known_type, 1, 0)

    # 使用 GridSearchCV 搜索最佳参数
    grid_search = GridSearchCV(SVC(kernel='rbf', probability=True), param_grid, cv=5)
    grid_search.fit(X_train, y_train_binary)
    
    # 使用最佳参数训练模型
    best_model = grid_search.best_estimator_
    classifiers[known_type] = best_model  # 存储最佳模型

# 多分类预测
y_pred_multi = []
for sample in X_test:
    # 字典存储每个分类器的预测得分
    votes = defaultdict(float)
    for known_type, clf in classifiers.items():
        # 获取当前分类器对该样本属于该类的概率
        prob = clf.predict_proba([sample])[0][1]  # 获取属于正类（1）的概率
        votes[known_type] = prob  # 将分类器的预测概率添加到投票中

    # 从投票结果中选择概率最高的类别作为最终预测
    predicted_class = max(votes, key=votes.get)
    y_pred_multi.append(predicted_class)

# 输出整体多分类的分类报告，包括未知类型
print("\n多分类模型的分类报告(包括未知类型):")
print(classification_report(y_test, y_pred_multi))
