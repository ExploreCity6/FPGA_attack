# import pandas as pd
# from sklearn.model_selection import train_test_split
# from half_transcend_ce import *
# from sklearn.utils import shuffle
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.metrics import accuracy_score 
# from sklearn.neighbors import KNeighborsClassifier
# import half_ce_siml_multi as aaa 
# from sklearn.preprocessing import LabelEncoder
# import numpy as np
# from my_tool import *
# from sklearn.model_selection import GridSearchCV
# from sklearn.svm import SVC
# from sklearn.preprocessing import StandardScaler
# # 如果是分类任务，需要对类别标签进行编码
# encoder = LabelEncoder()
# def mlp_split(data):
#     data = shuffle(data, random_state=7).reset_index(drop=True)
#     train = data.iloc[:int(len(data)*2/3), :data.shape[1]].reset_index(drop=True)
#     cal = data.iloc[int(len(data)*2/3)+1:, :data.shape[1]].reset_index(drop=True)
#     return train, cal
# # def mlp_split(data):
# #     # 随机打乱数据集
# #     data = shuffle(data, random_state=7).reset_index(drop=True)
# #     # 计算分割点
# #     split_index = int(len(data) * 2 / 3)
# #     # 分割数据集为训练集和校准集
# #     train = data.iloc[:split_index, :].reset_index(drop=True)
# #     cal = data.iloc[split_index:, :].reset_index(drop=True)
# #     # 返回训练集和校准集
# #     return train, cal


# data = pd.read_csv('D:/Users/空城丶/Desktop/half_transcend_ce/openworld1/del one/train1.csv',header=None)
# test = pd.read_csv('D:/Users/空城丶/Desktop/half_transcend_ce/openworld1/del one/test1.csv',header=None)
# train_train,train_cal=mlp_split(data)


# X_train = train_train.iloc[:, 1:]
# y_train = train_train.iloc[:, 0]
# X_test = test.iloc[:, 1:]
# y_test = test.iloc[:, 0]

# y_cal = train_cal.iloc[:,0]
# X_cal = train_cal.iloc[:, 1:]

# # 创建一个StandardScaler对象
# scaler = StandardScaler()

# # 使用训练数据来拟合缩放器，并缩放训练数据
# X_train_scaled = scaler.fit_transform(X_train)

# # 使用已经拟合的缩放器来缩放验证和测试数据
# X_cal_scaled = scaler.transform(X_cal)
# X_test_scaled = scaler.transform(X_test)


# # model = RandomForestClassifier(n_estimators=100, random_state=42)
# param_grid = {
#     'C': [1, 10, 100, 1000],  # 尝试更大的C值
#     'gamma': [0.001, 0.01, ],  # 尝试不同的gamma值
# }

# # 创建带有参数网格的网格搜索对象
# grid_search = GridSearchCV(SVC(kernel='rbf', probability=True), param_grid, cv=5)

# # 执行网格搜索
# grid_search.fit(X_train_scaled, y_train)

# # 输出最佳参数
# print("Best parameters:", grid_search.best_params_)

# # 使用最佳参数创建模型
# best_model = grid_search.best_estimator_

# y_combined = np.concatenate((y_train, y_test), axis=0)
# encoder.fit_transform(y_combined)
# y_train = encoder.transform(y_train)
# y_cal = encoder.transform(y_cal)
# y_test = encoder.transform(y_test)


# # print(np.unique(y_train))
# # print(np.unique(y_cal))
# # 在训练集上训练分类器
# best_model.fit(X_train_scaled, y_train)
# # print("ok")
# # 在测试集上进行预测
# y_pred = best_model.predict(X_test_scaled)
# # print(y_pred)
# # print("ok")
# train_prob, cal_prob, cal_y_pred, test_prob, test_y_pred=mlp_calculate_ncm(X_train_scaled,X_test_scaled,X_test_scaled,best_model)
# keep_mask, reject_rate, order_idx, X_anom_score=aaa.start_half_transcend(train_prob,y_train,cal_prob,y_cal,cal_y_pred,test_prob,y_test,test_y_pred)
# # print(train_prob)
# # print(test_prob)
# # print(keep_mask)
# # print(X_anom_score)
# # print(test_y_pred)
# # print(y_test)



import pandas as pd
from sklearn.model_selection import train_test_split
from half_ce_siml_multi import *
from sklearn.utils import shuffle
from sklearn.svm import SVC
import half_ce_siml_multi as aaa 
import numpy as np
from my_tool import *
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


def svm_calculate_ncm(train_X, cal_X, X_test, model):
    # import torch
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model.to('cpu')
    
    # 使用 SVM 的 decision_function 代替 predict_proba
    train_prob = model.decision_function(train_X)
    cal_prob = model.decision_function(cal_X)
    cal_y_pred = model.predict(cal_X)

    test_prob = model.decision_function(X_test)
    test_y_pred = model.predict(X_test)

    return train_prob, cal_prob, cal_y_pred, test_prob, test_y_pred



# def svm_calculate_ncm(train_X, train_y, cal_X, cal_y, X_test, model):
#     # 使用 SVM 的 decision_function 代替 predict_proba
#     train_scores = model.decision_function(train_X)
#     cal_scores = model.decision_function(cal_X)
#     test_scores = model.decision_function(X_test)

#     # 根据真实标签对分数进行处理
#     train_prob = [[-score, score] if label == 1 else [score, -score] 
#                   for score, label in zip(train_scores, train_y)]
#     cal_prob = [[-score, score] if label == 1 else [score, -score] 
#                 for score, label in zip(cal_scores, cal_y)]
#     test_prob = [[score, -score] for score in test_scores]

#     cal_y_pred = model.predict(cal_X)
#     test_y_pred = model.predict(X_test)

#     return train_prob, cal_prob, cal_y_pred, test_prob, test_y_pred


# def svm_calculate_ncm(train_X, cal_X, X_test, model):
#     # 使用 SVM 的 decision_function 代替 predict_proba
#     train_scores = model.decision_function(train_X)
#     cal_scores = model.decision_function(cal_X)
#     test_scores = model.decision_function(X_test)

#     # 将每个分数转换成 ncm[[], []] 的形式
#     train_prob = [[score, -score] for score in train_scores]
#     cal_prob = [[score, -score] for score in cal_scores]
#     test_prob = [[score, -score] for score in test_scores]

#     cal_y_pred = model.predict(cal_X)
#     test_y_pred = model.predict(X_test)

#     return train_prob, cal_prob, cal_y_pred, test_prob, test_y_pred



def mlp_split(data):
    data = shuffle(data, random_state=7).reset_index(drop=True)
    train = data.iloc[:int(len(data)*2/3), :data.shape[1]].reset_index(drop=True)
    cal = data.iloc[int(len(data)*2/3)+1:, :data.shape[1]].reset_index(drop=True)
    return train, cal
# def mlp_split(data):
#     # 随机打乱数据集
#     data = shuffle(data, random_state=7).reset_index(drop=True)
#     # 计算分割点
#     split_index = int(len(data) * 2 / 3)
#     # 分割数据集为训练集和校准集
#     train = data.iloc[:split_index, :].reset_index(drop=True)
#     cal = data.iloc[split_index:, :].reset_index(drop=True)
#     # 返回训练集和校准集
#     return train, cal

data = pd.read_csv('D:/Users/空城丶/Desktop/half_transcend_ce/openworld/del three/train no fmv.csv',header=None)
test = pd.read_csv('D:/Users/空城丶/Desktop/half_transcend_ce/openworld/del three/test no fmv.csv',header=None)
train_train,train_cal=mlp_split(data)


X_train = train_train.iloc[:, 1:]
y_train = train_train.iloc[:, 0]
X_test = test.iloc[:, 1:]
y_test = test.iloc[:, 0]

y_cal = train_cal.iloc[:,0]
X_cal = train_cal.iloc[:, 1:]

# 创建一个StandardScaler对象
scaler = StandardScaler()

# 使用训练数据来拟合缩放器，并缩放训练数据
X_train_scaled = scaler.fit_transform(X_train)

# 使用已经拟合的缩放器来缩放验证和测试数据
X_cal_scaled = scaler.transform(X_cal)
X_test_scaled = scaler.transform(X_test)


# # 使用最佳参数创建模型
# best_model = SVC(kernel='sigmoid', probability=True)

# y_combined = np.concatenate((y_train, y_test), axis=0)
# encoder.fit_transform(y_combined)
# y_train = encoder.transform(y_train)
# y_cal = encoder.transform(y_cal)
# y_test = encoder.transform(y_test)


# # print(np.unique(y_train))
# # print(np.unique(y_cal))
# # 在训练集上训练分类器
# best_model.fit(X_train_scaled, y_train)
# # print("ok")
# # 在测试集上进行预测
# y_pred = best_model.predict(X_test_scaled)

# model = RandomForestClassifier(random_state=42)
# model = RandomForestClassifier(n_estimators=100, random_state=42)
param_grid = {
    'C': [10],
    'gamma': [0.01],
}

grid_search = GridSearchCV(SVC(kernel='rbf', probability=True), param_grid, cv=5)

# 执行网格搜索
grid_search.fit(X_train, y_train)

# 使用最佳参数创建模型
best_model = grid_search.best_estimator_



# 在训练集上训练分类器
best_model.fit(X_train_scaled, y_train)
# print("ok")
# 在测试集上进行预测
y_pred = best_model.predict(X_test_scaled)


# print(y_pred)
# print("ok")


train_prob, cal_prob, cal_y_pred, test_prob, test_y_pred=mlp_calculate_ncm(X_train_scaled,X_cal_scaled,X_test_scaled,best_model)

# train_prob, cal_prob, cal_y_pred, test_prob, test_y_pred=svm_calculate_ncm(X_train_scaled,y_train,X_cal_scaled,y_cal,X_test_scaled,best_model)


# # 假设 test_prob 是一个包含每个样本对每个类别的概率的数组或矩阵
# # 为列命名为类0、类1、类2等
# test_prob_df = pd.DataFrame(test_prob, columns=[f'{i}' for i in range(len(test_prob[0]))])

# # 保存为 CSV 文件
# test_prob_df.to_csv('NCM.csv', index=False)


# train_prob, cal_prob, cal_y_pred, test_prob, test_y_pred=mlp_calculate_ncm(X_train_scaled,X_cal_scaled,X_test_scaled,best_model)
# test_prob = convertncm(test_prob, y_test)
# cal_prob = convertncm(cal_prob, y_cal)
# print(train_prob)


keep_mask, reject_rate, order_idx, X_anom_score=aaa.start_half_transcend(train_prob,y_train,cal_prob,y_cal,cal_y_pred,test_prob,y_test,test_y_pred)

print(X_anom_score)
# print(train_prob)
# print(test_prob)
# print(keep_mask)



import pandas as pd

# 假设 keep_mask 和 X_test, y_test 已经生成
# keep_mask 是布尔值数组，用于筛选测试集样本
# X_test 和 y_test 分别是测试集的特征和标签

# 如果 X_test 是 NumPy 数组，先转换为 DataFrame
if isinstance(X_test, np.ndarray):
    X_test = pd.DataFrame(X_test)

# 筛选出值为 True 的测试集样本
filtered_test = X_test[keep_mask].reset_index(drop=True)

# 筛选出对应的测试集标签
filtered_labels = y_test[keep_mask].reset_index(drop=True)

# 将特征和标签合并成一个 DataFrame
filtered_test_with_labels = pd.concat([filtered_labels, filtered_test], axis=1)

# # 设置列名，确保数据有意义
# filtered_test_with_labels.columns = ['Label'] + [f'Feature_{i+1}' for i in range(filtered_test.shape[1])]

# 保存为 CSV 文件
filtered_test_with_labels.to_csv('test no fmv.csv', index=False, header=False)




# # Given data
# data = keep_mask

# Calculate the number of True and False in the first 100 and the rest of the elements
# first_100 = data[:100]
# rest = data[100:]

# true_count_first_100 = sum(first_100)
# false_count_first_100 = 100 - true_count_first_100

# true_count_rest = sum(rest)
# false_count_rest = len(rest) - true_count_rest

# print(true_count_first_100, false_count_first_100, true_count_rest, false_count_rest)





# import pandas as pd

# # Load the CSV files
# test_y_pred_df = pd.read_csv('test_p_val_lists.csv')  # Replace with your actual file path
# test_prob_df = pd.read_csv('NCM.csv')  # Replace with your actual file path

# # Get the class with the maximum value for each row in test_y_pred
# test_y_pred_max_class = test_y_pred_df.idxmax(axis=1)

# # Get the first 100 and last 100 rows for test_y_pred (each class per row)
# first_100_test_y_pred = test_y_pred_max_class[:100]
# last_100_test_y_pred = test_y_pred_max_class[-100:]

# # Get the class with the maximum value for each row in test_prob
# test_prob_max_class = test_prob_df.idxmax(axis=1)

# # Get the first 100 and last 100 rows for test_prob (each class per row)
# first_100_test_prob = test_prob_max_class[:100]
# last_100_test_prob = test_prob_max_class[-100:]

# # Statistically count the occurrence of each class in the first 100 and last 100 for both datasets
# first_100_test_y_pred_counts = first_100_test_y_pred.value_counts().reset_index()
# first_100_test_y_pred_counts.columns = ['Class', 'Count_First_100']

# last_100_test_y_pred_counts = last_100_test_y_pred.value_counts().reset_index()
# last_100_test_y_pred_counts.columns = ['Class', 'Count_Last_100']

# first_100_test_prob_counts = first_100_test_prob.value_counts().reset_index()
# first_100_test_prob_counts.columns = ['Class', 'Count_First_100']

# last_100_test_prob_counts = last_100_test_prob.value_counts().reset_index()
# last_100_test_prob_counts.columns = ['Class', 'Count_Last_100']

# # Merge the counts for first and last 100 for each dataset
# test_y_pred_counts = pd.merge(first_100_test_y_pred_counts, last_100_test_y_pred_counts, on='Class', how='outer').fillna(0)
# test_prob_counts = pd.merge(first_100_test_prob_counts, last_100_test_prob_counts, on='Class', how='outer').fillna(0)

# # Save the results into CSV files
# test_y_pred_counts.to_csv('Cred.csv', index=False)
# test_prob_counts.to_csv('NCM1.csv', index=False)

# import pandas as pd

# # Load the CSV files
# test_y_pred_df = pd.read_csv('test_p_val_lists.csv')  # Replace with your actual file path
# test_prob_df = pd.read_csv('NCM.csv')  # Replace with your actual file path

# # Get the class with the maximum value for each row in test_y_pred
# test_y_pred_max_class = test_y_pred_df.idxmax(axis=1)

# # Get the first 100, middle 100, and last 100 rows for test_y_pred (each class per row)
# first_100_test_y_pred = test_y_pred_max_class[:100]
# middle_100_test_y_pred = test_y_pred_max_class[100:200]
# last_100_test_y_pred = test_y_pred_max_class[-100:]

# # Get the class with the maximum value for each row in test_prob
# test_prob_max_class = test_prob_df.idxmax(axis=1)

# # Get the first 100, middle 100, and last 100 rows for test_prob (each class per row)
# first_100_test_prob = test_prob_max_class[:100]
# middle_100_test_prob = test_prob_max_class[100:200]
# last_100_test_prob = test_prob_max_class[-100:]

# # Statistically count the occurrence of each class in the first 100, middle 100, and last 100 for both datasets
# first_100_test_y_pred_counts = first_100_test_y_pred.value_counts().reset_index()
# first_100_test_y_pred_counts.columns = ['Class', 'Count_First_100']

# middle_100_test_y_pred_counts = middle_100_test_y_pred.value_counts().reset_index()
# middle_100_test_y_pred_counts.columns = ['Class', 'Count_Middle_100']

# last_100_test_y_pred_counts = last_100_test_y_pred.value_counts().reset_index()
# last_100_test_y_pred_counts.columns = ['Class', 'Count_Last_100']

# first_100_test_prob_counts = first_100_test_prob.value_counts().reset_index()
# first_100_test_prob_counts.columns = ['Class', 'Count_First_100']

# middle_100_test_prob_counts = middle_100_test_prob.value_counts().reset_index()
# middle_100_test_prob_counts.columns = ['Class', 'Count_Middle_100']

# last_100_test_prob_counts = last_100_test_prob.value_counts().reset_index()
# last_100_test_prob_counts.columns = ['Class', 'Count_Last_100']

# # Merge the counts for first, middle, and last 100 for each dataset
# test_y_pred_counts = pd.merge(first_100_test_y_pred_counts, middle_100_test_y_pred_counts, on='Class', how='outer').fillna(0)
# test_y_pred_counts = pd.merge(test_y_pred_counts, last_100_test_y_pred_counts, on='Class', how='outer').fillna(0)

# test_prob_counts = pd.merge(first_100_test_prob_counts, middle_100_test_prob_counts, on='Class', how='outer').fillna(0)
# test_prob_counts = pd.merge(test_prob_counts, last_100_test_prob_counts, on='Class', how='outer').fillna(0)

# # Save the results into CSV files
# test_y_pred_counts.to_csv('Cred.csv', index=False)
# test_prob_counts.to_csv('NCM1.csv', index=False)


# import pandas as pd

# # Load the CSV files
# test_y_pred_df = pd.read_csv('test_p_val_lists.csv')  # Replace with your actual file path
# test_prob_df = pd.read_csv('NCM.csv')  # Replace with your actual file path

# # Get the class with the maximum value for each row in test_y_pred
# test_y_pred_max_class = test_y_pred_df.idxmax(axis=1)

# # Get the first 100, middle 100, middle next 100, and last 100 rows for test_y_pred (each class per row)
# first_100_test_y_pred = test_y_pred_max_class[:100]
# middle_100_test_y_pred = test_y_pred_max_class[100:200]
# middle_next_100_test_y_pred = test_y_pred_max_class[200:300]
# last_100_test_y_pred = test_y_pred_max_class[-100:]

# # Get the class with the maximum value for each row in test_prob
# test_prob_max_class = test_prob_df.idxmax(axis=1)

# # Get the first 100, middle 100, middle next 100, and last 100 rows for test_prob (each class per row)
# first_100_test_prob = test_prob_max_class[:100]
# middle_100_test_prob = test_prob_max_class[100:200]
# middle_next_100_test_prob = test_prob_max_class[200:300]
# last_100_test_prob = test_prob_max_class[-100:]

# # Statistically count the occurrence of each class in the first 100, middle 100, middle next 100, and last 100 for both datasets
# first_100_test_y_pred_counts = first_100_test_y_pred.value_counts().reset_index()
# first_100_test_y_pred_counts.columns = ['Class', 'Count_First_100']

# middle_100_test_y_pred_counts = middle_100_test_y_pred.value_counts().reset_index()
# middle_100_test_y_pred_counts.columns = ['Class', 'Count_Middle_100']

# middle_next_100_test_y_pred_counts = middle_next_100_test_y_pred.value_counts().reset_index()
# middle_next_100_test_y_pred_counts.columns = ['Class', 'Count_Middle_Next_100']

# last_100_test_y_pred_counts = last_100_test_y_pred.value_counts().reset_index()
# last_100_test_y_pred_counts.columns = ['Class', 'Count_Last_100']

# first_100_test_prob_counts = first_100_test_prob.value_counts().reset_index()
# first_100_test_prob_counts.columns = ['Class', 'Count_First_100']

# middle_100_test_prob_counts = middle_100_test_prob.value_counts().reset_index()
# middle_100_test_prob_counts.columns = ['Class', 'Count_Middle_100']

# middle_next_100_test_prob_counts = middle_next_100_test_prob.value_counts().reset_index()
# middle_next_100_test_prob_counts.columns = ['Class', 'Count_Middle_Next_100']

# last_100_test_prob_counts = last_100_test_prob.value_counts().reset_index()
# last_100_test_prob_counts.columns = ['Class', 'Count_Last_100']

# # Merge the counts for first, middle, middle next, and last 100 for each dataset
# test_y_pred_counts = pd.merge(first_100_test_y_pred_counts, middle_100_test_y_pred_counts, on='Class', how='outer').fillna(0)
# test_y_pred_counts = pd.merge(test_y_pred_counts, middle_next_100_test_y_pred_counts, on='Class', how='outer').fillna(0)
# test_y_pred_counts = pd.merge(test_y_pred_counts, last_100_test_y_pred_counts, on='Class', how='outer').fillna(0)

# test_prob_counts = pd.merge(first_100_test_prob_counts, middle_100_test_prob_counts, on='Class', how='outer').fillna(0)
# test_prob_counts = pd.merge(test_prob_counts, middle_next_100_test_prob_counts, on='Class', how='outer').fillna(0)
# test_prob_counts = pd.merge(test_prob_counts, last_100_test_prob_counts, on='Class', how='outer').fillna(0)

# # Save the results into CSV files
# test_y_pred_counts.to_csv('Cred.csv', index=False)
# test_prob_counts.to_csv('NCM1.csv', index=False)