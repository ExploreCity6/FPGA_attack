import pandas as pd
from sklearn.model_selection import train_test_split
from half_transcend_ce import *
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score 
from sklearn.neighbors import KNeighborsClassifier
import half_ce_siml_multi as aaa 
from sklearn.preprocessing import LabelEncoder
import numpy as np
from my_tool import *
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
# 如果是分类任务，需要对类别标签进行编码


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

def mlp_split(data):
    data = shuffle(data, random_state=7).reset_index(drop=True)
    train = data.iloc[:int(len(data)*2/3), :data.shape[1]].reset_index(drop=True)
    cal = data.iloc[int(len(data)*2/3)+1:, :data.shape[1]].reset_index(drop=True)

    return train, cal
data = pd.read_csv('D:/Users/空城丶/Desktop/half_transcend_ce/data 4-10-200-1000/train.csv')

# 创建测试数据的副本·
# random = test_data.copy()

# # 将除了第 257 列之外的所有列都置为 0
# random.iloc[:, :256] = 0

# data = pd.concat([data, random], ignore_index=True)

# 剩余的数据保留在测试数据中

train, test = train_test_split(data, test_size=0.2, random_state=42)

train_train,train_cal=mlp_split(train)
y_train = train_train.iloc[:, 0]
X_train = train_train.iloc[:, 1:]
y_cal = train_cal.iloc[:, 0]
X_cal = train_cal.iloc[:, 1:]
y_test = test.iloc[:, 0]
X_test = test.iloc[:, 1:]

# 创建一个StandardScaler对象
scaler = StandardScaler()

# 使用训练数据来拟合缩放器，并缩放训练数据
X_train_scaled = scaler.fit_transform(X_train)

# 使用已经拟合的缩放器来缩放验证和测试数据
X_cal_scaled = scaler.transform(X_cal)
X_test_scaled = scaler.transform(X_test)

# model = RandomForestClassifier(random_state=42)
# model = RandomForestClassifier(n_estimators=100, random_state=42)
param_grid = {
    'C': [10],  # 尝试不同的正则化参数C
    'gamma': [0.01],  # 尝试不同的gamma值
}

# 创建带有参数网格的网格搜索对象
grid_search = GridSearchCV(SVC(kernel='rbf', probability=True), param_grid, cv=5)

# 执行网格搜索
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("Best parameters:", grid_search.best_params_)

# 使用最佳参数创建模型
best_model = grid_search.best_estimator_

# y_combined = np.concatenate((y_train, y_test), axis=0)
# encoder.fit_transform(y_combined)
# y_train = encoder.transform(y_train)
# y_cal = encoder.transform(y_cal)
# y_test = encoder.transform(y_test)


# print(np.unique(y_train))
# print(np.unique(y_cal))
# 在训练集上训练分类器
best_model.fit(X_train_scaled, y_train)
# print("ok")
# 在测试集上进行预测
y_pred = best_model.predict(X_test_scaled)
# print("ok")
train_prob, cal_prob, cal_y_pred, test_prob, test_y_pred=svm_calculate_ncm(X_train_scaled,X_cal_scaled,X_test_scaled,best_model)
keep_mask, reject_rate, order_idx, X_anom_score=aaa.start_half_transcend(train_prob,y_train,cal_prob,y_cal,cal_y_pred,test_prob,y_test,test_y_pred)
# print(train_prob)
# print(cal_prob)
# print(test_prob)
print(keep_mask)
print(X_anom_score)
# print(test_y_pred)
# print(y_test)

