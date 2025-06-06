import pandas as pd
from half_transcend_ce import *
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
import half_ce_siml_multi as aaa 
import numpy as np
from my_tool import *
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# def mlp_calculate_ncm(train_X, cal_X, X_test, model):
#     # 获取最佳模型 (RandomForestClassifier) 从GridSearchCV
#     best_model = model.best_estimator_

#     # 获取随机森林中的所有决策树
#     trees = best_model.estimators_

#     # 初始化存储每个决策树预测结果的列表
#     train_tree_preds = []
#     cal_tree_preds = []
#     test_tree_preds = []

#     # 对每棵树进行预测 (使用predict而非predict_proba)
#     for tree in trees:
#         train_tree_pred = tree.predict(train_X)
#         cal_tree_pred = tree.predict(cal_X)
#         test_tree_pred = tree.predict(X_test)

#         # 将每个决策树的预测结果添加到列表中
#         train_tree_preds.append(train_tree_pred)
#         cal_tree_preds.append(cal_tree_pred)
#         test_tree_preds.append(test_tree_pred)

#     # 将决策结果转置，使得每个样本的决策树预测结果排列在一起
#     train_tree_preds = np.array(train_tree_preds).T
#     cal_tree_preds = np.array(cal_tree_preds).T
#     test_tree_preds = np.array(test_tree_preds).T

#     # 获取所有可能的类别标签
#     classes = best_model.classes_

#     # 计算投票差异得分 (类似SVM的decision_function)
#     def compute_decision_function(tree_preds):
#         # 对于每个样本，计算每个类别的得分
#         decision_scores = []
#         for preds in tree_preds:
#             scores = []
#             for cls in classes:
#                 # 对每个类别，计算支持该类别的树的数量
#                 class_votes = np.sum(preds == cls)
#                 # 减去其他类别的树的投票数（总树数 - 支持该类别的树数）
#                 other_votes = len(preds) - class_votes
#                 # 类别得分 = 支持该类别的树数 - 支持其他类别的树数
#                 score = class_votes - other_votes
#                 scores.append(score)
#             decision_scores.append(scores)
#         return np.array(decision_scores)

#     # 计算每个样本的决策得分
#     train_prob = compute_decision_function(train_tree_preds)
#     cal_prob = compute_decision_function(cal_tree_preds)
#     test_prob = compute_decision_function(test_tree_preds)

#     # 使用多数投票法预测最终的类标签
#     cal_y_pred = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), axis=1, arr=cal_tree_preds)
#     test_y_pred = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), axis=1, arr=test_tree_preds)

#     # 返回类似于SVM decision_function的分数，以及最终的预测结果
#     return train_prob, cal_prob, cal_y_pred, test_prob, test_y_pred




import numpy as np
from scipy.stats import entropy

import numpy as np
from scipy.stats import entropy

def random_calculate_ncm(train_X, cal_X, X_test, model):
    # 获取随机森林中的所有决策树
    trees = model.estimators_

    # 初始化存储每个样本的路径深度
    def get_path_depth(X):
        n_samples = X.shape[0]
        path_depths = np.zeros(n_samples)

        # 对每棵树进行预测，并记录每个样本的路径深度
        for tree in trees:
            tree_paths = tree.apply(X)  # 获取每个样本到达叶子节点的路径
            path_depths += np.sum(tree_paths != -2, axis=1)  # 计算路径深度

        return path_depths

    # 计算每个数据集的路径深度
    train_depth = get_path_depth(train_X)
    cal_depth = get_path_depth(cal_X)
    test_depth = get_path_depth(X_test)

    # 定义一个函数来计算每个样本的非标准化熵
    def calculate_non_normalized_entropy(depths):
        # 使用路径深度计算每个样本的非标准化熵
        sample_entropies = []
        for depth in depths:
            # 计算非标准化熵，将路径深度作为参数
            entropies = entropy([depth, len(trees) - depth], base=2)
            sample_entropies.append(entropies)  # 每个样本的熵值
        return sample_entropies

    # 计算每个样本的非标准化熵
    train_entropy = calculate_non_normalized_entropy(train_depth)
    cal_entropy = calculate_non_normalized_entropy(cal_depth)
    test_entropy = calculate_non_normalized_entropy(test_depth)

    # 使用模型的 predict 方法预测最终的类标签
    cal_y_pred = model.predict(cal_X).tolist()
    test_y_pred = model.predict(X_test).tolist()

    return train_entropy, cal_entropy, cal_y_pred, test_entropy, test_y_pred








# def random_calculate_ncm(train_X, cal_X, X_test, model):
#     # 获取每个样本在所有决策树中叶节点的索引
#     train_leaf_indices = model.apply(train_X)  # shape: (n_samples, n_trees)
#     cal_leaf_indices = model.apply(cal_X)
#     test_leaf_indices = model.apply(X_test)

#     # 计算叶节点相似性距离
#     def calculate_leaf_similarity(leaf_indices1, leaf_indices2):
#         # 比较叶节点索引是否相同，得到相同叶节点的数量
#         similarity_matrix = np.zeros((leaf_indices1.shape[0], leaf_indices2.shape[0]))
#         for i, leaf1 in enumerate(leaf_indices1):
#             for j, leaf2 in enumerate(leaf_indices2):
#                 similarity_matrix[i, j] = np.sum(leaf1 == leaf2)
#         return similarity_matrix

#     # 计算训练集、校准集和测试集之间的叶节点相似度
#     train_similarity = calculate_leaf_similarity(train_leaf_indices, train_leaf_indices)
#     cal_similarity = calculate_leaf_similarity(cal_leaf_indices, cal_leaf_indices)
#     test_similarity = calculate_leaf_similarity(test_leaf_indices, test_leaf_indices)

#     # 使用模型的predict方法预测最终的类标签
#     cal_y_pred = model.predict(cal_X)
#     test_y_pred = model.predict(X_test)

#     return train_similarity, cal_similarity, cal_y_pred, test_similarity, test_y_pred

def mlp_split(data):
    data = shuffle(data, random_state=7).reset_index(drop=True)
    train = data.iloc[:int(len(data)*2/3), :].reset_index(drop=True)
    cal = data.iloc[int(len(data)*2/3):, :].reset_index(drop=True)
    return train, cal

# 加载数据
data = pd.read_csv('D:/Users/空城丶/Desktop/half_transcend_ce/openworld/del one/train1.csv',header=None)
test = pd.read_csv('D:/Users/空城丶/Desktop/half_transcend_ce/openworld/del one/test1.csv',header=None)

# 数据预处理
train_train, train_cal = mlp_split(data)

y_train = train_train.iloc[:, 0]
X_train = train_train.iloc[:, 1:]
y_cal = train_cal.iloc[:, 0]
X_cal = train_cal.iloc[:, 1:]
y_test = test.iloc[:, 0]
X_test = test.iloc[:, 1:]

# 标准化特征
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_cal_scaled = scaler.transform(X_cal)
X_test_scaled = scaler.transform(X_test)


# 定义参数网格
param_grid = {
    'n_estimators': [200],
    'max_depth': [None], 
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'max_features': ['sqrt']
}

# 创建随机森林分类器
random_forest_classifier = RandomForestClassifier(random_state=42)

# 使用GridSearchCV来找到最佳参数
grid_search = GridSearchCV(estimator=random_forest_classifier, param_grid=param_grid, cv=5, n_jobs=1)
grid_search.fit(X_train_scaled, y_train)
best_model = grid_search.best_estimator_

# model = RandomForestClassifier(random_state=42)
# model.fit(X_train_scaled, y_train)

# 调用 mlp_calculate_ncm 函数
train_prob, cal_prob, cal_y_pred, test_prob, test_y_pred = random_calculate_ncm(X_train_scaled, X_cal_scaled, X_test_scaled, best_model)
# print(train_prob)
print(test_prob)

# 假设 aaa.start_half_transcend 函数存在并且正确导入
keep_mask, reject_rate, order_idx, X_anom_score = aaa.start_half_transcend(train_prob, y_train, cal_prob, y_cal, cal_y_pred, test_prob, y_test, test_y_pred)
# print(X_anom_score)
# Given data
# data = keep_mask

# # Calculate the number of True and False in the first 41 and the rest of the elements
# first_41 = data[:41]
# rest = data[41:]

# true_count_first_41 = sum(first_41)
# false_count_first_41 = 41 - true_count_first_41

# true_count_rest = sum(rest)
# false_count_rest = len(rest) - true_count_rest

# print(true_count_first_41, false_count_first_41, true_count_rest, false_count_rest)