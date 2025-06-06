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

def random_calculate_ncm(train_X, cal_X, X_test, model):
    # 获取随机森林中的所有决策树
    trees = model.estimators_

    # 初始化存储每个决策树预测结果的列表
    train_tree_preds = []
    cal_tree_preds = []
    test_tree_preds = []

    # 对每棵树进行预测 (使用predict而非predict_proba)
    for tree in trees:
        train_tree_pred = tree.predict(train_X)
        cal_tree_pred = tree.predict(cal_X)
        test_tree_pred = tree.predict(X_test)

        # 将每个决策树的预测结果添加到列表中
        train_tree_preds.append(train_tree_pred)
        cal_tree_preds.append(cal_tree_pred)
        test_tree_preds.append(test_tree_pred)

    # 将预测结果转置，使得每个样本的决策树预测结果排列在一起
    train_tree_preds = np.array(train_tree_preds).T
    cal_tree_preds = np.array(cal_tree_preds).T
    test_tree_preds = np.array(test_tree_preds).T

    # 计算预测的一致性 (标准差)
    train_prob = [list(train_tree_preds[i]) for i in range(len(train_tree_preds))]
    cal_prob = [list(cal_tree_preds[i]) for i in range(len(cal_tree_preds))]
    test_prob = [list(test_tree_preds[i]) for i in range(len(test_tree_preds))]

    # 使用多数投票法预测最终的类标签
    cal_y_pred = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), axis=1, arr=cal_tree_preds)
    test_y_pred = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), axis=1, arr=test_tree_preds)

    return train_prob, cal_prob, cal_y_pred, test_prob, test_y_pred

def mlp_split(data):
    data = shuffle(data, random_state=7).reset_index(drop=True)
    train = data.iloc[:int(len(data)*2/3), :].reset_index(drop=True)
    cal = data.iloc[int(len(data)*2/3):, :].reset_index(drop=True)
    return train, cal

# 加载数据
data = pd.read_csv('D:/Users/空城丶/Desktop/half_transcend_ce/data 4-10-200-1000/train.csv')

# 数据预处理
train, test = train_test_split(data, test_size=0.2, random_state=42)
train_train, train_cal = mlp_split(train)

y_train = train_train.iloc[:, 0]
X_train = train_train.iloc[:, 1:]
y_cal = train_cal.iloc[:, 0]
X_cal = train_cal.iloc[:, 1:]
y_test = test.iloc[:, 0]
X_test = test.iloc[:, 1:]

# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_cal_scaled = scaler.transform(X_cal)
X_test_scaled = scaler.transform(X_test)

random_forest_classifier = RandomForestClassifier(
    random_state=42,
    n_estimators=200,  # 使用最佳参数中的树的数量
    max_depth=None,  # 使用最佳参数中的最大深度
    min_samples_split=2,  # 使用最佳参数中的分割所需的最小样本数
    min_samples_leaf=1,  # 叶子节点的最小样本数保持默认，因为它在网格搜索中没有被调整
    max_features='sqrt'  # 使用最佳参数中的最大特征数
)

# 训练模型
random_forest_classifier.fit(X_train_scaled, y_train)

# 预测测试集结果
y_pred = random_forest_classifier.predict(X_test_scaled)

# 假设 mlp_calculate_ncm 和 start_half_transcend 函数存在并且正确导入
train_prob, cal_prob, cal_y_pred, test_prob, test_y_pred = random_calculate_ncm(X_train_scaled, X_cal_scaled, X_test_scaled, random_forest_classifier)
# print(train_prob)
keep_mask, reject_rate, order_idx, X_anom_score = aaa.start_half_transcend(train_prob, y_train, cal_prob, y_cal, cal_y_pred, test_prob, y_test, test_y_pred)