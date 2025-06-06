# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score

# # 加载数据
# file_path = 'D:/Users/空城丶/Desktop/fpga-code/openworld/del four/test no cfmv.csv'  # 请替换为您本地文件的路径
# data = pd.read_csv(file_path)

# # 假设第一列是标签，将其分离
# original_labels = data.iloc[:, 0]  # 原始类别
# X = data.drop(columns=data.columns[0])  # 剩余数据用于聚类

# # 使用肘部法确定最佳簇数
# wcss = []
# for k in range(1, 11):
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     kmeans.fit(X)
#     wcss.append(kmeans.inertia_)

# # 使用轮廓系数法进一步确定最佳簇数
# silhouette_scores = []
# for k in range(2, 11):
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     cluster_labels = kmeans.fit_predict(X)
#     silhouette_avg = silhouette_score(X, cluster_labels)
#     silhouette_scores.append(silhouette_avg)

# # 选择最佳的k值 (这里假设轮廓系数最高的k值为最佳值，或使用肘部法的“肘点”值)
# best_k = silhouette_scores.index(max(silhouette_scores)) + 2  # 因为 silhouette_scores 从 k=2 开始

# # 使用最佳k值进行 K-means 聚类
# kmeans = KMeans(n_clusters=best_k, random_state=42)
# data['Cluster'] = kmeans.fit_predict(X)  # 将聚类结果添加到数据框中

# # 添加原始类别列到结果数据框中
# data['Original_Label'] = original_labels

# # 检查每个原始类别是否被分配到单一的聚类簇，并记录详细信息
# consistency_results = []
# for label in original_labels.unique():
#     # 过滤出属于当前类别的数据点
#     label_data = data[data['Original_Label'] == label]
#     # 获取该类别的分配簇编号和各簇的数据点数量
#     cluster_counts = label_data['Cluster'].value_counts().to_dict()
    
#     is_consistent = len(cluster_counts) == 1  # True 表示都在同一簇中，False 表示分配到了多个簇
    
#     # 根据分配的簇数量保存详细的簇信息
#     if is_consistent:
#         # 如果数据全部在同一个簇，显示唯一簇编号和数据点数量
#         cluster_info = f"All in cluster {list(cluster_counts.keys())[0]} (count: {list(cluster_counts.values())[0]})"
#     else:
#         # 如果数据分布在多个簇，显示所有簇编号和每个簇的数据点数量
#         cluster_info = ", ".join([f"Cluster {cluster}: {count} points" for cluster, count in cluster_counts.items()])
    
#     consistency_results.append({
#         'Original_Label': label,
#         'Is_Consistently_Clustered': is_consistent,
#         'Assigned_Clusters_Info': cluster_info
#     })

# # 将结果保存到文件
# consistency_df = pd.DataFrame(consistency_results)
# consistency_output_path = 'cfmv.csv'
# consistency_df.to_csv(consistency_output_path, index=False)


# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score
# import numpy as np

# # 加载数据
# file_path = 'D:/Users/空城丶/Desktop/fpga-code/openworld/del three/test no fmv.csv'  # 请替换为您本地文件的路径
# data = pd.read_csv(file_path)

# # 假设第一列是标签，将其分离
# original_labels = data.iloc[:, 0]  # 原始类别
# X = data.drop(columns=data.columns[0])  # 剩余数据用于聚类

# # 使用肘部法确定最佳簇数
# wcss = []
# for k in range(1, 11):
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     kmeans.fit(X)
#     wcss.append(kmeans.inertia_)

# # 计算二阶差分找到“肘点”作为最佳 k 值
# wcss_diff = np.diff(wcss, 2)  # 计算二阶差分
# best_k = np.argmin(wcss_diff) + 2  # 找到二阶差分最小的点，并调整索引

# # 输出最佳 k 值
# print(f'使用肘部法确定的最佳簇数 k: {best_k}')

# # 使用最佳 k 值进行 K-means 聚类
# kmeans = KMeans(n_clusters=best_k, random_state=42)
# data['Cluster'] = kmeans.fit_predict(X)  # 将聚类结果添加到数据框中

# # 添加原始类别列到结果数据框中
# data['Original_Label'] = original_labels

# # 检查每个原始类别是否被分配到单一的聚类簇，并记录详细信息
# consistency_results = []
# for label in original_labels.unique():
#     # 过滤出属于当前类别的数据点
#     label_data = data[data['Original_Label'] == label]
#     # 获取该类别的分配簇编号和各簇的数据点数量
#     cluster_counts = label_data['Cluster'].value_counts().to_dict()
    
#     is_consistent = len(cluster_counts) == 1  # True 表示都在同一簇中，False 表示分配到了多个簇
    
#     # 根据分配的簇数量保存详细的簇信息
#     if is_consistent:
#         # 如果数据全部在同一个簇，显示唯一簇编号和数据点数量
#         cluster_info = f"All in cluster {list(cluster_counts.keys())[0]} (count: {list(cluster_counts.values())[0]})"
#     else:
#         # 如果数据分布在多个簇，显示所有簇编号和每个簇的数据点数量
#         cluster_info = ", ".join([f"Cluster {cluster}: {count} points" for cluster, count in cluster_counts.items()])
    
#     consistency_results.append({
#         'Original_Label': label,
#         'Is_Consistently_Clustered': is_consistent,
#         'Assigned_Clusters_Info': cluster_info
#     })

# # 将结果保存到文件
# consistency_df = pd.DataFrame(consistency_results)
# consistency_output_path = 'fmv.csv'
# consistency_df.to_csv(consistency_output_path, index=False)


# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score

# # 加载数据
# file_path = 'D:/Users/空城丶/Desktop/fpga-code/openworld/del two/test no mv.csv'  # 请替换为您本地文件的路径
# data = pd.read_csv(file_path)

# # 假设第一列是标签，将其分离
# original_labels = data.iloc[:, 0]  # 原始类别
# X = data.drop(columns=data.columns[0])  # 剩余数据用于聚类

# # 使用轮廓系数法确定最佳簇数
# silhouette_scores = []
# for k in range(2, 11):
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     cluster_labels = kmeans.fit_predict(X)
#     silhouette_avg = silhouette_score(X, cluster_labels)
#     silhouette_scores.append(silhouette_avg)

# # 选择最佳的k值 (轮廓系数最高的k值)
# best_k = silhouette_scores.index(max(silhouette_scores)) + 2  # 因为 silhouette_scores 从 k=2 开始

# # 使用最佳k值进行 K-means 聚类
# kmeans = KMeans(n_clusters=best_k, random_state=42)
# data['Cluster'] = kmeans.fit_predict(X)  # 将聚类结果添加到数据框中

# # 添加原始类别列到结果数据框中
# data['Original_Label'] = original_labels

# # 检查每个原始类别是否被分配到单一的聚类簇，并记录详细信息
# consistency_results = []
# for label in original_labels.unique():
#     # 过滤出属于当前类别的数据点
#     label_data = data[data['Original_Label'] == label]
#     # 获取该类别的分配簇编号和各簇的数据点数量
#     cluster_counts = label_data['Cluster'].value_counts().to_dict()
    
#     is_consistent = len(cluster_counts) == 1  # True 表示都在同一簇中，False 表示分配到了多个簇
    
#     # 根据分配的簇数量保存详细的簇信息
#     if is_consistent:
#         # 如果数据全部在同一个簇，显示唯一簇编号和数据点数量
#         cluster_info = f"All in cluster {list(cluster_counts.keys())[0]} (count: {list(cluster_counts.values())[0]})"
#     else:
#         # 如果数据分布在多个簇，显示所有簇编号和每个簇的数据点数量
#         cluster_info = ", ".join([f"Cluster {cluster}: {count} points" for cluster, count in cluster_counts.items()])
    
#     consistency_results.append({
#         'Original_Label': label,
#         'Is_Consistently_Clustered': is_consistent,
#         'Assigned_Clusters_Info': cluster_info
#     })

# # 将结果保存到文件
# consistency_df = pd.DataFrame(consistency_results)
# consistency_output_path = 'mv.csv'
# consistency_df.to_csv(consistency_output_path, index=False)
# print(f'Consistency results saved to {consistency_output_path}')


# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score

# # 加载数据
# file_path = 'D:/Users/空城丶/Desktop/fpga-code/openworld/del two/test no mv.csv'  # 请替换为您本地文件的路径
# data = pd.read_csv(file_path)

# # 假设第一列是标签，将其分离
# original_labels = data.iloc[:, 0]  # 原始类别
# X = data.drop(columns=data.columns[0])  # 剩余数据用于聚类

# # 使用轮廓系数法确定最佳簇数
# silhouette_scores = []
# for k in range(2, 11):
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     cluster_labels = kmeans.fit_predict(X)
#     silhouette_avg = silhouette_score(X, cluster_labels)
#     silhouette_scores.append(silhouette_avg)

# # 选择最佳的k值 (轮廓系数最高的k值)
# best_k = silhouette_scores.index(max(silhouette_scores)) + 2  # 因为 silhouette_scores 从 k=2 开始

# # 使用最佳k值进行 K-means 聚类
# kmeans = KMeans(n_clusters=best_k, random_state=42)
# data['Cluster'] = kmeans.fit_predict(X)  # 将聚类结果添加到数据框中

# # 添加原始类别列到结果数据框中
# data['Original_Label'] = original_labels

# # 检查每个原始类别是否被分配到单一的聚类簇，并记录详细信息
# consistency_results = []
# for label in original_labels.unique():
#     # 过滤出属于当前类别的数据点
#     label_data = data[data['Original_Label'] == label]
#     # 获取该类别的分配簇编号和各簇的数据点数量
#     cluster_counts = label_data['Cluster'].value_counts().to_dict()
    
#     is_consistent = len(cluster_counts) == 1  # True 表示都在同一簇中，False 表示分配到了多个簇
    
#     # 根据分配的簇数量保存详细的簇信息
#     if is_consistent:
#         # 如果数据全部在同一个簇，显示唯一簇编号和数据点数量
#         cluster_info = f"All in cluster {list(cluster_counts.keys())[0]} (count: {list(cluster_counts.values())[0]})"
#     else:
#         # 如果数据分布在多个簇，显示所有簇编号和每个簇的数据点数量
#         cluster_info = ", ".join([f"Cluster {cluster}: {count} points" for cluster, count in cluster_counts.items()])
    
#     consistency_results.append({
#         'Original_Label': label,
#         'Is_Consistently_Clustered': is_consistent,
#         'Assigned_Clusters_Info': cluster_info
#     })

# # 将结果保存到文件
# consistency_df = pd.DataFrame(consistency_results)
# consistency_output_path = 'mv.csv'
# consistency_df.to_csv(consistency_output_path, index=False)
# print(f'Consistency results saved to {consistency_output_path}')

# # 可视化聚类结果
# plt.figure(figsize=(8, 6))

# # 如果数据是二维的，直接绘制聚类结果
# if X.shape[1] == 2:
#     plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=data['Cluster'], cmap='viridis', marker='o')
#     plt.xlabel('Feature 1')
#     plt.ylabel('Feature 2')
# else:
#     # 如果数据维度超过2，使用PCA进行降维
#     from sklearn.decomposition import PCA
#     pca = PCA(n_components=2)
#     X_pca = pca.fit_transform(X)
#     plt.scatter(X_pca[:, 0], X_pca[:, 1], c=data['Cluster'], cmap='viridis', marker='o')
#     plt.xlabel('PCA Component 1')
#     plt.ylabel('PCA Component 2')

# plt.title(f'K-means Clustering (k={best_k})')
# plt.colorbar(label='Cluster Label')
# plt.show()



import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# 加载数据
file_path = 'D:/Users/空城丶/Desktop/fpga-code/openworld/del two/test no mv.csv'  # 请替换为您本地文件的路径
data = pd.read_csv(file_path)

# 假设第一列是标签，将其分离
original_labels = data.iloc[:, 0]  # 原始类别
X = data.drop(columns=data.columns[0])  # 剩余数据用于聚类

# 使用轮廓系数法确定最佳簇数
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# 选择最佳的k值 (轮廓系数最高的k值)
best_k = silhouette_scores.index(max(silhouette_scores)) + 2  # 因为 silhouette_scores 从 k=2 开始

# 使用最佳k值进行 K-means 聚类
kmeans = KMeans(n_clusters=best_k, random_state=42)
cluster_labels = kmeans.fit_predict(X)  # 聚类结果

# 将聚类结果添加到数据框中
data['Cluster'] = cluster_labels
data['Original_Label'] = original_labels

# 定义一个函数，将聚类标签映射到真实标签
def map_cluster_to_class(cluster_labels, original_labels):
    cluster_to_class = {}
    # 获取聚类簇与真实类别的映射关系
    for cluster in np.unique(cluster_labels):
        class_label = np.bincount(original_labels[cluster_labels == cluster]).argmax()  # 获取聚类簇中最多的真实类别
        cluster_to_class[cluster] = class_label
    # 将聚类标签映射为真实标签
    mapped_labels = np.array([cluster_to_class[cluster] for cluster in cluster_labels])
    return mapped_labels

# 映射聚类标签到真实标签
mapped_labels = map_cluster_to_class(cluster_labels, original_labels)

# 计算准确率、精度、召回率和F1分数
accuracy = accuracy_score(original_labels, mapped_labels)
precision = precision_score(original_labels, mapped_labels, average='weighted', zero_division=0)
recall = recall_score(original_labels, mapped_labels, average='weighted', zero_division=0)
f1 = f1_score(original_labels, mapped_labels, average='weighted', zero_division=0)

# 输出分类效果指标
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# 检查每个原始类别是否被分配到单一的聚类簇，并记录详细信息
consistency_results = []
for label in original_labels.unique():
    # 过滤出属于当前类别的数据点
    label_data = data[data['Original_Label'] == label]
    # 获取该类别的分配簇编号和各簇的数据点数量
    cluster_counts = label_data['Cluster'].value_counts().to_dict()
    
    is_consistent = len(cluster_counts) == 1  # True 表示都在同一簇中，False 表示分配到了多个簇
    
    # 根据分配的簇数量保存详细的簇信息
    if is_consistent:
        # 如果数据全部在同一个簇，显示唯一簇编号和数据点数量
        cluster_info = f"All in cluster {list(cluster_counts.keys())[0]} (count: {list(cluster_counts.values())[0]})"
    else:
        # 如果数据分布在多个簇，显示所有簇编号和每个簇的数据点数量
        cluster_info = ", ".join([f"Cluster {cluster}: {count} points" for cluster, count in cluster_counts.items()])
    
    consistency_results.append({
        'Original_Label': label,
        'Is_Consistently_Clustered': is_consistent,
        'Assigned_Clusters_Info': cluster_info
    })

# 将结果保存到文件
consistency_df = pd.DataFrame(consistency_results)
consistency_output_path = 'mv.csv'
consistency_df.to_csv(consistency_output_path, index=False)
print(f'Consistency results saved to {consistency_output_path}')
