import joblib
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# df = pd.read_excel('./data/position5.xlsx')
# # y = df['nongdu'].values
# # X = df.iloc[:, 1:].values  # 特征变量，从第2列开始到最后
#
# # 提取 y 为 -4 的所有行
# filtered_rows = df[df['nongdu'] == -4]
#
# # 提取特定列的数据（如 X 和 y）
# filtered_y = filtered_rows['nongdu'].values
# filtered_X = filtered_rows.iloc[:, 1:].values  # 从第2列开始到最后的特征变量
#
# if filtered_X.shape[0] >= 100:
#     first_100_rows = filtered_X[:50, :]  # 前 100 行
#     last_100_rows = filtered_X[-50:, :]  # 后 100 行
# else:
#     raise ValueError("filtered_X does not have at least 200 rows.")
#
# # 输出结果
# print(last_100_rows.shape)
# print(first_100_rows.shape)
#
# # 标准化特征数据
# scaler = StandardScaler()
# features_scaled = scaler.fit_transform(last_100_rows)
#
# # PCA降维
# pca = PCA(n_components=1)  # 只取第一个主成分
# pca_values = pca.fit_transform(features_scaled)
# # 转换形状，确保是二维数组
# pca_values = pca_values.flatten()  # 将形状从 (100, 1) 转为 (100,)
#
# # 计算PCA值的标准偏差
# pca_std = np.std(pca_values)

# 禁用科学计数法并设置小数点后保留三位
# np.set_printoptions(suppress=True, precision=3)
#
# # 将 PCA 值转换为 DataFrame，并保存为一行
# df = pd.DataFrame([pca_values], columns=[f'PC{i+1}' for i in range(len(pca_values))])
# # 保存到 Excel 文件
# output_file = "./data/xin2/data1.csv"
# df.to_csv(output_file, index=False, header=True)

# 绘制PCA值的散点图
# plt.figure(figsize=(10, 6))
# plt.scatter(range(1, len(pca_values) + 1), pca_values, color='b', marker='o', label='PCA Values')
# plt.xlabel('Sample Index')
# plt.ylabel('PCA Value')
# plt.title('PCA Value for Each Sample (5 position)')

# 显示标准偏差的参考线（可以作为波动范围的参考）
# plt.axhline(y=np.mean(pca_values) + pca_std, color='r', linestyle='--', label=f'Mean + 1 Std Dev ({pca_std:.2f})')
# plt.axhline(y=np.mean(pca_values) - pca_std, color='r', linestyle='--', label=f'Mean - 1 Std Dev ({pca_std:.2f})')

# plt.legend()
# # plt.grid(True)
# plt.show()

# 输出PCA值的标准偏差
# print(f"PCA values standard deviation: {pca_std:.2f}")


a = [-4, -5, -6, -7, -8, -9]
b = [34.65, 37.77, 39.88, 41.27, 40.37, 41.05]

a_5 = [-4, -5, -6, -7, -8, -9]
b_5 = [55.60, 55.46, 55.40, 59.06, 63.73, 60.12]

# 创建图形
plt.figure(figsize=(8, 5))

# 绘制折线图
# plt.plot(a, b, marker='o', linestyle='-', color='b', label='std')
plt.plot(a_5, b_5, marker='o', linestyle='-', color='b', label='std')

# 设置标题和标签
plt.title('Std for Concentration (5 position)')
plt.xlabel('Concentration')
plt.ylabel('Std Value')

# 显示图例
plt.legend()

# 显示网格
plt.grid(True)

# 显示图形
plt.show()


# import pandas as pd
# from sklearn.metrics import mean_squared_error, r2_score
# import ast
#
# # Load the CSV file
# file_path = './p11.csv'
# data = pd.read_csv(file_path)
#
# # Display the first few rows to understand the structure
# data.head()
#
# # Parse y_test and y_pred columns into actual lists
# y_test = [item for sublist in data['y_test'].apply(ast.literal_eval) for item in sublist]
# y_pred = [item for sublist in data['y_pred'].apply(ast.literal_eval) for item in sublist]
#
#
# print(len(y_test))
# print(y_test)
#
# # Calculate MSE and R²
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
#
# print(mse, r2)
#
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # 设置绘图样式
# sns.set(style="whitegrid")
#
# # 绘制拟合效果图
# plt.figure(figsize=(8, 6))
# plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label="Predicted vs True")
# plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label="Ideal Fit")
#
# # 图形标题和标签
# plt.title("Fit between Predicted and True Values", fontsize=14)
# plt.xlabel("True Values (y_test)", fontsize=12)
# plt.ylabel("Predicted Values (y_pred)", fontsize=12)
# plt.legend(fontsize=12)
# plt.tight_layout()
#
# # 显示图像
# plt.show()
