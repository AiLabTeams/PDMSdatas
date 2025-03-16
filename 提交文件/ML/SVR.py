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
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# 使用SG进行滤波
def sg_filter(X, m, p, d):
    X_filtered = np.zeros_like(X)
    for i in range(X.shape[0]):
        X_filtered[i, :] = savgol_filter(X[i, :], window_length=m, polyorder=p, deriv=d)
    return X_filtered


# 读取第二批数据
# df = pd.read_excel('./PDMS1/position1.xlsx')
# y = df['nongdu'].values
# X = df.iloc[:, 1:].values  # 特征变量，从第2列开始到最后
#
# # 将第二批数据划分为训练集和验证集
# X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42)

# 读取第一批数据作为测试集
df_test = pd.read_excel('./PDMS2/position2_1.xlsx')
y_test = df_test['nongdu'].values
X_test = df_test.iloc[:, 1:].values  # 特征变量，从第2列开始到最后

# X_train = sg_filter(X_train, 9, 3, 0)
# X_valid = sg_filter(X_valid, 9, 3, 0)
# X_test = sg_filter(X_test, 9, 3, 0)
#
# # 归一化
# scaler = MinMaxScaler()
# X_train = scaler.fit_transform(X_train)
# X_valid = scaler.transform(X_valid)
# X_test = scaler.transform(X_test)
#
# # 使用支持向量机回归
# model = SVR(kernel='rbf', C=100, epsilon=0.001)
#
# #训练模型
# model.fit(X_train, y_train)
#
# #验证集结果
# y_valid_pred = model.predict(X_valid)
# valid_mse = mean_squared_error(y_valid, y_valid_pred)
# valid_r2 = r2_score(y_valid, y_valid_pred)
#
# print("在验证集性能：")
# print(f'valid_mse: {valid_mse:.4f}')
# print(f'valid_r2: {valid_r2:.4f}')
#
# #测试集结果
# y_test_pred= model.predict(X_test)
# test_mse = mean_squared_error(y_test,y_test_pred)
# test_r2 = r2_score(y_test,y_test_pred)
#
# print("在测试集性能：")
# print(f'test_mse: {test_mse:.4f}')
# print(f'test_r2: {test_r2:.4f}')
#
# # 创建一个 DataFrame，将 y_test_pred 写入第一列
# df = pd.DataFrame({'y_test_pred_1': y_test_pred})
# # 保存为 CSV 文件
# output_file = './results_1/y_test_pred_1.csv'
# df.to_csv(output_file, index=False)
#
# plt.figure(figsize=(8,6))
# plt.scatter(y_test, y_test_pred, color='blue', label='Predicted vs True')
# plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', label='Perfect Prediction Line')
# plt.xlabel('True Concentration')
# plt.ylabel('Predicted Concentration')
# plt.title('True vs Predicted Concentration')
# plt.legend()
# plt.show()

# fusion 结果
file_path = './results_1/y_test_pred_12345_fusion.csv'  # 替换为你的文件路径
df = pd.read_csv(file_path)

# 按行计算平均值
y_means_fusion = df.mean(axis=1)
print(y_means_fusion.shape)


# 创建一个新的 DataFrame，只包含 y_means 这一列
output_df = pd.DataFrame({'y_means_fusion': y_means_fusion})
# 保存为 CSV 文件
output_file = './results_1/y_means_fusion.csv'
output_df.to_csv(output_file, index=False)

fusion_mse = mean_squared_error(y_test,y_means_fusion)
fusion_r2 = r2_score(y_test,y_means_fusion)
print("测试集：")
print(f'fusion_mse: {fusion_mse:.4f}')
print(f'fusion_r2: {fusion_r2:.4f}')

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_means_fusion, color='blue', label='Predicted vs True')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', label='Perfect Prediction Line')
plt.xlabel('True Concentration')
plt.ylabel('Predicted Concentration')
plt.title('True vs Predicted Concentration')
plt.legend()
plt.show()