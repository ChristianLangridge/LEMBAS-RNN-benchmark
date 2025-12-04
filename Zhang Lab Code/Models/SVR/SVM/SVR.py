import pandas as pd
import cupy as cp
from cuml.svm import SVR
from sklearn.model_selection import train_test_split
import numpy as np

# -------------------------------
# 1. 加载数据并转换为GPU格式
# -------------------------------
X_df = pd.read_csv('/proj/berzelius-2025-61/users/x_kejli/svm/TF.tsv',
                   sep='\t', index_col=0).astype(np.float32)
y_df = pd.read_csv('/proj/berzelius-2025-61/users/x_kejli/svm/Geneexpression.tsv',
                   sep='\t', index_col=0).astype(np.float32)

# 转成 cupy 数组
X = cp.asarray(X_df.values)
y = cp.asarray(y_df.values)

# -------------------------------
# 2. 用 numpy 索引在 GPU 上切分
# -------------------------------
# 先生成全样本的索引
indices = np.arange(X.shape[0])
# 在 CPU 上划分索引
train_idx, test_idx = train_test_split(
    indices, test_size=0.2, random_state=42)

# 用 numpy 索引直接在 cupy 上切
X_train = X[train_idx]
X_test  = X[test_idx]
y_train = y[train_idx]
y_test  = y[test_idx]

# 保留测试集的原始行索引
test_index = y_df.index[test_idx]

# -------------------------------
# 3. 手动实现多输出回归
# -------------------------------
print("开始训练多输出 SVR 模型...")
models = []
n_targets = y_train.shape[1]
for col in range(n_targets):
    print(f"  训练第 {col+1}/{n_targets} 个目标...")
    svr = SVR(kernel='rbf', C=1.0, gamma='scale')
    svr.fit(X_train, y_train[:, col])
    models.append(svr)
print("训练完成！")

# -------------------------------
# 4. 预测所有目标变量
# -------------------------------
y_pred = cp.zeros_like(y_test)
for i, m in enumerate(models):
    y_pred[:, i] = m.predict(X_test)

# -------------------------------
# 5. 保存预测结果（保留测试集索引）
# -------------------------------
y_pred_df = pd.DataFrame(
    y_pred.get(),
    columns=y_df.columns,
    index=test_index
)
y_pred_df.to_csv(
    '/proj/berzelius-2025-61/users/x_kejli/svm/Y_hat_SVR.tsv',
    sep='\t'
)

# -------------------------------
# 6. 评估指标计算（全部在 GPU 上）
# -------------------------------
def gpu_pearson(y_true, y_pred):
    mu_t = cp.mean(y_true, axis=0)
    mu_p = cp.mean(y_pred, axis=0)
    cov  = cp.mean((y_true - mu_t) * (y_pred - mu_p), axis=0)
    return cov / (cp.std(y_true, axis=0) * cp.std(y_pred, axis=0))

mse = cp.mean((y_test - y_pred) ** 2).get()
r2  = (1 - cp.sum((y_test - y_pred) ** 2) /
       cp.sum((y_test - cp.mean(y_test, axis=0)) ** 2)).get()
pearson_mean = cp.mean(gpu_pearson(y_test, y_pred)).get()

print(f"\n测试集 MSE: {mse:.6f}")
print(f"测试集 R²: {r2:.6f}")
print(f"平均 Pearson 相关系数: {pearson_mean:.6f}")
print("完成全部流程！")







