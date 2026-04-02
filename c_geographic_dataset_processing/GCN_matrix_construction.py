import pandas as pd
import numpy as np
import math
import yaml

# ==================== 从 configs.yaml 读取配置 ====================
CONFIG_PATH = "configs.yaml"  # 可根据需要修改为绝对路径

with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config_all = yaml.safe_load(f)

# 假设配置存放在 'geo_matrix_construction' 键下
cfg = config_all.get('geo_matrix_construction', {})

# 读取配置项（若缺失则使用默认值）
input_csv = cfg.get('input_path')
output_npy = cfg.get('output_path')
sigma_config = cfg.get('sigma', None)          # 如果指定则使用该数值，否则自动计算
earth_radius = cfg.get('earth_radius', 6371.0) # 地球半径，一般不需要修改
# ================================================================

# -------------------- 读取数据 --------------------
df = pd.read_csv(input_csv)
lons = df['lon'].values
lats = df['lat'].values
n = len(df)

# -------------------- 计算距离矩阵（haversine公式，返回千米） --------------------
def haversine(lon1, lat1, lon2, lat2, R=earth_radius):
    """
    计算两点间的大圆距离（单位：千米）
    """
    dlon = math.radians(lon2 - lon1)
    dlat = math.radians(lat2 - lat1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

dist_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i != j:
            dist_matrix[i, j] = haversine(lons[i], lats[i], lons[j], lats[j])

# -------------------- 确定sigma（带宽） --------------------
non_zero_dists = dist_matrix[dist_matrix > 0]
if sigma_config is None:
    sigma = np.median(non_zero_dists)  # 默认使用中位数
    print(f"自动计算 sigma = {sigma:.4f} km")
else:
    sigma = sigma_config
    print(f"使用配置的 sigma = {sigma} km")

# -------------------- 高斯核权重 --------------------
A = np.exp(- (dist_matrix ** 2) / (2 * sigma ** 2))
# 确保自环为1（对角线）
np.fill_diagonal(A, 1)

# -------------------- 对称归一化 --------------------
D = np.diag(np.sum(A, axis=1))
D_inv_sqrt = np.linalg.inv(np.sqrt(D))
A_norm = D_inv_sqrt @ A @ D_inv_sqrt

# -------------------- 查看结果并保存 --------------------
print("归一化后的邻接矩阵（前7行）：")
print(A_norm)

np.save(output_npy, A_norm)
print(f"邻接矩阵已保存至 {output_npy}")