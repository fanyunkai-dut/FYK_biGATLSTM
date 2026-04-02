import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os
import random
import yaml

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用的设备: {device}")

# ==================== 从 configs.yaml 读取配置 ====================
CONFIG_PATH = "configs.yaml"  # 可根据需要改为绝对路径

with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config_all = yaml.safe_load(f)

# 假设训练相关的配置放在 'GCNLSTM_training' 键下
cfg = config_all.get('GCNLSTM_training', {})

# 超参数（若缺失则使用默认值）
SEQ_LEN = cfg.get('SEQ_LEN', 18)
BATCH_SIZE = cfg.get('BATCH_SIZE', 32)
GCN_HIDDEN = cfg.get('GCN_HIDDEN', 16)
LSTM_HIDDEN = cfg.get('LSTM_HIDDEN', 32)
OUTPUT_DIM = cfg.get('OUTPUT_DIM', 2)
LEARNING_RATE = cfg.get('LEARNING_RATE', 0.0005)
PATIENCE = cfg.get('PATIENCE', 5)
MAX_EPOCHS = cfg.get('MAX_EPOCHS', 300)
TRAIN_RATIO = cfg.get('TRAIN_RATIO', 0.7)
VAL_RATIO = cfg.get('VAL_RATIO', 0.15)
TEST_RATIO = cfg.get('TEST_RATIO', 0.15)
SEED = cfg.get('SEED', 42)

# 文件路径
DATA_PATH = cfg.get('DATA_PATH', '/home/fanyunkai/FYK_GCNLSTM/xiangjiang/preprocessed_data.npy')
ADJ_PATH = cfg.get('ADJ_PATH', '/home/fanyunkai/FYK_GCNLSTM/xiangjiang/GEO/adjacency_matrix.npy')
OUTPUT_DIR = cfg.get('OUTPUT_DIR', '/home/fanyunkai/FYK_GCNLSTM/xiangjiang/results')

# 创建输出目录（如果不存在）
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 特征列名及索引（根据你的列表，保持不变）

feature_names = cfg.get('feature_names')
TARGET_NAMES = cfg.get('target_names')

# 计算目标变量索引和掩码索引
TARGET_IDXS = [feature_names.index(name) for name in TARGET_NAMES]
# 假设掩码列名 = 目标名 + '_mask'，并验证这些掩码列是否存在
MASK_IDXS = []
for name in TARGET_NAMES:
    mask_name = name + '_mask'
    if mask_name not in feature_names:
        raise ValueError(f"配置错误：找不到目标 '{name}' 对应的掩码列 '{mask_name}'")
    MASK_IDXS.append(feature_names.index(mask_name))

# 输入维度
INPUT_DIM = len(feature_names)
# ================================================================================================

def set_seed(seed=42):
    """设置所有随机种子"""
    random.seed(seed)                     # Python 随机库
    np.random.seed(seed)                  # NumPy
    torch.manual_seed(seed)               # PyTorch CPU
    torch.cuda.manual_seed(seed)          # PyTorch GPU（如果使用单张显卡）
    torch.cuda.manual_seed_all(seed)       # 如果使用多张显卡
    # 关闭 cuDNN 的自动调优，以保证确定性（但会降低性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(SEED)

# ==================== 数据加载 ====================
print("加载数据...")
data = np.load(DATA_PATH)                 # (T, N, 16)
adj = np.load(ADJ_PATH)                   # (N, N) 已归一化
T, N, F = data.shape
print(f"数据形状: {data.shape}, 站点数: {N}, 特征数: {F}")

# ==================== 构建滑动窗口样本 ====================
print("构建滑动窗口样本...")
samples_X = []
samples_y = []
samples_mask = []

for t in range(SEQ_LEN, T):
    X_window = data[t-SEQ_LEN:t]           # (SEQ_LEN, N, 16)
    samples_X.append(X_window)
    y_t = data[t][:, TARGET_IDXS]           # (N, 2)
    samples_y.append(y_t)
    mask_t = data[t][:, MASK_IDXS]         # (N, 2)
    samples_mask.append(mask_t)

X = np.stack(samples_X, axis=0)             # (样本数, SEQ_LEN, N, 16)
y = np.stack(samples_y, axis=0)             # (样本数, N, 2)
mask = np.stack(samples_mask, axis=0)       # (样本数, N, 2)

print(f"样本总数: {X.shape[0]}")
print(f"X shape: {X.shape}, y shape: {y.shape}, mask shape: {mask.shape}")

# ==================== 数据划分 ====================
num_samples = X.shape[0]
train_end = int(num_samples * TRAIN_RATIO)
val_end = int(num_samples * (TRAIN_RATIO + VAL_RATIO))

train_X, train_y, train_mask = X[:train_end], y[:train_end], mask[:train_end]
val_X, val_y, val_mask = X[train_end:val_end], y[train_end:val_end], mask[train_end:val_end]
test_X, test_y, test_mask = X[val_end:], y[val_end:], mask[val_end:]

print(f"训练集样本数: {train_X.shape[0]}, 验证集: {val_X.shape[0]}, 测试集: {test_X.shape[0]}")

# ==================== 创建DataLoader ====================
train_dataset = TensorDataset(torch.FloatTensor(train_X), 
                              torch.FloatTensor(train_y), 
                              torch.FloatTensor(train_mask))
val_dataset = TensorDataset(torch.FloatTensor(val_X), 
                            torch.FloatTensor(val_y), 
                            torch.FloatTensor(val_mask))
test_dataset = TensorDataset(torch.FloatTensor(test_X), 
                             torch.FloatTensor(test_y), 
                             torch.FloatTensor(test_mask))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==================== 邻接矩阵转张量 ====================
adj_tensor = torch.FloatTensor(adj)   # 直接使用已归一化的邻接矩阵

# ==================== 定义GCN-LSTM模型 ====================
class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features, adj):
        super().__init__()
        self.adj = adj  # 归一化邻接矩阵，形状 (N, N)
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        # x: (batch, N, in_features)
        x = torch.matmul(self.adj.to(x.device), x)  # (batch, N, in_features)
        x = self.fc(x)                               # (batch, N, out_features)
        return x

class GCN_LSTM(nn.Module):
    def __init__(self, node_num, input_dim, gcn_dim, lstm_hidden, output_dim, adj):
        super().__init__()
        self.gcn = GraphConvLayer(input_dim, gcn_dim, adj)
        self.lstm = nn.LSTM(input_size=gcn_dim, hidden_size=lstm_hidden, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, output_dim)

    def forward(self, x):
        batch, seq_len, node_num, _ = x.shape
        # 对每个时间步应用GCN
        x_gcn = []
        for t in range(seq_len):
            xt = x[:, t, :, :]                     # (batch, node_num, input_dim)
            ht = self.gcn(xt)                       # (batch, node_num, gcn_dim)
            x_gcn.append(ht)
        x_gcn = torch.stack(x_gcn, dim=1)           # (batch, seq_len, node_num, gcn_dim)
        # 合并节点到batch维度
        x_gcn = x_gcn.permute(0,2,1,3).contiguous() # (batch, node_num, seq_len, gcn_dim)
        x_gcn = x_gcn.view(batch * node_num, seq_len, -1)  # (batch*node_num, seq_len, gcn_dim)
        lstm_out, (h_n, c_n) = self.lstm(x_gcn)    # lstm_out: (batch*node_num, seq_len, lstm_hidden)
        last_out = lstm_out[:, -1, :]               # (batch*node_num, lstm_hidden)
        pred = self.fc(last_out)                    # (batch*node_num, output_dim)
        pred = pred.view(batch, node_num, -1)       # (batch, node_num, output_dim)
        return pred

# 初始化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN_LSTM(node_num=N, 
                 input_dim=INPUT_DIM,
                 gcn_dim=GCN_HIDDEN, 
                 lstm_hidden=LSTM_HIDDEN, 
                 output_dim=OUTPUT_DIM, 
                 adj=adj_tensor).to(device)

# ==================== 定义损失函数和优化器 ====================
def masked_mse_loss(pred, target, mask):
    loss = (pred - target) ** 2
    loss = loss * mask
    return loss.sum() / (mask.sum() + 1e-8)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

# ==================== 训练准备 ====================
best_val_loss = float('inf')
counter = 0

# ==================== 训练循环 ====================
print("开始训练...")
for epoch in range(MAX_EPOCHS):
    model.train()
    train_loss = 0.0
    for batch_X, batch_y, batch_mask in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        batch_mask = batch_mask.to(device)
        
        optimizer.zero_grad()
        pred = model(batch_X)
        loss = masked_mse_loss(pred, batch_y, batch_mask)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y, batch_mask in val_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            batch_mask = batch_mask.to(device)
            pred = model(batch_X)
            loss = masked_mse_loss(pred, batch_y, batch_mask)
            val_loss += loss.item()
    
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    scheduler.step(val_loss)
    
    print(f"Epoch {epoch+1}/{MAX_EPOCHS} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model.pth'))
        counter = 0
    else:
        counter += 1
        if counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

# ==================== 测试评估 ====================
print("加载最佳模型进行测试...")
model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_model.pth')))
model.eval()

test_loss = 0.0
all_preds = []
all_targets = []
all_masks = []

with torch.no_grad():
    for batch_X, batch_y, batch_mask in test_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        batch_mask = batch_mask.to(device)
        pred = model(batch_X)
        loss = masked_mse_loss(pred, batch_y, batch_mask)
        test_loss += loss.item()
        all_preds.append(pred.cpu().numpy())
        all_targets.append(batch_y.cpu().numpy())
        all_masks.append(batch_mask.cpu().numpy())

test_loss /= len(test_loader)
print(f"测试集损失: {test_loss:.6f}")

# 收集所有预测和真实值
pred_concat = np.concatenate(all_preds, axis=0)   # (样本数, N, 2)
target_concat = np.concatenate(all_targets, axis=0)
mask_concat = np.concatenate(all_masks, axis=0)

# 保存测试结果到文件，供分站点分析使用
np.save(os.path.join(OUTPUT_DIR, 'test_pred.npy'), pred_concat)
np.save(os.path.join(OUTPUT_DIR, 'test_target.npy'), target_concat)
np.save(os.path.join(OUTPUT_DIR, 'test_mask.npy'), mask_concat)
print(f"测试结果已保存到 {OUTPUT_DIR}")

# 计算整体指标（仅对mask=1的位置）
valid_idx = mask_concat > 0.5
pred_valid = pred_concat[valid_idx]
target_valid = target_concat[valid_idx]

r2 = r2_score(target_valid, pred_valid)
rmse = np.sqrt(mean_squared_error(target_valid, pred_valid))
mae = mean_absolute_error(target_valid, pred_valid)

print("====== 测试集性能（基线）======")
print(f"R2: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")

# 将整体指标写入文本文件
with open(os.path.join(OUTPUT_DIR, 'Configs and results.txt'), 'w') as f:
    # 写入配置信息
    f.write("========== Configuration ==========\n")
    f.write(f"SEQ_LEN: {SEQ_LEN}\n")
    f.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
    f.write(f"GCN_HIDDEN: {GCN_HIDDEN}\n")
    f.write(f"LSTM_HIDDEN: {LSTM_HIDDEN}\n")
    f.write(f"OUTPUT_DIM: {OUTPUT_DIM}\n")
    f.write(f"LEARNING_RATE: {LEARNING_RATE}\n")
    f.write(f"PATIENCE: {PATIENCE}\n")
    f.write(f"MAX_EPOCHS: {MAX_EPOCHS}\n")
    f.write(f"TRAIN_RATIO: {TRAIN_RATIO}\n")
    f.write(f"VAL_RATIO: {VAL_RATIO}\n")
    f.write(f"TEST_RATIO: {TEST_RATIO}\n")
    f.write(f"SEED: {SEED}\n")
    f.write(f"DATA_PATH: {DATA_PATH}\n")
    f.write(f"ADJ_PATH: {ADJ_PATH}\n")
    f.write(f"OUTPUT_DIR: {OUTPUT_DIR}\n")
    f.write("===================================\n\n")
    f.write("========== Overall Metrics ==========\n")
    f.write(f"R2: {r2:.4f}\n")
    f.write(f"RMSE: {rmse:.4f}\n")
    f.write(f"MAE: {mae:.4f}\n")
    f.write("=====================================\n")

# ==================== 绘制散点图 ====================
plt.figure(figsize=(6,6))
plt.scatter(target_valid, pred_valid, alpha=0.3, s=1)
plt.plot([target_valid.min(), target_valid.max()], 
         [target_valid.min(), target_valid.max()], 'r--', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title(f'Overall (R2={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f})')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'scatter_overall.png'), dpi=150)
plt.close()
print(f"散点图已保存到 {OUTPUT_DIR}")