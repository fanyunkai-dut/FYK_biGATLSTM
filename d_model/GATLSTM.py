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
import pickle


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用的设备: {device}")

# ==================== 从 configs.yaml 读取配置 ====================
CONFIG_PATH = "/home/fanyunkai/FYK_biGATLSTM/configs.yaml"

with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config_all = yaml.safe_load(f)

cfg = config_all.get('GATLSTM_training', {})

SEQ_LEN = cfg.get('SEQ_LEN')
BATCH_SIZE = cfg.get('BATCH_SIZE')
GAT_HIDDEN = cfg.get('GCN_HIDDEN')
LSTM_HIDDEN = cfg.get('LSTM_HIDDEN')
OUTPUT_DIM = cfg.get('OUTPUT_DIM')
LEARNING_RATE = cfg.get('LEARNING_RATE')
PATIENCE = cfg.get('PATIENCE')
MAX_EPOCHS = cfg.get('MAX_EPOCHS')
TRAIN_RATIO = cfg.get('TRAIN_RATIO')
VAL_RATIO = cfg.get('VAL_RATIO')
TEST_RATIO = cfg.get('TEST_RATIO')
SEED = cfg.get('SEED')
NUM_HEADS = cfg.get('NUM_HEADS')

DATA_PATH = cfg.get('DATA_PATH')
FEATURE_NAMES_PATH = cfg.get('FEATURE_NAMES_PATH')
ADJ_PATH = cfg.get('ADJ_PATH')
OUTPUT_DIR = cfg.get('OUTPUT_DIR')

os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_NAMES = cfg.get('target_names', [])
INPUT_FEATURE_NAMES = cfg.get('input_feature_names', cfg.get('feature_names', []))
STANDARDIZE_FEATURE_NAMES = cfg.get('standardize_feature_names', None)

MASK_SUFFIX = cfg.get('MASK_SUFFIX', '_mask')

# dt 特征名后缀，仅用于你后面识别配置方便，不参与特殊逻辑
DT_SUFFIX = cfg.get('DT_SUFFIX', '_dt')


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)

# ==================== 数据加载 ====================
print("加载数据...")
full_data = np.load(DATA_PATH).astype(np.float32)
with open(FEATURE_NAMES_PATH, 'rb') as f:
    all_feature_names = pickle.load(f)

T, N, F_full = full_data.shape
print(f"完整数据形状: {full_data.shape}, 站点数: {N}, 特征数: {F_full}")
print(f"完整特征名: {all_feature_names}")

if not TARGET_NAMES:
    raise ValueError("配置文件中 target_names 不能为空")
if not INPUT_FEATURE_NAMES:
    raise ValueError("配置文件中 input_feature_names 不能为空")

missing_input = [name for name in INPUT_FEATURE_NAMES if name not in all_feature_names]
if missing_input:
    raise ValueError(f"以下 input_feature_names 不存在于预处理特征中: {missing_input}")

missing_target = [name for name in TARGET_NAMES if name not in all_feature_names]
if missing_target:
    raise ValueError(f"以下 target_names 不存在于预处理特征中: {missing_target}")

INPUT_IDXS = [all_feature_names.index(name) for name in INPUT_FEATURE_NAMES]
TARGET_IDXS = [all_feature_names.index(name) for name in TARGET_NAMES]

MASK_IDXS = []
for name in TARGET_NAMES:
    mask_name = name + MASK_SUFFIX
    if mask_name not in all_feature_names:
        raise ValueError(f"找不到目标 '{name}' 对应的掩码列 '{mask_name}'")
    MASK_IDXS.append(all_feature_names.index(mask_name))

print(f"本次输入特征 ({len(INPUT_FEATURE_NAMES)} 个): {INPUT_FEATURE_NAMES}")
print(f"本次目标特征 ({len(TARGET_NAMES)} 个): {TARGET_NAMES}")

x_full = full_data[:, :, INPUT_IDXS]     # (T, N, F_in)，允许有 NaN
y_full = full_data[:, :, TARGET_IDXS]    # (T, N, F_out)，允许有 NaN
mask_full = full_data[:, :, MASK_IDXS]   # (T, N, F_out)

F_in = x_full.shape[-1]
F_out = y_full.shape[-1]
print(f"X 形状: {x_full.shape}, y 形状: {y_full.shape}, mask 形状: {mask_full.shape}")

# ==================== 划分原始数据（未标准化） ====================
train_len = int(T * TRAIN_RATIO)
val_len = int(T * VAL_RATIO)
test_len = T - train_len - val_len

val_start = train_len + SEQ_LEN
test_start = val_start + val_len + SEQ_LEN

if test_start + SEQ_LEN > T:
    print("警告：测试集数据不足，自动调整比例。")
    test_start = max(test_start, train_len + SEQ_LEN + 1)
    test_len = T - test_start

train_x_raw = x_full[:train_len]
val_x_raw = x_full[val_start:val_start + val_len]
test_x_raw = x_full[test_start:test_start + test_len]

train_y_raw = y_full[:train_len]
val_y_raw = y_full[val_start:val_start + val_len]
test_y_raw = y_full[test_start:test_start + test_len]

train_mask_raw = mask_full[:train_len]
val_mask_raw = mask_full[val_start:val_start + val_len]
test_mask_raw = mask_full[test_start:test_start + test_len]

print(f"训练集原始时间步: {train_x_raw.shape[0]}, 验证集: {val_x_raw.shape[0]}, 测试集: {test_x_raw.shape[0]}")

# ==================== 确定需要标准化的 X 特征列 ====================
if STANDARDIZE_FEATURE_NAMES is None:
    norm_indices = []
    standardize_names_used = []
    for i, name in enumerate(INPUT_FEATURE_NAMES):
        if name.endswith(MASK_SUFFIX) or ('sin' in name) or ('cos' in name):
            continue
        norm_indices.append(i)
        standardize_names_used.append(name)
else:
    invalid_norm_names = [name for name in STANDARDIZE_FEATURE_NAMES if name not in INPUT_FEATURE_NAMES]
    if invalid_norm_names:
        print(f"警告：以下 standardize_feature_names 不在 input_feature_names 中，已忽略: {invalid_norm_names}")

    standardize_names_used = []
    norm_indices = []
    for name in STANDARDIZE_FEATURE_NAMES:
        if name not in INPUT_FEATURE_NAMES:
            continue
        if name.endswith(MASK_SUFFIX):
            print(f"警告：输入特征 {name} 是 mask 列，不应对 X 做标准化，已忽略")
            continue
        idx = INPUT_FEATURE_NAMES.index(name)
        norm_indices.append(idx)
        standardize_names_used.append(name)

print(f"X 中需要标准化的特征 ({len(standardize_names_used)} 个): {standardize_names_used}")

# ==================== 计算训练集 X 的统计量（忽略 NaN） ====================
def fit_x_scaler_nanaware(x_data, norm_indices, input_feature_names):
    means = []
    stds = []
    used_names = []

    for idx in norm_indices:
        vals = x_data[..., idx].reshape(-1)
        valid = ~np.isnan(vals)

        if np.sum(valid) == 0:
            print(f"警告：输入特征 {input_feature_names[idx]} 在训练集中全是 NaN，标准化参数将设为 mean=0,std=1")
            mean = 0.0
            std = 1.0
        else:
            mean = float(np.mean(vals[valid]))
            std = float(np.std(vals[valid]))
            if std == 0:
                std = 1.0

        means.append(mean)
        stds.append(std)
        used_names.append(input_feature_names[idx])

    return np.array(means, dtype=np.float32), np.array(stds, dtype=np.float32), used_names


x_means, x_stds, standardize_names_used = fit_x_scaler_nanaware(
    train_x_raw, norm_indices, INPUT_FEATURE_NAMES
)

# ==================== 计算训练集 y 的统计量（只用 mask=1 的真实观测） ====================
def fit_target_scaler(y_data, mask_data, target_names):
    y_means = np.zeros(len(target_names), dtype=np.float32)
    y_stds = np.ones(len(target_names), dtype=np.float32)

    for i, name in enumerate(target_names):
        valid_values = y_data[:, :, i][mask_data[:, :, i] > 0.5]
        valid_values = valid_values[~np.isnan(valid_values)]

        if valid_values.size == 0:
            raise ValueError(f"目标 {name} 在训练集中没有任何有效观测，无法标准化")

        y_means[i] = np.mean(valid_values)
        y_stds[i] = np.std(valid_values)
        if y_stds[i] == 0:
            y_stds[i] = 1.0

    return y_means, y_stds


y_means, y_stds = fit_target_scaler(train_y_raw, train_mask_raw, TARGET_NAMES)
print(f"y 将按 target_names 自动标准化: {TARGET_NAMES}")

# ==================== 标准化 X 和 y（保留 NaN，后面再统一填0） ====================
def apply_standardization_nanaware(data, norm_indices, means, stds):
    """
    对数据中指定索引的特征进行标准化，只作用于非 NaN 位置。
    """
    data_norm = data.copy()
    for idx, mean, std in zip(norm_indices, means, stds):
        valid = ~np.isnan(data_norm[..., idx])
        data_norm[..., idx][valid] = (data_norm[..., idx][valid] - mean) / std
    return data_norm


def apply_target_standardization_nanaware(y_data, mask_data, means, stds):
    """
    对目标变量做标准化，只作用于 mask=1 且非 NaN 的位置。
    其余位置保持 NaN，后面统一填0。
    """
    y_norm = y_data.copy()
    for i, (mean, std) in enumerate(zip(means, stds)):
        valid = (mask_data[..., i] > 0.5) & (~np.isnan(y_norm[..., i]))
        y_norm[..., i][valid] = (y_norm[..., i][valid] - mean) / std
    return y_norm


train_x = apply_standardization_nanaware(train_x_raw, norm_indices, x_means, x_stds)
val_x   = apply_standardization_nanaware(val_x_raw,   norm_indices, x_means, x_stds)
test_x  = apply_standardization_nanaware(test_x_raw,  norm_indices, x_means, x_stds)

train_y = apply_target_standardization_nanaware(train_y_raw, train_mask_raw, y_means, y_stds)
val_y   = apply_target_standardization_nanaware(val_y_raw,   val_mask_raw,   y_means, y_stds)
test_y  = apply_target_standardization_nanaware(test_y_raw,  test_mask_raw,  y_means, y_stds)

print("X 与 y 标准化完成（目前仍保留 NaN）。")

# ==================== 标准化后统一把 NaN 换成 0 ====================
def replace_nan_with_zero(arr):
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

train_x = replace_nan_with_zero(train_x)
val_x   = replace_nan_with_zero(val_x)
test_x  = replace_nan_with_zero(test_x)

train_y = replace_nan_with_zero(train_y)
val_y   = replace_nan_with_zero(val_y)
test_y  = replace_nan_with_zero(test_y)

print("已将标准化后的 X / y 中所有 NaN 统一替换为 0。")

# ==================== 保存标准化参数 ====================
scaler_params = {
    'x_means': x_means,
    'x_stds': x_stds,
    'x_norm_indices': norm_indices,
    'input_feature_names': INPUT_FEATURE_NAMES,
    'x_standardize_feature_names': standardize_names_used,
    'y_means': y_means,
    'y_stds': y_stds,
    'target_names': TARGET_NAMES,
    'mask_feature_names': [name + MASK_SUFFIX for name in TARGET_NAMES],
    'all_feature_names': all_feature_names,
}
with open(os.path.join(OUTPUT_DIR, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler_params, f)
print("标准化参数已保存到 scaler.pkl")

# ==================== 为每个数据集构建滑动窗口样本 ====================
def build_samples(x_data, y_data, mask_data, seq_len):
    """
    为给定数据集构建样本，每个样本使用 seq_len 个历史时间步。
    """
    T_local = x_data.shape[0]
    X_list, y_list, mask_list = [], [], []

    for t in range(seq_len, T_local):
        X_window = x_data[t - seq_len:t]
        y_t = y_data[t]
        mask_t = mask_data[t]

        X_list.append(X_window)
        y_list.append(y_t)
        mask_list.append(mask_t)

    return np.stack(X_list), np.stack(y_list), np.stack(mask_list)


print("构建训练集样本...")
train_X, train_y_samples, train_mask = build_samples(train_x, train_y, train_mask_raw, SEQ_LEN)
print("构建验证集样本...")
val_X, val_y_samples, val_mask = build_samples(val_x, val_y, val_mask_raw, SEQ_LEN)
print("构建测试集样本...")
test_X, test_y_samples, test_mask = build_samples(test_x, test_y, test_mask_raw, SEQ_LEN)

print(f"训练样本数: {train_X.shape[0]}, 验证样本数: {val_X.shape[0]}, 测试样本数: {test_X.shape[0]}")
print(f"X 形状: {train_X.shape}, y 形状: {train_y_samples.shape}, mask 形状: {train_mask.shape}")

# ==================== 创建 DataLoader ====================
train_dataset = TensorDataset(
    torch.tensor(train_X, dtype=torch.float32),
    torch.tensor(train_y_samples, dtype=torch.float32),
    torch.tensor(train_mask, dtype=torch.float32)
)
val_dataset = TensorDataset(
    torch.tensor(val_X, dtype=torch.float32),
    torch.tensor(val_y_samples, dtype=torch.float32),
    torch.tensor(val_mask, dtype=torch.float32)
)
test_dataset = TensorDataset(
    torch.tensor(test_X, dtype=torch.float32),
    torch.tensor(test_y_samples, dtype=torch.float32),
    torch.tensor(test_mask, dtype=torch.float32)
)

pin_memory = torch.cuda.is_available()

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=pin_memory)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin_memory)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin_memory)

# ==================== 邻接矩阵加载与处理 ====================
file_ext = os.path.splitext(ADJ_PATH)[1].lower()
if file_ext == '.npy':
    adj = np.load(ADJ_PATH)
elif file_ext == '.csv':
    adj = np.loadtxt(ADJ_PATH, delimiter=',')
    if adj.shape[0] == N and adj.shape[1] == N + 1:
        adj = adj[:, 1:]
        print("检测到邻接矩阵包含行号列，已自动去除第一列。")
    elif adj.shape[0] == N + 1 and adj.shape[1] == N + 1:
        adj = adj[1:, 1:]
        print("检测到邻接矩阵包含行号和列名，已自动去除第一行和第一列。")
    elif adj.shape != (N, N):
        raise ValueError(f"邻接矩阵形状应为 ({N},{N})，但加载后为 {adj.shape}")
else:
    raise ValueError(f"不支持的文件格式: {file_ext}，请使用 .npy 或 .csv")

adj_binary = (adj != 0).astype(np.float32)
np.fill_diagonal(adj_binary, 1.0)

adj_tensor = torch.tensor(adj_binary, dtype=torch.float32)
print(f"邻接矩阵已转换为二值矩阵，非零边数: {np.sum(adj_binary)}")

# ==================== 定义模型 ====================
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, adj, num_heads=1, concat=True, dropout=0.2, alpha=0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat

        self.register_buffer("adj", adj)

        self.W = nn.ModuleList([
            nn.Linear(in_features, out_features, bias=False) for _ in range(num_heads)
        ])
        self.a = nn.ParameterList([
            nn.Parameter(torch.empty(2 * out_features, 1)) for _ in range(num_heads)
        ])
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

        for w in self.W:
            nn.init.xavier_uniform_(w.weight)
        for a in self.a:
            nn.init.xavier_uniform_(a)

    def forward(self, x):
        batch, N_local, _ = x.size()
        adj_mask = self.adj.unsqueeze(0).expand(batch, -1, -1)

        head_outputs = []
        for h in range(self.num_heads):
            Wh = self.W[h](x)
            a = self.a[h]

            Wh_i = Wh.unsqueeze(2).expand(-1, -1, N_local, -1)
            Wh_j = Wh.unsqueeze(1).expand(-1, N_local, -1, -1)
            a_input = torch.cat([Wh_i, Wh_j], dim=-1)

            e = self.leakyrelu(torch.matmul(a_input, a).squeeze(-1))
            e = e.masked_fill(adj_mask == 0, float('-inf'))
            attention = torch.softmax(e, dim=-1)
            attention = self.dropout(attention)

            h_prime = torch.matmul(attention, Wh)
            head_outputs.append(h_prime)

        if self.concat:
            output = torch.cat(head_outputs, dim=-1)
        else:
            output = torch.mean(torch.stack(head_outputs, dim=0), dim=0)
        return output


class GAT_LSTM(nn.Module):
    def __init__(self, node_num, input_dim, gat_dim, lstm_hidden, output_dim, adj, num_heads=1):
        super().__init__()
        self.gat = GraphAttentionLayer(
            input_dim, gat_dim, adj, num_heads=num_heads, concat=True
        )
        lstm_input_dim = gat_dim * num_heads
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_hidden,
            batch_first=True
        )
        self.fc = nn.Linear(lstm_hidden, output_dim)

    def forward(self, x):
        batch, seq_len, node_num, _ = x.shape

        x_gat = []
        for t in range(seq_len):
            xt = x[:, t, :, :]
            ht = self.gat(xt)
            x_gat.append(ht)

        x_gat = torch.stack(x_gat, dim=1)
        x_gat = x_gat.permute(0, 2, 1, 3).contiguous()
        x_gat = x_gat.view(batch * node_num, seq_len, -1)

        lstm_out, _ = self.lstm(x_gat)
        last_out = lstm_out[:, -1, :]
        pred = self.fc(last_out)
        pred = pred.view(batch, node_num, -1)

        return pred


if OUTPUT_DIM is None or OUTPUT_DIM != F_out:
    OUTPUT_DIM = F_out
    print(f"OUTPUT_DIM 已自动设置为目标维度: {OUTPUT_DIM}")

model = GAT_LSTM(
    node_num=N,
    input_dim=F_in,
    gat_dim=GAT_HIDDEN,
    lstm_hidden=LSTM_HIDDEN,
    output_dim=OUTPUT_DIM,
    adj=adj_tensor,
    num_heads=NUM_HEADS
).to(device)

# ==================== 定义损失函数和优化器 ====================
def masked_mse_loss(pred, target, mask):
    loss = (pred - target) ** 2
    loss = loss * mask
    return loss.sum() / (mask.sum() + 1e-8)


optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=5, factor=0.5
)

best_val_loss = float('inf')
counter = 0

# ==================== 训练循环 ====================
print("开始训练...")
for epoch in range(MAX_EPOCHS):
    model.train()
    train_loss = 0.0

    for batch_X, batch_y, batch_mask in train_loader:
        batch_X = batch_X.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        batch_mask = batch_mask.to(device, non_blocking=True)

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
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            batch_mask = batch_mask.to(device, non_blocking=True)

            pred = model(batch_X)
            loss = masked_mse_loss(pred, batch_y, batch_mask)
            val_loss += loss.item()

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    scheduler.step(val_loss)

    print(f"Epoch {epoch + 1}/{MAX_EPOCHS} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model.pth'))
        counter = 0
    else:
        counter += 1
        if counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch + 1}")
            break

# ==================== 测试评估 ====================
print("加载最佳模型进行测试...")
model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_model.pth'), map_location=device))
model.eval()

test_loss = 0.0
all_preds = []
all_targets = []
all_masks = []

with torch.no_grad():
    for batch_X, batch_y, batch_mask in test_loader:
        batch_X = batch_X.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        batch_mask = batch_mask.to(device, non_blocking=True)

        pred = model(batch_X)
        loss = masked_mse_loss(pred, batch_y, batch_mask)
        test_loss += loss.item()

        all_preds.append(pred.cpu().numpy())
        all_targets.append(batch_y.cpu().numpy())
        all_masks.append(batch_mask.cpu().numpy())

test_loss /= len(test_loader)
print(f"测试集损失: {test_loss:.6f}")

pred_concat = np.concatenate(all_preds, axis=0)
target_concat = np.concatenate(all_targets, axis=0)
mask_concat = np.concatenate(all_masks, axis=0)

np.save(os.path.join(OUTPUT_DIR, 'test_pred.npy'), pred_concat)
np.save(os.path.join(OUTPUT_DIR, 'test_target.npy'), target_concat)
np.save(os.path.join(OUTPUT_DIR, 'test_mask.npy'), mask_concat)
print(f"测试结果已保存到 {OUTPUT_DIR}")

# ==================== 保存训练集和验证集预测结果 ====================
def save_predictions(loader, name):
    all_preds = []
    all_targets = []
    all_masks = []
    model.eval()
    with torch.no_grad():
        for batch_X, batch_y, batch_mask in loader:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            batch_mask = batch_mask.to(device, non_blocking=True)

            pred = model(batch_X)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
            all_masks.append(batch_mask.cpu().numpy())

    pred_concat = np.concatenate(all_preds, axis=0)
    target_concat = np.concatenate(all_targets, axis=0)
    mask_concat = np.concatenate(all_masks, axis=0)

    np.save(os.path.join(OUTPUT_DIR, f'{name}_pred.npy'), pred_concat)
    np.save(os.path.join(OUTPUT_DIR, f'{name}_target.npy'), target_concat)
    np.save(os.path.join(OUTPUT_DIR, f'{name}_mask.npy'), mask_concat)
    print(f"{name} 集预测结果已保存到 {OUTPUT_DIR}")


save_predictions(train_loader, 'train')
save_predictions(val_loader, 'val')

# ==================== 计算整体指标 ====================
valid_idx = mask_concat > 0.5
pred_valid = pred_concat[valid_idx]
target_valid = target_concat[valid_idx]

r2 = r2_score(target_valid, pred_valid)
rmse = np.sqrt(mean_squared_error(target_valid, pred_valid))
mae = mean_absolute_error(target_valid, pred_valid)

print("====== 测试集性能（GAT-LSTM）======")
print(f"R2: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")

with open(os.path.join(OUTPUT_DIR, 'Configs and results.txt'), 'w', encoding='utf-8') as f:
    f.write("========== Configuration ==========\n")
    f.write(f"SEQ_LEN: {SEQ_LEN}\n")
    f.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
    f.write(f"GAT_HIDDEN: {GAT_HIDDEN}\n")
    f.write(f"LSTM_HIDDEN: {LSTM_HIDDEN}\n")
    f.write(f"OUTPUT_DIM: {OUTPUT_DIM}\n")
    f.write(f"LEARNING_RATE: {LEARNING_RATE}\n")
    f.write(f"PATIENCE: {PATIENCE}\n")
    f.write(f"MAX_EPOCHS: {MAX_EPOCHS}\n")
    f.write(f"TRAIN_RATIO: {TRAIN_RATIO}\n")
    f.write(f"VAL_RATIO: {VAL_RATIO}\n")
    f.write(f"TEST_RATIO: {TEST_RATIO}\n")
    f.write(f"SEED: {SEED}\n")
    f.write(f"NUM_HEADS: {NUM_HEADS}\n")
    f.write(f"DATA_PATH: {DATA_PATH}\n")
    f.write(f"FEATURE_NAMES_PATH: {FEATURE_NAMES_PATH}\n")
    f.write(f"ADJ_PATH: {ADJ_PATH}\n")
    f.write(f"OUTPUT_DIR: {OUTPUT_DIR}\n")
    f.write(f"INPUT_FEATURE_NAMES: {INPUT_FEATURE_NAMES}\n")
    f.write(f"STANDARDIZE_FEATURE_NAMES: {standardize_names_used}\n")
    f.write(f"TARGET_NAMES: {TARGET_NAMES}\n")
    f.write("===================================\n\n")
    f.write("========== Overall Metrics ==========\n")
    f.write(f"R2: {r2:.4f}\n")
    f.write(f"RMSE: {rmse:.4f}\n")
    f.write(f"MAE: {mae:.4f}\n")
    f.write("=====================================\n")

plt.figure(figsize=(6, 6))
plt.scatter(target_valid, pred_valid, alpha=0.3, s=1)
plt.plot(
    [target_valid.min(), target_valid.max()],
    [target_valid.min(), target_valid.max()],
    'r--', lw=2
)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title(f'Overall (R2={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f})')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'scatter_overall.png'), dpi=150)
plt.close()

print(f"散点图已保存到 {OUTPUT_DIR}")
