import os
import copy
import random
import pickle
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# =========================================================
# 0. 基础设置
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# =========================================================
# 1. 读取配置
#    这版优先兼容你当前给出的 GATLSTM_training 配置键名
# =========================================================
CONFIG_PATH = "/home/fanyunkai/FYK_GCNLSTM/configs.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config_all = yaml.safe_load(f)

train_cfg = config_all.get("GATLSTM_training", {})
data_cfg = config_all.get("data_preprocessing", {})

# ---------- 路径 ----------
DATA_PATH = train_cfg.get("DATA_PATH")
FEATURE_NAMES_PATH = train_cfg.get("FEATURE_NAMES_PATH")
ADJ_PATH = train_cfg.get("ADJ_PATH")
OUTPUT_DIR = train_cfg.get("OUTPUT_DIR")
ensure_dir(OUTPUT_DIR)

MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
RESULT_CSV_PATH = os.path.join(OUTPUT_DIR, "metrics_summary.csv")
PRED_SAVE_PATH = os.path.join(OUTPUT_DIR, "test_predictions.npz")
SCALER_SAVE_PATH = os.path.join(OUTPUT_DIR, "scalers.npz")
HISTORY_CSV_PATH = os.path.join(OUTPUT_DIR, "train_history.csv")

# ---------- 训练参数（兼容你原 config 键名） ----------
SEQ_LEN = train_cfg.get("SEQ_LEN", 18)
PRED_LEN = train_cfg.get("PRED_LEN", 1)
if PRED_LEN != 1:
    print(f"警告：当前代码已改为与第一段输出格式一致，只支持单步预测；检测到 PRED_LEN={PRED_LEN}，将按 1 处理。")
PRED_LEN = 1
BATCH_SIZE = train_cfg.get("BATCH_SIZE", 32)
GAT_HIDDEN_DIM = train_cfg.get("GAT_HIDDEN_DIM", train_cfg.get("GCN_HIDDEN", 32))
LSTM_HIDDEN_DIM = train_cfg.get("LSTM_HIDDEN_DIM", train_cfg.get("LSTM_HIDDEN", 64))
NUM_HEADS = train_cfg.get("NUM_HEADS", 4)
DROPOUT = train_cfg.get("DROPOUT", 0.2)
ALPHA = train_cfg.get("ALPHA", 0.2)
LR = train_cfg.get("LR", train_cfg.get("LEARNING_RATE", 5e-4))
WEIGHT_DECAY = train_cfg.get("WEIGHT_DECAY", 0.0)
PATIENCE = train_cfg.get("PATIENCE", 5)
EPOCHS = train_cfg.get("EPOCHS", train_cfg.get("MAX_EPOCHS", 300))
TRAIN_RATIO = train_cfg.get("TRAIN_RATIO", 0.7)
VAL_RATIO = train_cfg.get("VAL_RATIO", 0.15)
TEST_RATIO = train_cfg.get("TEST_RATIO", 0.15)
SEED = train_cfg.get("SEED", 42)
CONFIG_OUTPUT_DIM = train_cfg.get("OUTPUT_DIM", None)

# ---------- 特征信息 ----------
# 1) 优先读取 feature_names.pkl
# 2) 若无则退回到 config 里的 feature_names
feature_names_from_cfg = train_cfg.get("feature_names", [])
target_names = train_cfg.get("target_names", data_cfg.get("wq_names", []))

if DATA_PATH is None:
    raise ValueError("configs.yaml 中 GATLSTM_training.DATA_PATH 未设置。")

if FEATURE_NAMES_PATH and os.path.exists(FEATURE_NAMES_PATH):
    with open(FEATURE_NAMES_PATH, "rb") as f:
        feature_names = pickle.load(f)
    print(f"从 pickle 读取特征名: {FEATURE_NAMES_PATH}")
elif feature_names_from_cfg:
    feature_names = feature_names_from_cfg
    print("未找到 feature_names.pkl，改用 config 中的 feature_names。")
else:
    raise ValueError("既没有可用的 FEATURE_NAMES_PATH，也没有在 config 中提供 feature_names。")

# 固定时间特征名
TIME_FEATURE_NAMES = [
    "hour_sin", "hour_cos",
    "month_sin", "month_cos",
    "year_offset",
]

# 目标掩码名
MASK_SUFFIX = data_cfg.get("MASK_SUFFIX", "_mask")
mask_names = [f"{x}{MASK_SUFFIX}" for x in target_names]

# 由完整特征名自动推断气象特征：
# = 全部特征 - 时间特征 - 目标变量 - 目标掩码
meteo_names = [
    x for x in feature_names
    if x not in TIME_FEATURE_NAMES and x not in target_names and x not in mask_names
]

# 若 data_preprocessing 中提供了站点名则用它，否则自动补
site_names = data_cfg.get("sites_names", [])

set_seed(SEED)

print("\n========== 配置读取结果 ==========")
print(f"DATA_PATH: {DATA_PATH}")
print(f"FEATURE_NAMES_PATH: {FEATURE_NAMES_PATH}")
print(f"ADJ_PATH: {ADJ_PATH}")
print(f"OUTPUT_DIR: {OUTPUT_DIR}")
print(f"SEQ_LEN={SEQ_LEN}, PRED_LEN(固定为1)={PRED_LEN}, BATCH_SIZE={BATCH_SIZE}")
print(f"GAT_HIDDEN_DIM={GAT_HIDDEN_DIM}, LSTM_HIDDEN_DIM={LSTM_HIDDEN_DIM}, NUM_HEADS={NUM_HEADS}")
print(f"LR={LR}, PATIENCE={PATIENCE}, EPOCHS={EPOCHS}")
print(f"TRAIN/VAL/TEST = {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}")
print("全部特征:", feature_names)
print("目标特征:", target_names)
print("推断出的气象特征:", meteo_names)
print("目标掩码:", mask_names)
print("=================================\n")

if CONFIG_OUTPUT_DIM is not None and CONFIG_OUTPUT_DIM != len(target_names):
    print(f"警告：config 中 OUTPUT_DIM={CONFIG_OUTPUT_DIM}，但 target_names 数量={len(target_names)}。"
          f" 实际将以 target_names 数量为准。")

missing_time = [x for x in TIME_FEATURE_NAMES if x not in feature_names]
missing_target = [x for x in target_names if x not in feature_names]
missing_mask = [x for x in mask_names if x not in feature_names]

if missing_time:
    raise ValueError(f"缺少时间特征: {missing_time}")
if missing_target:
    raise ValueError(f"缺少目标特征: {missing_target}")
if missing_mask:
    raise ValueError(f"缺少目标掩码特征: {missing_mask}")
if not meteo_names:
    raise ValueError("没有推断出任何气象特征，请检查 feature_names 和 target_names。")


# =========================================================
# 2. 加载数据，构造输入输出索引
#    消融设置：输入 = 时间 + 气象；输出 = 目标水质
# =========================================================
all_data = np.load(DATA_PATH).astype(np.float32)   # [T, S, F_all]
print(f"数据形状 all_data: {all_data.shape}")

T_total, S_total, F_total = all_data.shape
if len(site_names) == 0:
    site_names = [f"site_{i}" for i in range(S_total)]

input_feature_names = TIME_FEATURE_NAMES + meteo_names
input_idx = [feature_names.index(x) for x in input_feature_names]
target_idx = [feature_names.index(x) for x in target_names]
mask_idx = [feature_names.index(x) for x in mask_names]

print("\n========== 消融实验设置 ==========")
print("输入特征（仅时间+气象）:")
print(input_feature_names)
print("预测目标（水质）:")
print(target_names)
print("掩码（仅用于 loss 和评估，不输入模型）:")
print(mask_names)
print("=================================\n")


# =========================================================
# 3. 按时间切分数据
# =========================================================
ratio_sum = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
if abs(ratio_sum - 1.0) > 1e-6:
    raise ValueError(f"训练/验证/测试比例之和必须为1，当前为 {ratio_sum}")

train_end = int(T_total * TRAIN_RATIO)
val_end = int(T_total * (TRAIN_RATIO + VAL_RATIO))

if train_end <= SEQ_LEN:
    raise ValueError("训练集长度不足以构造滑动窗口，请减小 SEQ_LEN 或调整数据划分。")
if val_end <= train_end:
    raise ValueError("验证集长度异常，请检查 TRAIN_RATIO 和 VAL_RATIO。")

train_raw = all_data[:train_end]
val_raw = all_data[train_end - SEQ_LEN:val_end]
test_raw = all_data[val_end - SEQ_LEN:]

print(f"train_raw: {train_raw.shape}")
print(f"val_raw  : {val_raw.shape}")
print(f"test_raw : {test_raw.shape}")


def split_xy_mask(raw_data, input_idx, target_idx, mask_idx):
    x = raw_data[:, :, input_idx].astype(np.float32)
    y = raw_data[:, :, target_idx].astype(np.float32)
    y_mask = raw_data[:, :, mask_idx].astype(np.float32)
    return x, y, y_mask


x_train_raw, y_train_raw, m_train_raw = split_xy_mask(train_raw, input_idx, target_idx, mask_idx)
x_val_raw, y_val_raw, m_val_raw = split_xy_mask(val_raw, input_idx, target_idx, mask_idx)
x_test_raw, y_test_raw, m_test_raw = split_xy_mask(test_raw, input_idx, target_idx, mask_idx)

print(f"x_train_raw: {x_train_raw.shape}, y_train_raw: {y_train_raw.shape}, m_train_raw: {m_train_raw.shape}")


# =========================================================
# 4. 标准化
#    X：对输入做标准化
#    y：只用训练集有效观测做目标标准化
# =========================================================
def fit_x_stats(x_train):
    x2 = x_train.reshape(-1, x_train.shape[-1])
    mean = x2.mean(axis=0).astype(np.float32)
    std = x2.std(axis=0).astype(np.float32)
    std[std < 1e-8] = 1.0
    return mean, std


def transform_x(x, mean, std):
    return ((x - mean[None, None, :]) / std[None, None, :]).astype(np.float32)


def fit_y_stats(y_train, y_mask_train, target_names):
    f_out = y_train.shape[-1]
    mean = np.zeros(f_out, dtype=np.float32)
    std = np.ones(f_out, dtype=np.float32)

    for k in range(f_out):
        vals = y_train[:, :, k]
        mask = y_mask_train[:, :, k] > 0.5
        valid_vals = vals[mask]
        if len(valid_vals) == 0:
            print(f"警告：目标 {target_names[k]} 在训练集没有有效观测，均值/标准差回退到 0/1")
            mean[k] = 0.0
            std[k] = 1.0
        else:
            mean[k] = valid_vals.mean()
            std_k = valid_vals.std()
            std[k] = std_k if std_k > 1e-8 else 1.0
    return mean, std


def transform_y(y, mean, std):
    return ((y - mean[None, None, :]) / std[None, None, :]).astype(np.float32)


def inverse_transform_y(y_scaled, mean, std):
    return y_scaled * std + mean


x_mean, x_std = fit_x_stats(x_train_raw)
x_train = transform_x(x_train_raw, x_mean, x_std)
x_val = transform_x(x_val_raw, x_mean, x_std)
x_test = transform_x(x_test_raw, x_mean, x_std)

y_mean, y_std = fit_y_stats(y_train_raw, m_train_raw, target_names)
y_train = transform_y(y_train_raw, y_mean, y_std)
y_val = transform_y(y_val_raw, y_mean, y_std)
y_test = transform_y(y_test_raw, y_mean, y_std)

np.savez(SCALER_SAVE_PATH, x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)


# =========================================================
# 5. 构造滑动窗口
# =========================================================
def make_sequences(x, y, y_mask, seq_len):
    """
    构造与第一段代码一致的单步样本：
    X: (num_samples, seq_len, S, F_in)
    Y: (num_samples, S, F_out)
    M: (num_samples, S, F_out)
    """
    x_seq, y_seq, m_seq = [], [], []
    t_total = x.shape[0]
    for t in range(seq_len, t_total):
        x_seq.append(x[t - seq_len:t])
        y_seq.append(y[t])
        m_seq.append(y_mask[t])
    x_seq = np.stack(x_seq).astype(np.float32)
    y_seq = np.stack(y_seq).astype(np.float32)
    m_seq = np.stack(m_seq).astype(np.float32)
    return x_seq, y_seq, m_seq


X_train, Y_train, M_train = make_sequences(x_train, y_train, m_train_raw, SEQ_LEN)
X_val, Y_val, M_val = make_sequences(x_val, y_val, m_val_raw, SEQ_LEN)
X_test, Y_test, M_test = make_sequences(x_test, y_test, m_test_raw, SEQ_LEN)

print(f"X_train: {X_train.shape}, Y_train: {Y_train.shape}, M_train: {M_train.shape}")
print(f"X_val  : {X_val.shape}, Y_val  : {Y_val.shape}, M_val  : {M_val.shape}")
print(f"X_test : {X_test.shape}, Y_test : {Y_test.shape}, M_test : {M_test.shape}")


# =========================================================
# 6. DataLoader
# =========================================================
train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(Y_train, dtype=torch.float32),
    torch.tensor(M_train, dtype=torch.float32),
)

val_dataset = TensorDataset(
    torch.tensor(X_val, dtype=torch.float32),
    torch.tensor(Y_val, dtype=torch.float32),
    torch.tensor(M_val, dtype=torch.float32),
)

test_dataset = TensorDataset(
    torch.tensor(X_test, dtype=torch.float32),
    torch.tensor(Y_test, dtype=torch.float32),
    torch.tensor(M_test, dtype=torch.float32),
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# =========================================================
# 7. 读取邻接矩阵
#    对 csv 更鲁棒：自动去掉非数值行列
# =========================================================
def load_adjacency(adj_path, num_nodes):
    if adj_path is None or not os.path.exists(adj_path):
        print("未提供有效 ADJ_PATH，使用单位阵作为邻接矩阵。")
        return np.eye(num_nodes, dtype=np.float32)

    ext = os.path.splitext(adj_path)[1].lower()
    if ext == ".npy":
        adj = np.load(adj_path).astype(np.float32)
    else:
        df = pd.read_csv(adj_path, header=None)
        # 尝试整体转数值；若有行名/列名会变成 NaN
        df = df.apply(pd.to_numeric, errors="coerce")
        # 去掉全 NaN 的行/列
        df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
        adj = df.to_numpy(dtype=np.float32)

    # 若首行首列是标签导致尺寸多 1，尝试自动修正
    if adj.shape == (num_nodes + 1, num_nodes + 1):
        adj = adj[1:, 1:]
    elif adj.shape == (num_nodes + 1, num_nodes):
        adj = adj[1:, :]
    elif adj.shape == (num_nodes, num_nodes + 1):
        adj = adj[:, 1:]

    if adj.shape != (num_nodes, num_nodes):
        raise ValueError(f"邻接矩阵形状应为 ({num_nodes}, {num_nodes})，当前为 {adj.shape}；请检查 ADJ_PATH 文件。")

    adj = (adj > 0).astype(np.float32)
    np.fill_diagonal(adj, 1.0)
    return adj


adj_matrix = load_adjacency(ADJ_PATH, S_total)
adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32, device=device)
print(f"邻接矩阵形状: {tuple(adj_matrix.shape)}")


# =========================================================
# 8. GAT-LSTM 模型
#    输入：[B, L, S, F_in]
#    输出：[B, S, F_out]（与第一段代码一致）
# =========================================================
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0, alpha=0.2):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Linear(out_dim, 1, bias=False)
        self.a_dst = nn.Linear(out_dim, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, adj):
        # h: [B, S, in_dim]
        Wh = self.W(h)                      # [B, S, out_dim]
        e_src = self.a_src(Wh)              # [B, S, 1]
        e_dst = self.a_dst(Wh)              # [B, S, 1]
        e = e_src + e_dst.transpose(1, 2)   # [B, S, S]
        e = self.leakyrelu(e)

        minus_inf = torch.full_like(e, -9e15)
        attention = torch.where(adj.unsqueeze(0) > 0, e, minus_inf)
        attention = torch.softmax(attention, dim=-1)
        attention = self.dropout(attention)

        h_prime = torch.bmm(attention, Wh)  # [B, S, out_dim]
        return h_prime


class MultiHeadGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads=4, dropout=0.0, alpha=0.2):
        super().__init__()
        self.heads = nn.ModuleList([
            GraphAttentionLayer(in_dim, hidden_dim, dropout=dropout, alpha=alpha)
            for _ in range(num_heads)
        ])
        self.out_proj = nn.Linear(hidden_dim * num_heads, hidden_dim)
        self.act = nn.ELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        # x: [B, S, in_dim]
        head_outs = [head(x, adj) for head in self.heads]
        x = torch.cat(head_outs, dim=-1)   # [B, S, hidden_dim * num_heads]
        x = self.out_proj(x)               # [B, S, hidden_dim]
        x = self.act(x)
        x = self.dropout(x)
        return x


class GATLSTM(nn.Module):
    def __init__(self, in_dim, gat_hidden_dim, lstm_hidden_dim, out_dim,
                 num_heads=4, dropout=0.2, alpha=0.2):
        super().__init__()
        self.out_dim = out_dim

        self.gat = MultiHeadGAT(
            in_dim=in_dim,
            hidden_dim=gat_hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            alpha=alpha,
        )
        self.lstm = nn.LSTM(
            input_size=gat_hidden_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden_dim, out_dim)

    def forward(self, x, adj):
        # x: [B, L, S, F_in]
        bsz, seq_len, num_nodes, _ = x.shape

        gat_seq = []
        for t in range(seq_len):
            xt = x[:, t, :, :]              # [B, S, F_in]
            ht = self.gat(xt, adj)          # [B, S, G]
            gat_seq.append(ht)

        gat_seq = torch.stack(gat_seq, dim=1)          # [B, L, S, G]
        gat_seq = gat_seq.permute(0, 2, 1, 3)          # [B, S, L, G]
        gat_seq = gat_seq.reshape(bsz * num_nodes, seq_len, -1)  # [B*S, L, G]

        lstm_out, _ = self.lstm(gat_seq)               # [B*S, L, H]
        last_hidden = lstm_out[:, -1, :]               # [B*S, H]
        last_hidden = self.dropout(last_hidden)

        out = self.fc(last_hidden)                     # [B*S, out_dim]
        out = out.reshape(bsz, num_nodes, self.out_dim)
        return out


# =========================================================
# 9. 损失函数与评估函数
# =========================================================
def masked_mse_loss(pred, target, mask):
    valid = mask > 0.5
    valid_count = valid.sum()
    if valid_count.item() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    diff2 = (pred - target) ** 2
    return diff2[valid].mean()


def compute_metrics(y_true, y_pred):
    if len(y_true) == 0:
        return {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan}
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {"R2": r2, "RMSE": rmse, "MAE": mae}


def evaluate_loader(model, loader, adj, y_mean, y_std, device):
    model.eval()
    losses = []
    pred_list, true_list, mask_list = [], [], []

    with torch.no_grad():
        for batch_x, batch_y, batch_m in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_m = batch_m.to(device)

            pred = model(batch_x, adj)
            loss = masked_mse_loss(pred, batch_y, batch_m)
            losses.append(loss.item())

            pred_np = pred.detach().cpu().numpy()
            y_np = batch_y.detach().cpu().numpy()
            m_np = batch_m.detach().cpu().numpy()

            pred_np = inverse_transform_y(
                pred_np,
                y_mean[None, None, :],
                y_std[None, None, :],
            )
            y_np = inverse_transform_y(
                y_np,
                y_mean[None, None, :],
                y_std[None, None, :],
            )

            pred_list.append(pred_np)
            true_list.append(y_np)
            mask_list.append(m_np)

    pred_all = np.concatenate(pred_list, axis=0)
    true_all = np.concatenate(true_list, axis=0)
    mask_all = np.concatenate(mask_list, axis=0)
    return float(np.mean(losses)), pred_all, true_all, mask_all


# =========================================================
# 10. 初始化模型
# =========================================================
input_dim = len(input_feature_names)
output_dim = len(target_names)

model = GATLSTM(
    in_dim=input_dim,
    gat_hidden_dim=GAT_HIDDEN_DIM,
    lstm_hidden_dim=LSTM_HIDDEN_DIM,
    out_dim=output_dim,
    num_heads=NUM_HEADS,
    dropout=DROPOUT,
    alpha=ALPHA,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

print(model)


# =========================================================
# 11. 训练
# =========================================================
best_val_loss = float("inf")
best_state = None
wait = 0
train_history = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    batch_losses = []

    for batch_x, batch_y, batch_m in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_m = batch_m.to(device)

        optimizer.zero_grad()
        pred = model(batch_x, adj_matrix)
        loss = masked_mse_loss(pred, batch_y, batch_m)
        loss.backward()
        optimizer.step()

        batch_losses.append(loss.item())

    train_loss = float(np.mean(batch_losses)) if batch_losses else np.nan
    val_loss, _, _, _ = evaluate_loader(model, val_loader, adj_matrix, y_mean, y_std, device)
    train_history.append([epoch, train_loss, val_loss])

    print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())
        torch.save(best_state, MODEL_SAVE_PATH)
        wait = 0
    else:
        wait += 1
        if wait >= PATIENCE:
            print(f"早停触发：连续 {PATIENCE} 个 epoch 验证集未提升。")
            break

# 保存训练历史
train_history_df = pd.DataFrame(train_history, columns=["epoch", "train_loss", "val_loss"])
train_history_df.to_csv(HISTORY_CSV_PATH, index=False, encoding="utf-8-sig")
print(f"训练历史已保存: {HISTORY_CSV_PATH}")


# =========================================================
# 12. 加载最佳模型并评估
# =========================================================
print("\n加载最佳模型并开始评估...")
if best_state is None:
    print("警告：训练过程中未捕获到最佳模型，使用当前模型参数进行评估。")
    best_state = copy.deepcopy(model.state_dict())
    torch.save(best_state, MODEL_SAVE_PATH)

model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
model.eval()

train_loss_best, train_pred, train_true, train_mask = evaluate_loader(
    model, train_loader, adj_matrix, y_mean, y_std, device
)
val_loss_best, val_pred, val_true, val_mask = evaluate_loader(
    model, val_loader, adj_matrix, y_mean, y_std, device
)
test_loss_best, test_pred, test_true, test_mask = evaluate_loader(
    model, test_loader, adj_matrix, y_mean, y_std, device
)

print(f"Train Loss: {train_loss_best:.6f}")
print(f"Val   Loss: {val_loss_best:.6f}")
print(f"Test  Loss: {test_loss_best:.6f}")


# =========================================================
# 13. 保存预测结果（与第一段代码保持完全一致）
# =========================================================
np.save(os.path.join(OUTPUT_DIR, 'train_pred.npy'), train_pred)
np.save(os.path.join(OUTPUT_DIR, 'train_target.npy'), train_true)
np.save(os.path.join(OUTPUT_DIR, 'train_mask.npy'), train_mask)

np.save(os.path.join(OUTPUT_DIR, 'val_pred.npy'), val_pred)
np.save(os.path.join(OUTPUT_DIR, 'val_target.npy'), val_true)
np.save(os.path.join(OUTPUT_DIR, 'val_mask.npy'), val_mask)

np.save(os.path.join(OUTPUT_DIR, 'test_pred.npy'), test_pred)
np.save(os.path.join(OUTPUT_DIR, 'test_target.npy'), test_true)
np.save(os.path.join(OUTPUT_DIR, 'test_mask.npy'), test_mask)

np.savez(
    PRED_SAVE_PATH,
    train_pred=train_pred,
    train_true=train_true,
    train_mask=train_mask,
    val_pred=val_pred,
    val_true=val_true,
    val_mask=val_mask,
    test_pred=test_pred,
    test_true=test_true,
    test_mask=test_mask,
)
print(f"训练/验证/测试预测结果已保存到: {OUTPUT_DIR}")
print(f"汇总结果已保存到: {PRED_SAVE_PATH}")


# =========================================================
# 14. 计算整体指标和分目标指标
# =========================================================
valid_all = test_mask > 0.5
pred_valid_all = test_pred[valid_all]
target_valid_all = test_true[valid_all]

overall_metrics = compute_metrics(target_valid_all, pred_valid_all)

print("\n====== 测试集整体性能（GAT-LSTM） ======")
print(f"R2  : {overall_metrics['R2']:.4f}")
print(f"RMSE: {overall_metrics['RMSE']:.4f}")
print(f"MAE : {overall_metrics['MAE']:.4f}")

metrics_rows = []
metrics_rows.append({
    "scope": "overall",
    "target": "all",
    "R2": overall_metrics["R2"],
    "RMSE": overall_metrics["RMSE"],
    "MAE": overall_metrics["MAE"],
    "sample_count": int(valid_all.sum()),
})

for k, target_name in enumerate(target_names):
    valid_k = test_mask[..., k] > 0.5
    pred_k = test_pred[..., k][valid_k]
    true_k = test_true[..., k][valid_k]
    metric_k = compute_metrics(true_k, pred_k)

    print(f"\n--- 目标: {target_name} ---")
    print(f"R2  : {metric_k['R2']:.4f}")
    print(f"RMSE: {metric_k['RMSE']:.4f}")
    print(f"MAE : {metric_k['MAE']:.4f}")

    metrics_rows.append({
        "scope": "target",
        "target": target_name,
        "R2": metric_k["R2"],
        "RMSE": metric_k["RMSE"],
        "MAE": metric_k["MAE"],
        "sample_count": int(valid_k.sum()),
    })

metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv(RESULT_CSV_PATH, index=False, encoding="utf-8-sig")
print(f"\n指标汇总已保存到: {RESULT_CSV_PATH}")


# =========================================================
# 15. 保存配置与结果说明文件
# =========================================================
config_result_txt = os.path.join(OUTPUT_DIR, 'Configs and results.txt')
with open(config_result_txt, 'w', encoding='utf-8') as f:
    f.write("========== Configuration ==========\n")
    f.write(f"DATA_PATH: {DATA_PATH}\n")
    f.write(f"FEATURE_NAMES_PATH: {FEATURE_NAMES_PATH}\n")
    f.write(f"ADJ_PATH: {ADJ_PATH}\n")
    f.write(f"OUTPUT_DIR: {OUTPUT_DIR}\n")
    f.write(f"SEQ_LEN: {SEQ_LEN}\n")
    f.write(f"PRED_LEN: 1\n")
    f.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
    f.write(f"GAT_HIDDEN_DIM: {GAT_HIDDEN_DIM}\n")
    f.write(f"LSTM_HIDDEN_DIM: {LSTM_HIDDEN_DIM}\n")
    f.write(f"NUM_HEADS: {NUM_HEADS}\n")
    f.write(f"DROPOUT: {DROPOUT}\n")
    f.write(f"ALPHA: {ALPHA}\n")
    f.write(f"LR: {LR}\n")
    f.write(f"WEIGHT_DECAY: {WEIGHT_DECAY}\n")
    f.write(f"PATIENCE: {PATIENCE}\n")
    f.write(f"EPOCHS: {EPOCHS}\n")
    f.write(f"TRAIN_RATIO: {TRAIN_RATIO}\n")
    f.write(f"VAL_RATIO: {VAL_RATIO}\n")
    f.write(f"TEST_RATIO: {TEST_RATIO}\n")
    f.write(f"SEED: {SEED}\n")
    f.write(f"INPUT_FEATURES: {input_feature_names}\n")
    f.write(f"TARGETS: {target_names}\n")
    f.write("===================================\n\n")

    f.write("========== Best Loss ==========\n")
    f.write(f"Train Loss: {train_loss_best:.6f}\n")
    f.write(f"Val Loss: {val_loss_best:.6f}\n")
    f.write(f"Test Loss: {test_loss_best:.6f}\n")
    f.write("================================\n\n")

    f.write("========== Overall Metrics ==========\n")
    f.write(f"R2: {overall_metrics['R2']:.4f}\n")
    f.write(f"RMSE: {overall_metrics['RMSE']:.4f}\n")
    f.write(f"MAE: {overall_metrics['MAE']:.4f}\n")
    f.write("=====================================\n\n")

    f.write("========== Metrics By Target ==========\n")
    for _, row in metrics_df[metrics_df["scope"] == "target"].iterrows():
        f.write(f"Target: {row['target']}\n")
        f.write(f"  R2: {row['R2']:.4f}\n")
        f.write(f"  RMSE: {row['RMSE']:.4f}\n")
        f.write(f"  MAE: {row['MAE']:.4f}\n")
        f.write(f"  Sample Count: {int(row['sample_count'])}\n")
    f.write("=======================================\n")

print(f"配置与结果说明已保存到: {config_result_txt}")


# =========================================================
# 16. 绘制整体散点图（与第一段代码一致）
# =========================================================
if len(target_valid_all) > 0:
    plt.figure(figsize=(6, 6))
    plt.scatter(target_valid_all, pred_valid_all, alpha=0.3, s=4)

    xy_min = min(target_valid_all.min(), pred_valid_all.min())
    xy_max = max(target_valid_all.max(), pred_valid_all.max())
    plt.plot([xy_min, xy_max], [xy_min, xy_max], 'r--', lw=2)

    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(
        f"Overall (R2={overall_metrics['R2']:.3f}, "
        f"RMSE={overall_metrics['RMSE']:.3f}, "
        f"MAE={overall_metrics['MAE']:.3f})"
    )
    plt.tight_layout()
    scatter_path = os.path.join(OUTPUT_DIR, 'scatter_overall.png')
    plt.savefig(scatter_path, dpi=150)
    plt.close()
    print(f"整体散点图已保存到: {scatter_path}")
else:
    print("警告：测试集没有有效观测点，未绘制整体散点图。")


# =========================================================
# 17. 按目标分别绘制散点图
# =========================================================
for k, target_name in enumerate(target_names):
    valid_k = test_mask[..., k] > 0.5
    pred_k = test_pred[..., k][valid_k]
    true_k = test_true[..., k][valid_k]

    if len(true_k) == 0:
        print(f"目标 {target_name} 没有有效观测点，跳过散点图。")
        continue

    metric_k = compute_metrics(true_k, pred_k)

    plt.figure(figsize=(6, 6))
    plt.scatter(true_k, pred_k, alpha=0.3, s=4)

    xy_min = min(true_k.min(), pred_k.min())
    xy_max = max(true_k.max(), pred_k.max())
    plt.plot([xy_min, xy_max], [xy_min, xy_max], 'r--', lw=2)

    plt.xlabel(f'True {target_name}')
    plt.ylabel(f'Pred {target_name}')
    plt.title(
        f"{target_name} (R2={metric_k['R2']:.3f}, "
        f"RMSE={metric_k['RMSE']:.3f}, "
        f"MAE={metric_k['MAE']:.3f})"
    )
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, f'scatter_{target_name}.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"目标 {target_name} 的散点图已保存到: {fig_path}")

print("\n全部流程完成。")