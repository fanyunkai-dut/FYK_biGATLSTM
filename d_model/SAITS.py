import os
import random
import pickle
import numpy as np
import yaml
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用的设备: {device}")

# ==================== 从 configs.yaml 读取配置 ====================
CONFIG_PATH = "configs.yaml"
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config_all = yaml.safe_load(f)

cfg = config_all.get('SAITS_block_imputation_training', {})

SEQ_LEN = cfg.get('SEQ_LEN', 192)
WINDOW_STRIDE = cfg.get('WINDOW_STRIDE', 1)
BATCH_SIZE = cfg.get('BATCH_SIZE', 128)
D_MODEL = cfg.get('D_MODEL', 128)
N_HEAD = cfg.get('N_HEAD', 4)
N_LAYERS = cfg.get('N_LAYERS', 2)
D_FF = cfg.get('D_FF', 256)
DROPOUT = cfg.get('DROPOUT', 0.1)
LEARNING_RATE = cfg.get('LEARNING_RATE', 1e-3)
PATIENCE = cfg.get('PATIENCE', 10)
MAX_EPOCHS = cfg.get('MAX_EPOCHS', 100)
TRAIN_RATIO = cfg.get('TRAIN_RATIO', 0.7)
VAL_RATIO = cfg.get('VAL_RATIO', 0.15)
TEST_RATIO = cfg.get('TEST_RATIO', 0.15)
SEED = cfg.get('SEED', 42)
NUM_WORKERS = cfg.get('NUM_WORKERS', 4)
PERSISTENT_WORKERS = cfg.get('PERSISTENT_WORKERS', True)

DATA_PATH = cfg.get('DATA_PATH')
FEATURE_NAMES_PATH = cfg.get('FEATURE_NAMES_PATH')
OUTPUT_DIR = cfg.get('OUTPUT_DIR')
TARGET_NAMES = cfg.get('target_names', ['总氮', '总磷', '水温', 'pH', '溶解氧'])
MASK_SUFFIX = cfg.get('MASK_SUFFIX', '_mask')
DT_SUFFIX = cfg.get('DT_SUFFIX', '_dt')

block_cfg = cfg.get('block_masking', {})
TRAIN_BLOCK_LENGTHS = block_cfg.get('train_block_lengths', [3, 6, 12, 18, 42, 180])
TRAIN_NUM_BLOCKS_PER_SAMPLE = block_cfg.get('train_num_blocks_per_sample', 1)
TRAIN_BLOCK_SEED = block_cfg.get('train_block_seed', 42)
VAL_BLOCK_LENGTHS = block_cfg.get('val_block_lengths', [3, 6, 12, 18, 42, 180])
VAL_NUM_BLOCKS_PER_SAMPLE = block_cfg.get('val_num_blocks_per_sample', 1)
VAL_BLOCK_SEED = block_cfg.get('val_block_seed', 202)
TEST_BLOCK_LENGTHS = block_cfg.get('test_block_lengths', [3, 6, 12, 18, 42, 180])
TEST_NUM_BLOCKS_PER_SAMPLE = block_cfg.get('test_num_blocks_per_sample', 1)
TEST_BLOCK_SEED = block_cfg.get('test_block_seed', 123)

os.makedirs(OUTPUT_DIR, exist_ok=True)


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


# ==================== 数据与遮挡工具函数 ====================
def extract_target_triplets(all_feature_names, target_names, mask_suffix='_mask', dt_suffix='_dt'):
    value_indices, mask_indices, dt_indices = [], [], []
    for name in target_names:
        if name not in all_feature_names:
            raise ValueError(f"找不到目标变量: {name}")
        mask_name = name + mask_suffix
        dt_name = name + dt_suffix
        if mask_name not in all_feature_names:
            raise ValueError(f"找不到掩码列: {mask_name}")
        if dt_name not in all_feature_names:
            raise ValueError(f"找不到 dt 列: {dt_name}")
        value_indices.append(all_feature_names.index(name))
        mask_indices.append(all_feature_names.index(mask_name))
        dt_indices.append(all_feature_names.index(dt_name))
    return value_indices, mask_indices, dt_indices


def fit_target_scaler(values, masks):
    # values: (T, N, F), masks: (T, N, F)
    F = values.shape[-1]
    means = np.zeros(F, dtype=np.float32)
    stds = np.ones(F, dtype=np.float32)
    for f in range(F):
        valid = (masks[..., f] > 0.5) & (~np.isnan(values[..., f]))
        vals = values[..., f][valid]
        if vals.size == 0:
            raise ValueError(f"目标第 {f} 维在训练集中没有有效观测，无法标准化")
        means[f] = np.mean(vals)
        std = np.std(vals)
        stds[f] = std if std > 0 else 1.0
    return means, stds


def apply_value_standardization(values, masks, means, stds):
    out = values.copy().astype(np.float32)
    for f in range(values.shape[-1]):
        valid = (masks[..., f] > 0.5) & (~np.isnan(out[..., f]))
        out[..., f][valid] = (out[..., f][valid] - means[f]) / stds[f]
    return out


def choose_non_overlapping_blocks(valid_all, candidate_lengths, num_blocks, rng):
    T_local = len(valid_all)
    occupied = np.zeros(T_local, dtype=bool)
    blocks = []
    lengths = list(candidate_lengths)

    for _ in range(num_blocks):
        rng.shuffle(lengths)
        chosen = False
        for L in lengths:
            if L > T_local:
                continue
            possible_starts = []
            for start in range(0, T_local - L + 1):
                end = start + L
                if occupied[start:end].any():
                    continue
                if np.all(valid_all[start:end]):
                    possible_starts.append(start)
            if possible_starts:
                start = int(rng.choice(possible_starts))
                end = start + L
                blocks.append((start, end))
                occupied[start:end] = True
                chosen = True
                break
        if not chosen:
            break
    return blocks


def mask_single_station_window(values, masks, block_lengths, num_blocks, rng):
    """
    values: (T, F) 标准化后的目标值
    masks:  (T, F) 自然观测掩码

    返回：
        x_in:        (T, F) 人工遮挡后的输入值（缺失处填0）
        m_in:        (T, F) 人工遮挡后的输入 mask
        target_x:    (T, F) 目标真实值（标准化）
        target_mask: (T, F) 仅在“人工遮挡且原本真实观测”的位置为1
    """
    T_local, F = values.shape
    target_x = values.copy().astype(np.float32)
    natural_mask = masks.copy().astype(np.float32)

    valid_all = np.all(natural_mask > 0.5, axis=1)
    blocks = choose_non_overlapping_blocks(valid_all, block_lengths, num_blocks, rng)

    target_mask = np.zeros((T_local, F), dtype=np.float32)
    x_in = values.copy().astype(np.float32)
    m_in = natural_mask.copy().astype(np.float32)

    for start, end in blocks:
        target_mask[start:end, :] = 1.0
        x_in[start:end, :] = 0.0
        m_in[start:end, :] = 0.0

    # 自然缺失位置也保证输入值为0
    x_in[m_in < 0.5] = 0.0

    return x_in, m_in, target_x, target_mask


def build_station_windows(values, masks, seq_len, stride):
    """
    把 (T, N, F) 切成 station-wise windows
    输出为 list，每个元素：
        values_win: (seq_len, F)
        masks_win:  (seq_len, F)
    """
    T_local, N_local, F = values.shape
    windows = []
    for n in range(N_local):
        for start in range(0, T_local - seq_len + 1, stride):
            end = start + seq_len
            windows.append((
                values[start:end, n, :].copy(),
                masks[start:end, n, :].copy(),
            ))
    return windows


class SAITSTrainDataset(Dataset):
    def __init__(self, windows, block_lengths, num_blocks_per_sample, seed=42):
        self.windows = windows
        self.block_lengths = list(block_lengths)
        self.num_blocks_per_sample = num_blocks_per_sample
        self.base_seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        values, masks = self.windows[idx]
        rng = np.random.default_rng(self.base_seed + self.epoch * 1000003 + idx)
        x_in, m_in, target_x, target_mask = mask_single_station_window(
            values, masks, self.block_lengths, self.num_blocks_per_sample, rng
        )
        return (
            torch.tensor(x_in, dtype=torch.float32),
            torch.tensor(m_in, dtype=torch.float32),
            torch.tensor(target_x, dtype=torch.float32),
            torch.tensor(target_mask, dtype=torch.float32),
        )


class SAITSEvalDataset(Dataset):
    def __init__(self, windows, block_length, num_blocks_per_sample=1, seed=123):
        self.samples = []
        for idx, (values, masks) in enumerate(windows):
            rng = np.random.default_rng(seed + idx)
            x_in, m_in, target_x, target_mask = mask_single_station_window(
                values, masks, [block_length], num_blocks_per_sample, rng
            )
            self.samples.append((x_in, m_in, target_x, target_mask))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x_in, m_in, target_x, target_mask = self.samples[idx]
        return (
            torch.tensor(x_in, dtype=torch.float32),
            torch.tensor(m_in, dtype=torch.float32),
            torch.tensor(target_x, dtype=torch.float32),
            torch.tensor(target_mask, dtype=torch.float32),
        )


# ==================== SAITS 模型 ====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, T, D)
        return x + self.pe[:, :x.size(1), :]


class DiagonalMaskedEncoder(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout, n_layers, max_len=5000):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x):
        # x: (B, T, D)
        T = x.size(1)
        # 不允许一个时间步直接看自己，避免简单复制
        diag_mask = torch.eye(T, dtype=torch.bool, device=x.device)
        x = self.pos_encoder(x)
        x = self.encoder(x, mask=diag_mask)
        return x


class SAITS(nn.Module):
    def __init__(self, input_dim, d_model=128, n_head=4, n_layers=2, d_ff=256, dropout=0.1, seq_len=192):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        # stage 1
        self.input_proj_1 = nn.Linear(input_dim * 2, d_model)
        self.encoder_1 = DiagonalMaskedEncoder(d_model, n_head, d_ff, dropout, n_layers, max_len=seq_len + 8)
        self.output_proj_1 = nn.Linear(d_model, input_dim)

        # stage 2
        self.input_proj_2 = nn.Linear(input_dim * 2, d_model)
        self.encoder_2 = DiagonalMaskedEncoder(d_model, n_head, d_ff, dropout, n_layers, max_len=seq_len + 8)
        self.output_proj_2 = nn.Linear(d_model, input_dim)

        # combine
        self.combine_gate = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x, m):
        # x, m: (B, T, F)
        inp1 = torch.cat([x, m], dim=-1)
        h1 = self.encoder_1(self.input_proj_1(inp1))
        x_tilde_1 = self.output_proj_1(h1)
        x_hat_1 = m * x + (1.0 - m) * x_tilde_1

        inp2 = torch.cat([x_hat_1, m], dim=-1)
        h2 = self.encoder_2(self.input_proj_2(inp2))
        x_tilde_2 = self.output_proj_2(h2)
        x_hat_2 = m * x + (1.0 - m) * x_tilde_2

        gate = self.combine_gate(torch.cat([x_tilde_1, x_tilde_2], dim=-1))
        x_comb = gate * x_tilde_1 + (1.0 - gate) * x_tilde_2
        x_final = m * x + (1.0 - m) * x_comb

        return {
            'imputation_1': x_hat_1,
            'imputation_2': x_hat_2,
            'imputation_final': x_final,
        }


def saits_loss(outputs, target_x, target_mask):
    denom = target_mask.sum() + 1e-8
    l1 = (((outputs['imputation_1'] - target_x) ** 2) * target_mask).sum() / denom
    l2 = (((outputs['imputation_2'] - target_x) ** 2) * target_mask).sum() / denom
    lf = (((outputs['imputation_final'] - target_x) ** 2) * target_mask).sum() / denom
    return 0.5 * l1 + 0.5 * l2 + lf


# ==================== 数据加载与预处理 ====================
print("加载数据...")
full_data = np.load(DATA_PATH).astype(np.float32)
with open(FEATURE_NAMES_PATH, 'rb') as f:
    all_feature_names = pickle.load(f)

T, N, F_full = full_data.shape
print(f"完整数据形状: {full_data.shape}, 站点数: {N}, 特征数: {F_full}")
print(f"完整特征名: {all_feature_names}")

value_idxs, mask_idxs, dt_idxs = extract_target_triplets(all_feature_names, TARGET_NAMES, MASK_SUFFIX, DT_SUFFIX)
values_full = full_data[:, :, value_idxs]   # (T, N, 5)
masks_full = full_data[:, :, mask_idxs]     # (T, N, 5)

# 时间切分：block-imputation，不需要 one-step 的 seq 偏移
train_len = int(T * TRAIN_RATIO)
val_len = int(T * VAL_RATIO)
test_len = T - train_len - val_len

train_slice = slice(0, train_len)
val_slice = slice(train_len, train_len + val_len)
test_slice = slice(train_len + val_len, T)

train_values_raw = values_full[train_slice]
val_values_raw = values_full[val_slice]
test_values_raw = values_full[test_slice]
train_masks_raw = masks_full[train_slice]
val_masks_raw = masks_full[val_slice]
test_masks_raw = masks_full[test_slice]

# 目标变量标准化：只基于训练集真实观测
value_means, value_stds = fit_target_scaler(train_values_raw, train_masks_raw)
train_values_std = apply_value_standardization(train_values_raw, train_masks_raw, value_means, value_stds)
val_values_std = apply_value_standardization(val_values_raw, val_masks_raw, value_means, value_stds)
test_values_std = apply_value_standardization(test_values_raw, test_masks_raw, value_means, value_stds)

# 缺失位置填0，是否可用由 mask 控制
train_values_std = np.nan_to_num(train_values_std, nan=0.0).astype(np.float32)
val_values_std = np.nan_to_num(val_values_std, nan=0.0).astype(np.float32)
test_values_std = np.nan_to_num(test_values_std, nan=0.0).astype(np.float32)
train_masks_raw = train_masks_raw.astype(np.float32)
val_masks_raw = val_masks_raw.astype(np.float32)
test_masks_raw = test_masks_raw.astype(np.float32)

scaler_params = {
    'target_names': TARGET_NAMES,
    'value_means': value_means,
    'value_stds': value_stds,
}
with open(os.path.join(OUTPUT_DIR, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler_params, f)
print("标准化参数已保存到 scaler.pkl")

train_windows = build_station_windows(train_values_std, train_masks_raw, SEQ_LEN, WINDOW_STRIDE)
val_windows = build_station_windows(val_values_std, val_masks_raw, SEQ_LEN, WINDOW_STRIDE)
test_windows = build_station_windows(test_values_std, test_masks_raw, SEQ_LEN, WINDOW_STRIDE)
print(f"训练窗口数: {len(train_windows)}, 验证窗口数: {len(val_windows)}, 测试窗口数: {len(test_windows)}")

train_dataset = SAITSTrainDataset(train_windows, TRAIN_BLOCK_LENGTHS, TRAIN_NUM_BLOCKS_PER_SAMPLE, TRAIN_BLOCK_SEED)
val_dataset_dict = {L: SAITSEvalDataset(val_windows, L, VAL_NUM_BLOCKS_PER_SAMPLE, VAL_BLOCK_SEED + i)
                    for i, L in enumerate(VAL_BLOCK_LENGTHS)}
test_dataset_dict = {L: SAITSEvalDataset(test_windows, L, TEST_NUM_BLOCKS_PER_SAMPLE, TEST_BLOCK_SEED + i)
                     for i, L in enumerate(TEST_BLOCK_LENGTHS)}

pin_memory = torch.cuda.is_available()
use_persistent = bool(PERSISTENT_WORKERS and NUM_WORKERS > 0)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=pin_memory,
    num_workers=NUM_WORKERS,
    persistent_workers=use_persistent,
)
val_loader_dict = {
    L: DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=NUM_WORKERS,
        persistent_workers=use_persistent,
    )
    for L, ds in val_dataset_dict.items()
}
test_loader_dict = {
    L: DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=NUM_WORKERS,
        persistent_workers=use_persistent,
    )
    for L, ds in test_dataset_dict.items()
}

# ==================== 模型训练 ====================
model = SAITS(
    input_dim=len(TARGET_NAMES),
    d_model=D_MODEL,
    n_head=N_HEAD,
    n_layers=N_LAYERS,
    d_ff=D_FF,
    dropout=DROPOUT,
    seq_len=SEQ_LEN,
).to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

best_val_loss = float('inf')
best_epoch = -1
counter = 0


def run_epoch(loader, training=False, epoch=0):
    if training:
        model.train()
        if hasattr(loader.dataset, 'set_epoch'):
            loader.dataset.set_epoch(epoch)
    else:
        model.eval()

    total_loss = 0.0
    total_batches = 0

    for x_in, m_in, target_x, target_mask in loader:
        x_in = x_in.to(device, non_blocking=True)
        m_in = m_in.to(device, non_blocking=True)
        target_x = target_x.to(device, non_blocking=True)
        target_mask = target_mask.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad()

        outputs = model(x_in, m_in)
        loss = saits_loss(outputs, target_x, target_mask)

        if training:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(total_batches, 1)


print("\n开始训练 SAITS baseline...")
for epoch in range(MAX_EPOCHS):
    train_loss = run_epoch(train_loader, training=True, epoch=epoch)
    val_losses = []
    for L, loader in val_loader_dict.items():
        val_l = run_epoch(loader, training=False)
        val_losses.append(val_l)
    val_loss = float(np.mean(val_losses)) if val_losses else np.nan
    scheduler.step(val_loss)

    print(f"Epoch {epoch + 1}/{MAX_EPOCHS} - Train Loss: {train_loss:.6f}, Val Mean Loss: {val_loss:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model.pth'))
        counter = 0
    else:
        counter += 1
        if counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch + 1}")
            break


# ==================== 评估 ====================
def evaluate_loader(loader):
    model.eval()
    all_pred, all_target, all_evalmask = [], [], []
    with torch.no_grad():
        for x_in, m_in, target_x, target_mask in loader:
            x_in = x_in.to(device, non_blocking=True)
            m_in = m_in.to(device, non_blocking=True)
            target_x = target_x.to(device, non_blocking=True)
            target_mask = target_mask.to(device, non_blocking=True)

            outputs = model(x_in, m_in)
            imputation = outputs['imputation_final']

            all_pred.append(imputation.cpu().numpy())
            all_target.append(target_x.cpu().numpy())
            all_evalmask.append(target_mask.cpu().numpy())

    pred = np.concatenate(all_pred, axis=0)
    target = np.concatenate(all_target, axis=0)
    evalmask = np.concatenate(all_evalmask, axis=0)
    valid = evalmask > 0.5

    if np.sum(valid) == 0:
        return None, pred, target, evalmask

    pred_valid = pred[valid]
    target_valid = target[valid]
    metrics = {
        'R2': float(r2_score(target_valid, pred_valid)),
        'RMSE': float(np.sqrt(mean_squared_error(target_valid, pred_valid))),
        'MAE': float(mean_absolute_error(target_valid, pred_valid)),
        'n_valid': int(np.sum(valid)),
    }
    return metrics, pred, target, evalmask


print("\n加载最佳模型进行测试...")
model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_model.pth'), map_location=device))
model.eval()

all_results = {}
for L, loader in test_loader_dict.items():
    metrics, pred, target, evalmask = evaluate_loader(loader)
    all_results[L] = metrics
    np.save(os.path.join(OUTPUT_DIR, f'block{L}_pred.npy'), pred)
    np.save(os.path.join(OUTPUT_DIR, f'block{L}_target.npy'), target)
    np.save(os.path.join(OUTPUT_DIR, f'block{L}_evalmask.npy'), evalmask)

print("====== SAITS 人工大块遮挡 block-imputation 测试 ======")
for L in sorted(all_results.keys()):
    m = all_results[L]
    if m is None:
        print(f"[block={L}] 没有可评估样本")
    else:
        print(f"[block={L}] R2={m['R2']:.4f}, RMSE={m['RMSE']:.4f}, MAE={m['MAE']:.4f}, n_valid={m['n_valid']}")

with open(os.path.join(OUTPUT_DIR, 'Configs and results.txt'), 'w', encoding='utf-8') as f:
    f.write("========== SAITS Block Imputation Configuration ==========\n")
    f.write(f"SEQ_LEN: {SEQ_LEN}\n")
    f.write(f"WINDOW_STRIDE: {WINDOW_STRIDE}\n")
    f.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
    f.write(f"D_MODEL: {D_MODEL}\n")
    f.write(f"N_HEAD: {N_HEAD}\n")
    f.write(f"N_LAYERS: {N_LAYERS}\n")
    f.write(f"D_FF: {D_FF}\n")
    f.write(f"DROPOUT: {DROPOUT}\n")
    f.write(f"LEARNING_RATE: {LEARNING_RATE}\n")
    f.write(f"PATIENCE: {PATIENCE}\n")
    f.write(f"MAX_EPOCHS: {MAX_EPOCHS}\n")
    f.write(f"TRAIN_RATIO: {TRAIN_RATIO}\n")
    f.write(f"VAL_RATIO: {VAL_RATIO}\n")
    f.write(f"TEST_RATIO: {TEST_RATIO}\n")
    f.write(f"SEED: {SEED}\n")
    f.write(f"NUM_WORKERS: {NUM_WORKERS}\n")
    f.write(f"PERSISTENT_WORKERS: {PERSISTENT_WORKERS}\n")
    f.write(f"DATA_PATH: {DATA_PATH}\n")
    f.write(f"FEATURE_NAMES_PATH: {FEATURE_NAMES_PATH}\n")
    f.write(f"OUTPUT_DIR: {OUTPUT_DIR}\n")
    f.write(f"TARGET_NAMES: {TARGET_NAMES}\n")
    f.write(f"TRAIN_BLOCK_LENGTHS: {TRAIN_BLOCK_LENGTHS}\n")
    f.write(f"TRAIN_NUM_BLOCKS_PER_SAMPLE: {TRAIN_NUM_BLOCKS_PER_SAMPLE}\n")
    f.write(f"VAL_BLOCK_LENGTHS: {VAL_BLOCK_LENGTHS}\n")
    f.write(f"VAL_NUM_BLOCKS_PER_SAMPLE: {VAL_NUM_BLOCKS_PER_SAMPLE}\n")
    f.write(f"TEST_BLOCK_LENGTHS: {TEST_BLOCK_LENGTHS}\n")
    f.write(f"TEST_NUM_BLOCKS_PER_SAMPLE: {TEST_NUM_BLOCKS_PER_SAMPLE}\n")
    f.write(f"best_epoch: {best_epoch}\n")
    f.write(f"best_val_loss: {best_val_loss:.6f}\n\n")
    f.write("========== Test Metrics ==========\n")
    for L in sorted(all_results.keys()):
        m = all_results[L]
        if m is None:
            f.write(f"block={L}: no valid samples\n")
        else:
            f.write(f"block={L}: R2={m['R2']:.4f}, RMSE={m['RMSE']:.4f}, MAE={m['MAE']:.4f}, n_valid={m['n_valid']}\n")

print(f"\n结果已保存到 {OUTPUT_DIR}")




