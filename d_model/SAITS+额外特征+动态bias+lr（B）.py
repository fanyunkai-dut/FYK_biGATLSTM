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

cfg = config_all.get('SAITS_exogenous_dynamic_bias_lowrank_gate_block_imputation_training', {})


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
if OUTPUT_DIR is None:
    raise ValueError("配置项 OUTPUT_DIR 缺失，请检查对应 section 是否完整。")

TARGET_NAMES = cfg.get('target_names', ['总氮', '总磷', '水温', 'pH', '溶解氧'])
EXOGENOUS_FEATURE_NAMES = cfg.get('exogenous_feature_names', [])
TIME_FEATURE_NAMES = cfg.get('time_feature_names', ['hour_sin', 'hour_cos', 'month_sin', 'month_cos'])
MASK_SUFFIX = cfg.get('MASK_SUFFIX', '_mask')
DT_SUFFIX = cfg.get('DT_SUFFIX', '_dt')
EXO_HIDDEN = cfg.get('EXO_HIDDEN', 32)

BIAS_HIDDEN = cfg.get('BIAS_HIDDEN', 64)
BIAS_DROPOUT = cfg.get('BIAS_DROPOUT', 0.1)
BIAS_REG_WEIGHT = cfg.get('BIAS_REG_WEIGHT', 1e-4)
BIAS_INIT_SCALE = cfg.get('BIAS_INIT_SCALE', 1.0)
USE_VARIABLE_BIAS = cfg.get('USE_VARIABLE_BIAS', True)
USE_STATION_BIAS = cfg.get('USE_STATION_BIAS', True)
USE_TIME_BIAS = cfg.get('USE_TIME_BIAS', True)
USE_TIME_GATE = cfg.get('USE_TIME_GATE', True)

# ===== 低秩先验参数 =====
LR_RANK = cfg.get('LR_RANK', 5)
LR_MAX_ITERS = cfg.get('LR_MAX_ITERS', 8)
LR_TOL = cfg.get('LR_TOL', 1e-5)

# ===== 输出端小 gate 参数 =====
GATE_HIDDEN = cfg.get('GATE_HIDDEN', 64)
GATE_STATION_EMB = cfg.get('GATE_STATION_EMB', 16)
GATE_DROPOUT = cfg.get('GATE_DROPOUT', 0.1)

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


def extract_feature_indices(all_feature_names, feature_names, feature_group_name='features'):
    if not feature_names:
        raise ValueError(f"{feature_group_name} 不能为空")
    idxs = []
    missing = []
    for name in feature_names:
        if name not in all_feature_names:
            missing.append(name)
        else:
            idxs.append(all_feature_names.index(name))
    if missing:
        raise ValueError(f"以下 {feature_group_name} 不存在于特征中: {missing}")
    return idxs


def fit_target_scaler(values, masks):
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


def fit_exo_scaler(exo):
    F = exo.shape[-1]
    means = np.zeros(F, dtype=np.float32)
    stds = np.ones(F, dtype=np.float32)
    for f in range(F):
        vals = exo[..., f].reshape(-1)
        valid = ~np.isnan(vals)
        if np.sum(valid) == 0:
            means[f] = 0.0
            stds[f] = 1.0
        else:
            means[f] = np.mean(vals[valid])
            std = np.std(vals[valid])
            stds[f] = std if std > 0 else 1.0
    return means, stds


def apply_exo_standardization(exo, means, stds):
    out = exo.copy().astype(np.float32)
    for f in range(exo.shape[-1]):
        valid = ~np.isnan(out[..., f])
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


def masked_low_rank_reconstruct(values_in, masks_in, rank=5, max_iters=8, tol=1e-5):
    """
    仅用当前样本“遮挡后的输入 x_in / m_in”生成 low-rank prior。
    不使用 target_x，避免信息泄露。

    values_in: (T, F)
    masks_in:  (T, F)
    return:    (T, F)
    """
    X_obs = values_in.astype(np.float32)
    M = (masks_in > 0.5)
    T_local, F = X_obs.shape

    if np.sum(M) == 0:
        return np.zeros_like(X_obs, dtype=np.float32)

    rank = int(max(1, min(rank, min(T_local, F))))

    col_means = np.zeros(F, dtype=np.float32)
    for f in range(F):
        valid = M[:, f]
        if np.any(valid):
            col_means[f] = float(np.mean(X_obs[valid, f]))
        else:
            col_means[f] = 0.0

    X_filled = X_obs.copy()
    missing_pos = ~M
    if np.any(missing_pos):
        X_filled[missing_pos] = np.take(col_means, np.where(missing_pos)[1])

    prev_missing = X_filled[missing_pos].copy() if np.any(missing_pos) else None

    for _ in range(max_iters):
        try:
            U, S, Vt = np.linalg.svd(X_filled, full_matrices=False)
        except np.linalg.LinAlgError:
            return X_filled.astype(np.float32)

        X_low = (U[:, :rank] * S[:rank]) @ Vt[:rank, :]
        X_filled[M] = X_obs[M]
        X_filled[missing_pos] = X_low[missing_pos]

        if np.any(missing_pos):
            cur_missing = X_filled[missing_pos]
            diff = np.mean((cur_missing - prev_missing) ** 2)
            prev_missing = cur_missing.copy()
            if diff < tol:
                break

    return X_filled.astype(np.float32)


def mask_single_station_window(
    values,
    masks,
    exo,
    station_idx,
    time_feats,
    block_lengths,
    num_blocks,
    rng,
    lr_rank=5,
    lr_max_iters=8,
    lr_tol=1e-5,
):
    T_local, F = values.shape
    target_x = values.copy().astype(np.float32)
    natural_mask = masks.copy().astype(np.float32)
    exo_in = exo.copy().astype(np.float32)

    valid_all = np.all(natural_mask > 0.5, axis=1)
    blocks = choose_non_overlapping_blocks(valid_all, block_lengths, num_blocks, rng)

    target_mask = np.zeros((T_local, F), dtype=np.float32)
    x_in = values.copy().astype(np.float32)
    m_in = natural_mask.copy().astype(np.float32)

    for start, end in blocks:
        target_mask[start:end, :] = 1.0
        x_in[start:end, :] = 0.0
        m_in[start:end, :] = 0.0

    x_in[m_in < 0.5] = 0.0

    # ===== 关键：low-rank prior 在人工遮挡后生成，只看 x_in / m_in =====
    lr_in = masked_low_rank_reconstruct(
        x_in,
        m_in,
        rank=lr_rank,
        max_iters=lr_max_iters,
        tol=lr_tol,
    )

    return (
        x_in,
        m_in,
        exo_in,
        lr_in,
        int(station_idx),
        time_feats.astype(np.float32),
        target_x,
        target_mask,
    )


def build_station_windows(values, masks, exo, time_feats_full, seq_len, stride):
    T_local, N_local, _ = values.shape
    windows = []
    for n in range(N_local):
        for start in range(0, T_local - seq_len + 1, stride):
            end = start + seq_len
            windows.append((
                values[start:end, n, :].copy(),
                masks[start:end, n, :].copy(),
                exo[start:end, n, :].copy(),
                int(n),
                time_feats_full[start:end, n, :].copy(),
            ))
    return windows


class SAITSExoDynamicBiasLRGateTrainDataset(Dataset):
    def __init__(self, windows, block_lengths, num_blocks_per_sample, seed=42, lr_rank=5, lr_max_iters=8, lr_tol=1e-5):
        self.windows = windows
        self.block_lengths = list(block_lengths)
        self.num_blocks_per_sample = num_blocks_per_sample
        self.base_seed = int(seed)
        self.epoch = 0
        self.lr_rank = int(lr_rank)
        self.lr_max_iters = int(lr_max_iters)
        self.lr_tol = float(lr_tol)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        values, masks, exo, station_idx, time_feats = self.windows[idx]
        rng = np.random.default_rng(self.base_seed + self.epoch * 1000003 + idx)
        x_in, m_in, exo_in, lr_in, station_idx, time_feats, target_x, target_mask = mask_single_station_window(
            values,
            masks,
            exo,
            station_idx,
            time_feats,
            self.block_lengths,
            self.num_blocks_per_sample,
            rng,
            lr_rank=self.lr_rank,
            lr_max_iters=self.lr_max_iters,
            lr_tol=self.lr_tol,
        )
        return (
            torch.tensor(x_in, dtype=torch.float32),
            torch.tensor(m_in, dtype=torch.float32),
            torch.tensor(exo_in, dtype=torch.float32),
            torch.tensor(lr_in, dtype=torch.float32),
            torch.tensor(station_idx, dtype=torch.long),
            torch.tensor(time_feats, dtype=torch.float32),
            torch.tensor(target_x, dtype=torch.float32),
            torch.tensor(target_mask, dtype=torch.float32),
        )


class SAITSExoDynamicBiasLRGateEvalDataset(Dataset):
    def __init__(self, windows, block_length, num_blocks_per_sample=1, seed=123, lr_rank=5, lr_max_iters=8, lr_tol=1e-5):
        self.samples = []
        for idx, (values, masks, exo, station_idx, time_feats) in enumerate(windows):
            rng = np.random.default_rng(seed + idx)
            sample = mask_single_station_window(
                values,
                masks,
                exo,
                station_idx,
                time_feats,
                [block_length],
                num_blocks_per_sample,
                rng,
                lr_rank=lr_rank,
                lr_max_iters=lr_max_iters,
                lr_tol=lr_tol,
            )
            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x_in, m_in, exo_in, lr_in, station_idx, time_feats, target_x, target_mask = self.samples[idx]
        return (
            torch.tensor(x_in, dtype=torch.float32),
            torch.tensor(m_in, dtype=torch.float32),
            torch.tensor(exo_in, dtype=torch.float32),
            torch.tensor(lr_in, dtype=torch.float32),
            torch.tensor(station_idx, dtype=torch.long),
            torch.tensor(time_feats, dtype=torch.float32),
            torch.tensor(target_x, dtype=torch.float32),
            torch.tensor(target_mask, dtype=torch.float32),
        )


# ==================== 模型 ====================
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
        T_local = x.size(1)
        diag_mask = torch.eye(T_local, dtype=torch.bool, device=x.device)
        x = self.pos_encoder(x)
        x = self.encoder(x, mask=diag_mask)
        return x


class DynamicTVSBias(nn.Module):
    def __init__(
        self,
        input_dim,
        num_stations,
        time_feat_dim,
        bias_hidden=64,
        dropout=0.1,
        use_variable_bias=True,
        use_station_bias=True,
        use_time_bias=True,
        use_time_gate=True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.use_variable_bias = use_variable_bias
        self.use_station_bias = use_station_bias
        self.use_time_bias = use_time_bias
        self.use_time_gate = use_time_gate
        if self.use_variable_bias:
            self.variable_bias = nn.Parameter(torch.zeros(input_dim))
        else:
            self.register_parameter('variable_bias', None)

        if self.use_station_bias:
            self.station_emb = nn.Embedding(num_stations, bias_hidden)
            self.station_proj = nn.Linear(bias_hidden, input_dim)
        else:
            self.station_emb = None
            self.station_proj = None

        if self.use_time_bias:
            self.time_encoder = nn.Sequential(
                nn.Linear(time_feat_dim, bias_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.time_proj = nn.Linear(bias_hidden, input_dim)
        else:
            self.time_encoder = None
            self.time_proj = None

        if self.use_time_bias and self.use_time_gate:
            gate_in_dim = bias_hidden
            if self.use_station_bias:
                gate_in_dim += bias_hidden
            self.fusion_gate = nn.Sequential(
                nn.Linear(gate_in_dim, bias_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(bias_hidden, input_dim),
                nn.Sigmoid(),
            )
        else:
            self.fusion_gate = None

    def forward(self, station_idx, time_feats):
        batch_size, seq_len, _ = time_feats.shape
        bias = torch.zeros(batch_size, seq_len, self.input_dim, device=time_feats.device)

        station_h = None
        if self.use_variable_bias:
            bias = bias + self.variable_bias.view(1, 1, -1)

        if self.use_station_bias:
            station_h = self.station_emb(station_idx)
            station_bias = self.station_proj(station_h).unsqueeze(1)
            bias = bias + station_bias

        if self.use_time_bias:
            time_h = self.time_encoder(time_feats)
            time_bias = self.time_proj(time_h)
            if self.fusion_gate is not None:
                gate_parts = [time_h]
                if station_h is not None:
                    gate_parts.append(station_h.unsqueeze(1).expand(-1, seq_len, -1))
                gate_context = torch.cat(gate_parts, dim=-1)
                gate = self.fusion_gate(gate_context)
                time_bias = gate * time_bias
            bias = bias + time_bias

        bias_reg = (bias ** 2).mean()
        return bias, bias_reg


class SmallLRFusionGate(nn.Module):
    """
    小 gate：输出每个时间步、每个变量上的 alpha(t,f)
    alpha 越大，越信 low-rank prior；越小，越信 SAITS+bias
    """
    def __init__(self, input_dim, exo_hidden, time_feat_dim, num_stations, station_emb_dim=16, hidden=64, dropout=0.1):
        super().__init__()
        self.station_emb = nn.Embedding(num_stations, station_emb_dim)
        gate_in_dim = input_dim * 3 + exo_hidden + time_feat_dim + station_emb_dim
        self.net = nn.Sequential(
            nn.Linear(gate_in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, m, x_saits_bias, lr, exo_h, station_idx, time_feats):
        B, T_local, _ = m.shape
        station_h = self.station_emb(station_idx).unsqueeze(1).expand(-1, T_local, -1)
        gate_in = torch.cat([m, x_saits_bias, lr, exo_h, time_feats, station_h], dim=-1)
        alpha = self.net(gate_in)
        return alpha


class SAITSExogenousDynamicBiasLowRankGate(nn.Module):
    def __init__(
        self,
        input_dim,
        exo_dim,
        num_stations,
        time_feat_dim,
        d_model=128,
        n_head=4,
        n_layers=2,
        d_ff=256,
        dropout=0.1,
        seq_len=192,
        exo_hidden=32,
        bias_hidden=64,
        bias_dropout=0.1,
        bias_init_scale=1.0,
        use_variable_bias=True,
        use_station_bias=True,
        use_time_bias=True,
        use_time_gate=True,
        gate_hidden=64,
        gate_station_emb=16,
        gate_dropout=0.1,
    ):
        super().__init__()
        self.input_dim = input_dim

        self.exo_proj = nn.Sequential(
            nn.Linear(exo_dim, exo_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 主干仍然基本保持原来的 SAITS + exogenous 输入结构
        total_in_dim = input_dim * 2 + exo_hidden
        self.input_proj_1 = nn.Linear(total_in_dim, d_model)
        self.encoder_1 = DiagonalMaskedEncoder(d_model, n_head, d_ff, dropout, n_layers, max_len=seq_len + 8)
        self.output_proj_1 = nn.Linear(d_model, input_dim)

        self.input_proj_2 = nn.Linear(total_in_dim, d_model)
        self.encoder_2 = DiagonalMaskedEncoder(d_model, n_head, d_ff, dropout, n_layers, max_len=seq_len + 8)
        self.output_proj_2 = nn.Linear(d_model, input_dim)

        self.combine_gate = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.Sigmoid(),
        )

        self.bias_branch = DynamicTVSBias(
            input_dim=input_dim,
            num_stations=num_stations,
            time_feat_dim=time_feat_dim,
            bias_hidden=bias_hidden,
            dropout=bias_dropout,
            use_variable_bias=use_variable_bias,
            use_station_bias=use_station_bias,
            use_time_bias=use_time_bias,
            use_time_gate=use_time_gate,
        )
        self.bias_scale = nn.Parameter(torch.tensor(float(bias_init_scale), dtype=torch.float32))

        # 输出端 low-rank prior 小 gate
        self.lr_fusion_gate = SmallLRFusionGate(
            input_dim=input_dim,
            exo_hidden=exo_hidden,
            time_feat_dim=time_feat_dim,
            num_stations=num_stations,
            station_emb_dim=gate_station_emb,
            hidden=gate_hidden,
            dropout=gate_dropout,
        )

    def forward(self, x, m, exo, lr, station_idx, time_feats):
        exo_h = self.exo_proj(exo)

        inp1 = torch.cat([x, m, exo_h], dim=-1)
        h1 = self.encoder_1(self.input_proj_1(inp1))
        x_tilde_1 = self.output_proj_1(h1)
        x_hat_1 = m * x + (1.0 - m) * x_tilde_1

        inp2 = torch.cat([x_hat_1, m, exo_h], dim=-1)
        h2 = self.encoder_2(self.input_proj_2(inp2))
        x_tilde_2 = self.output_proj_2(h2)
        x_hat_2 = m * x + (1.0 - m) * x_tilde_2

        gate_12 = self.combine_gate(torch.cat([x_tilde_1, x_tilde_2], dim=-1))
        x_comb = gate_12 * x_tilde_1 + (1.0 - gate_12) * x_tilde_2

        bias_term, bias_reg = self.bias_branch(station_idx, time_feats)
        x_saits_bias = x_comb + self.bias_scale * bias_term

        # ===== 输出端小 gate 决定 alpha(t,f) =====
        alpha = self.lr_fusion_gate(m, x_saits_bias, lr, exo_h, station_idx, time_feats)
        x_fused_missing = alpha * lr + (1.0 - alpha) * x_saits_bias
        x_final = m * x + (1.0 - m) * x_fused_missing

        return {
            'imputation_1': x_hat_1,
            'imputation_2': x_hat_2,
            'imputation_saits_bias': m * x + (1.0 - m) * x_saits_bias,
            'imputation_final': x_final,
            'alpha': alpha,
            'bias_term': bias_term,
            'bias_reg': bias_reg,
        }


def saits_exo_dynamic_bias_loss(outputs, target_x, target_mask, bias_reg_weight=1e-4):
    denom = target_mask.sum() + 1e-8
    l1 = (((outputs['imputation_1'] - target_x) ** 2) * target_mask).sum() / denom
    l2 = (((outputs['imputation_2'] - target_x) ** 2) * target_mask).sum() / denom
    lf = (((outputs['imputation_final'] - target_x) ** 2) * target_mask).sum() / denom
    return 0.5 * l1 + 0.5 * l2 + lf + bias_reg_weight * outputs['bias_reg']


# ==================== 数据加载与预处理 ====================
print("加载数据...")
full_data = np.load(DATA_PATH).astype(np.float32)
with open(FEATURE_NAMES_PATH, 'rb') as f:
    all_feature_names = pickle.load(f)

T, N, F_full = full_data.shape
print(f"完整数据形状: {full_data.shape}, 站点数: {N}, 特征数: {F_full}")
print(f"完整特征名: {all_feature_names}")

value_idxs, mask_idxs, _ = extract_target_triplets(all_feature_names, TARGET_NAMES, MASK_SUFFIX, DT_SUFFIX)
exo_idxs = extract_feature_indices(all_feature_names, EXOGENOUS_FEATURE_NAMES, 'exogenous_feature_names')
time_feature_idxs = extract_feature_indices(all_feature_names, TIME_FEATURE_NAMES, 'time_feature_names')

values_full = full_data[:, :, value_idxs]
masks_full = full_data[:, :, mask_idxs]
exo_full = full_data[:, :, exo_idxs]
time_feats_full = full_data[:, :, time_feature_idxs]
time_feats_full = np.nan_to_num(time_feats_full, nan=0.0).astype(np.float32)

print(f"目标变量: {TARGET_NAMES}")
print(f"额外特征数: {len(EXOGENOUS_FEATURE_NAMES)}")
print(f"时间 bias 特征: {TIME_FEATURE_NAMES}")

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
train_exo_raw = exo_full[train_slice]
val_exo_raw = exo_full[val_slice]
test_exo_raw = exo_full[test_slice]
train_time_feats = time_feats_full[train_slice]
val_time_feats = time_feats_full[val_slice]
test_time_feats = time_feats_full[test_slice]

value_means, value_stds = fit_target_scaler(train_values_raw, train_masks_raw)
train_values_std = apply_value_standardization(train_values_raw, train_masks_raw, value_means, value_stds)
val_values_std = apply_value_standardization(val_values_raw, val_masks_raw, value_means, value_stds)
test_values_std = apply_value_standardization(test_values_raw, test_masks_raw, value_means, value_stds)

exo_means, exo_stds = fit_exo_scaler(train_exo_raw)
train_exo_std = apply_exo_standardization(train_exo_raw, exo_means, exo_stds)
val_exo_std = apply_exo_standardization(val_exo_raw, exo_means, exo_stds)
test_exo_std = apply_exo_standardization(test_exo_raw, exo_means, exo_stds)

train_values_std = np.nan_to_num(train_values_std, nan=0.0).astype(np.float32)
val_values_std = np.nan_to_num(val_values_std, nan=0.0).astype(np.float32)
test_values_std = np.nan_to_num(test_values_std, nan=0.0).astype(np.float32)
train_exo_std = np.nan_to_num(train_exo_std, nan=0.0).astype(np.float32)
val_exo_std = np.nan_to_num(val_exo_std, nan=0.0).astype(np.float32)
test_exo_std = np.nan_to_num(test_exo_std, nan=0.0).astype(np.float32)
train_masks_raw = train_masks_raw.astype(np.float32)
val_masks_raw = val_masks_raw.astype(np.float32)
test_masks_raw = test_masks_raw.astype(np.float32)

scaler_params = {
    'target_names': TARGET_NAMES,
    'exogenous_feature_names': EXOGENOUS_FEATURE_NAMES,
    'time_feature_names': TIME_FEATURE_NAMES,
    'value_means': value_means,
    'value_stds': value_stds,
    'exo_means': exo_means,
    'exo_stds': exo_stds,
    'lr_rank': LR_RANK,
    'lr_max_iters': LR_MAX_ITERS,
    'lr_tol': LR_TOL,
}
with open(os.path.join(OUTPUT_DIR, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler_params, f)
print("标准化参数已保存到 scaler.pkl")

train_windows = build_station_windows(train_values_std, train_masks_raw, train_exo_std, train_time_feats, SEQ_LEN, WINDOW_STRIDE)
val_windows = build_station_windows(val_values_std, val_masks_raw, val_exo_std, val_time_feats, SEQ_LEN, WINDOW_STRIDE)
test_windows = build_station_windows(test_values_std, test_masks_raw, test_exo_std, test_time_feats, SEQ_LEN, WINDOW_STRIDE)
print(f"训练窗口数: {len(train_windows)}, 验证窗口数: {len(val_windows)}, 测试窗口数: {len(test_windows)}")

train_dataset = SAITSExoDynamicBiasLRGateTrainDataset(
    train_windows,
    TRAIN_BLOCK_LENGTHS,
    TRAIN_NUM_BLOCKS_PER_SAMPLE,
    TRAIN_BLOCK_SEED,
    lr_rank=LR_RANK,
    lr_max_iters=LR_MAX_ITERS,
    lr_tol=LR_TOL,
)
val_dataset_dict = {
    L: SAITSExoDynamicBiasLRGateEvalDataset(
        val_windows,
        L,
        VAL_NUM_BLOCKS_PER_SAMPLE,
        VAL_BLOCK_SEED + i,
        lr_rank=LR_RANK,
        lr_max_iters=LR_MAX_ITERS,
        lr_tol=LR_TOL,
    )
    for i, L in enumerate(VAL_BLOCK_LENGTHS)
}
test_dataset_dict = {
    L: SAITSExoDynamicBiasLRGateEvalDataset(
        test_windows,
        L,
        TEST_NUM_BLOCKS_PER_SAMPLE,
        TEST_BLOCK_SEED + i,
        lr_rank=LR_RANK,
        lr_max_iters=LR_MAX_ITERS,
        lr_tol=LR_TOL,
    )
    for i, L in enumerate(TEST_BLOCK_LENGTHS)
}

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
model = SAITSExogenousDynamicBiasLowRankGate(
    input_dim=len(TARGET_NAMES),
    exo_dim=len(EXOGENOUS_FEATURE_NAMES),
    num_stations=N,
    time_feat_dim=len(TIME_FEATURE_NAMES),
    d_model=D_MODEL,
    n_head=N_HEAD,
    n_layers=N_LAYERS,
    d_ff=D_FF,
    dropout=DROPOUT,
    seq_len=SEQ_LEN,
    exo_hidden=EXO_HIDDEN,
    bias_hidden=BIAS_HIDDEN,
    bias_dropout=BIAS_DROPOUT,
    bias_init_scale=BIAS_INIT_SCALE,
    use_variable_bias=USE_VARIABLE_BIAS,
    use_station_bias=USE_STATION_BIAS,
    use_time_bias=USE_TIME_BIAS,
    use_time_gate=USE_TIME_GATE,
    gate_hidden=GATE_HIDDEN,
    gate_station_emb=GATE_STATION_EMB,
    gate_dropout=GATE_DROPOUT,
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

    for x_in, m_in, exo_in, lr_in, station_idx, time_feats, target_x, target_mask in loader:
        x_in = x_in.to(device, non_blocking=True)
        m_in = m_in.to(device, non_blocking=True)
        exo_in = exo_in.to(device, non_blocking=True)
        lr_in = lr_in.to(device, non_blocking=True)
        station_idx = station_idx.to(device, non_blocking=True)
        time_feats = time_feats.to(device, non_blocking=True)
        target_x = target_x.to(device, non_blocking=True)
        target_mask = target_mask.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad()

        outputs = model(x_in, m_in, exo_in, lr_in, station_idx, time_feats)
        loss = saits_exo_dynamic_bias_loss(outputs, target_x, target_mask, bias_reg_weight=BIAS_REG_WEIGHT)

        if training:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(total_batches, 1)


print("开始训练 SAITS + exogenous + dynamic bias + low-rank prior gate fusion...")
for epoch in range(MAX_EPOCHS):
    train_loss = run_epoch(train_loader, training=True, epoch=epoch)
    val_losses = []
    for _, loader in val_loader_dict.items():
        val_losses.append(run_epoch(loader, training=False))
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
    all_pred, all_target, all_evalmask, all_bias, all_lr, all_alpha = [], [], [], [], [], []
    with torch.no_grad():
        for x_in, m_in, exo_in, lr_in, station_idx, time_feats, target_x, target_mask in loader:
            x_in = x_in.to(device, non_blocking=True)
            m_in = m_in.to(device, non_blocking=True)
            exo_in = exo_in.to(device, non_blocking=True)
            lr_in = lr_in.to(device, non_blocking=True)
            station_idx = station_idx.to(device, non_blocking=True)
            time_feats = time_feats.to(device, non_blocking=True)
            target_x = target_x.to(device, non_blocking=True)
            target_mask = target_mask.to(device, non_blocking=True)

            outputs = model(x_in, m_in, exo_in, lr_in, station_idx, time_feats)
            imputation = outputs['imputation_final']
            bias_term = outputs['bias_term']
            alpha = outputs['alpha']

            all_pred.append(imputation.cpu().numpy())
            all_target.append(target_x.cpu().numpy())
            all_evalmask.append(target_mask.cpu().numpy())
            all_bias.append(bias_term.cpu().numpy())
            all_lr.append(lr_in.cpu().numpy())
            all_alpha.append(alpha.cpu().numpy())

    pred = np.concatenate(all_pred, axis=0)
    target = np.concatenate(all_target, axis=0)
    evalmask = np.concatenate(all_evalmask, axis=0)
    bias = np.concatenate(all_bias, axis=0)
    lr = np.concatenate(all_lr, axis=0)
    alpha = np.concatenate(all_alpha, axis=0)
    valid = evalmask > 0.5

    if np.sum(valid) == 0:
        return None, pred, target, evalmask, bias, lr, alpha

    pred_valid = pred[valid]
    target_valid = target[valid]
    metrics = {
        'R2': float(r2_score(target_valid, pred_valid)),
        'RMSE': float(np.sqrt(mean_squared_error(target_valid, pred_valid))),
        'MAE': float(mean_absolute_error(target_valid, pred_valid)),
        'n_valid': int(np.sum(valid)),
        'mean_alpha_on_eval': float(alpha[valid].mean()),
    }
    return metrics, pred, target, evalmask, bias, lr, alpha


print("加载最佳模型进行测试...")
model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_model.pth'), map_location=device))
model.eval()

all_results = {}
for L, loader in test_loader_dict.items():
    metrics, pred, target, evalmask, bias, lr, alpha = evaluate_loader(loader)
    all_results[L] = metrics
    np.save(os.path.join(OUTPUT_DIR, f'block{L}_pred.npy'), pred)
    np.save(os.path.join(OUTPUT_DIR, f'block{L}_target.npy'), target)
    np.save(os.path.join(OUTPUT_DIR, f'block{L}_evalmask.npy'), evalmask)
    np.save(os.path.join(OUTPUT_DIR, f'block{L}_bias.npy'), bias)
    np.save(os.path.join(OUTPUT_DIR, f'block{L}_lowrank_prior.npy'), lr)
    np.save(os.path.join(OUTPUT_DIR, f'block{L}_alpha.npy'), alpha)

print("====== SAITS + Exogenous + Dynamic Bias + Low-Rank Prior + Small Gate 人工大块遮挡 block-imputation 测试 ======")
for L in sorted(all_results.keys()):
    m = all_results[L]
    if m is None:
        print(f"[block={L}] 没有可评估样本")
    else:
        print(
            f"[block={L}] R2={m['R2']:.4f}, RMSE={m['RMSE']:.4f}, "
            f"MAE={m['MAE']:.4f}, n_valid={m['n_valid']}, "
            f"mean_alpha_on_eval={m['mean_alpha_on_eval']:.4f}"
        )

with open(os.path.join(OUTPUT_DIR, 'Configs and results.txt'), 'w', encoding='utf-8') as f:
    f.write("========== SAITS Exogenous Dynamic Bias Low-Rank Prior Small Gate Configuration ==========")
    f.write(f"SEQ_LEN: {SEQ_LEN}")
    f.write(f"WINDOW_STRIDE: {WINDOW_STRIDE}")
    f.write(f"BATCH_SIZE: {BATCH_SIZE}")
    f.write(f"D_MODEL: {D_MODEL}")
    f.write(f"N_HEAD: {N_HEAD}")
    f.write(f"N_LAYERS: {N_LAYERS}")
    f.write(f"D_FF: {D_FF}")
    f.write(f"DROPOUT: {DROPOUT}")
    f.write(f"LEARNING_RATE: {LEARNING_RATE}")
    f.write(f"PATIENCE: {PATIENCE}")
    f.write(f"MAX_EPOCHS: {MAX_EPOCHS}")
    f.write(f"TRAIN_RATIO: {TRAIN_RATIO}")
    f.write(f"VAL_RATIO: {VAL_RATIO}")
    f.write(f"TEST_RATIO: {TEST_RATIO}")
    f.write(f"SEED: {SEED}")
    f.write(f"NUM_WORKERS: {NUM_WORKERS}")
    f.write(f"PERSISTENT_WORKERS: {PERSISTENT_WORKERS}")
    f.write(f"EXO_HIDDEN: {EXO_HIDDEN}")
    f.write(f"LR_RANK: {LR_RANK}")
    f.write(f"LR_MAX_ITERS: {LR_MAX_ITERS}")
    f.write(f"LR_TOL: {LR_TOL}")
    f.write(f"GATE_HIDDEN: {GATE_HIDDEN}")
    f.write(f"GATE_STATION_EMB: {GATE_STATION_EMB}")
    f.write(f"GATE_DROPOUT: {GATE_DROPOUT}")
    f.write(f"DATA_PATH: {DATA_PATH}")
    f.write(f"FEATURE_NAMES_PATH: {FEATURE_NAMES_PATH}")
    f.write(f"OUTPUT_DIR: {OUTPUT_DIR}")
    f.write(f"TARGET_NAMES: {TARGET_NAMES}")
    f.write(f"EXOGENOUS_FEATURE_NAMES: {EXOGENOUS_FEATURE_NAMES}")
    f.write(f"TIME_FEATURE_NAMES: {TIME_FEATURE_NAMES}")
    f.write(f"BIAS_HIDDEN: {BIAS_HIDDEN}")
    f.write(f"BIAS_DROPOUT: {BIAS_DROPOUT}")
    f.write(f"BIAS_REG_WEIGHT: {BIAS_REG_WEIGHT}")
    f.write(f"BIAS_INIT_SCALE: {BIAS_INIT_SCALE}")
    f.write(f"USE_VARIABLE_BIAS: {USE_VARIABLE_BIAS}")
    f.write(f"USE_STATION_BIAS: {USE_STATION_BIAS}")
    f.write(f"USE_TIME_BIAS: {USE_TIME_BIAS}")
    f.write(f"USE_TIME_GATE: {USE_TIME_GATE}")
    f.write(f"TRAIN_BLOCK_LENGTHS: {TRAIN_BLOCK_LENGTHS}")
    f.write(f"TRAIN_NUM_BLOCKS_PER_SAMPLE: {TRAIN_NUM_BLOCKS_PER_SAMPLE}")
    f.write(f"VAL_BLOCK_LENGTHS: {VAL_BLOCK_LENGTHS}")
    f.write(f"VAL_NUM_BLOCKS_PER_SAMPLE: {VAL_NUM_BLOCKS_PER_SAMPLE}")
    f.write(f"TEST_BLOCK_LENGTHS: {TEST_BLOCK_LENGTHS}")
    f.write(f"TEST_NUM_BLOCKS_PER_SAMPLE: {TEST_NUM_BLOCKS_PER_SAMPLE}")
    f.write(f"best_epoch: {best_epoch}")
    f.write(f"best_val_loss: {best_val_loss:.6f}")
    f.write(f"learned_bias_scale: {model.bias_scale.item():.6f}")
    f.write("========== Test Metrics ==========")
    for L in sorted(all_results.keys()):
        m = all_results[L]
        if m is None:
            f.write(f"block={L}: no valid samples")
        else:
            f.write(
                f"block={L}: R2={m['R2']:.4f}, RMSE={m['RMSE']:.4f}, "
                f"MAE={m['MAE']:.4f}, n_valid={m['n_valid']}, "
                f"mean_alpha_on_eval={m['mean_alpha_on_eval']:.4f}"
            )

print(f"结果已保存到 {OUTPUT_DIR}")