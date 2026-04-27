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

cfg = config_all.get('GRIN_exogenous_block_imputation_training', {})

SEQ_LEN = cfg.get('SEQ_LEN', 192)
WINDOW_STRIDE = cfg.get('WINDOW_STRIDE', 1)
BATCH_SIZE = cfg.get('BATCH_SIZE', 128)
HIDDEN_SIZE = cfg.get('HIDDEN_SIZE', 128)
EXO_HIDDEN = cfg.get('EXO_HIDDEN', 32)
DROPOUT = cfg.get('DROPOUT', 0.1)
LEARNING_RATE = cfg.get('LEARNING_RATE', 1e-3)
PATIENCE = cfg.get('PATIENCE', 10)
MAX_EPOCHS = cfg.get('MAX_EPOCHS', 100)
TRAIN_RATIO = cfg.get('TRAIN_RATIO', 0.7)
VAL_RATIO = cfg.get('VAL_RATIO', 0.15)
TEST_RATIO = cfg.get('TEST_RATIO', 0.15)
SEED = cfg.get('SEED', 42)
CONSISTENCY_WEIGHT = cfg.get('CONSISTENCY_WEIGHT', 0.1)
USE_LOG1P_DT = cfg.get('USE_LOG1P_DT', True)
NUM_WORKERS = cfg.get('NUM_WORKERS', 4)
PERSISTENT_WORKERS = cfg.get('PERSISTENT_WORKERS', True)

DATA_PATH = cfg.get('DATA_PATH')
FEATURE_NAMES_PATH = cfg.get('FEATURE_NAMES_PATH')
ADJ_PATH = cfg.get('ADJ_PATH')
OUTPUT_DIR = cfg.get('OUTPUT_DIR')
TARGET_NAMES = cfg.get('target_names', ['总氮', '总磷', '水温', 'pH', '溶解氧'])
EXOGENOUS_FEATURE_NAMES = cfg.get('exogenous_feature_names', [])
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


def extract_exogenous_indices(all_feature_names, exogenous_feature_names):
    if not exogenous_feature_names:
        raise ValueError("exogenous_feature_names 不能为空")
    exo_indices = []
    for name in exogenous_feature_names:
        if name not in all_feature_names:
            raise ValueError(f"找不到外生特征: {name}")
        exo_indices.append(all_feature_names.index(name))
    return exo_indices


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


def fit_exo_scaler(exo):
    # exo: (T, N, F_exo)
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


def transform_dt(dt, use_log1p=True):
    dt = dt.astype(np.float32)
    return np.log1p(dt) if use_log1p else dt


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


def recompute_dt_from_mask(mask_2d):
    # mask_2d: (T, F)
    T_local, F = mask_2d.shape
    dt = np.zeros((T_local, F), dtype=np.float32)
    for f in range(F):
        gap = 0.0
        for t in range(T_local):
            if mask_2d[t, f] > 0.5:
                gap = 0.0
                dt[t, f] = 0.0
            else:
                gap += 1.0
                dt[t, f] = gap
    return dt


def load_adjacency(adj_path, num_nodes):
    ext = os.path.splitext(adj_path)[1].lower()

    if ext == '.npy':
        adj = np.load(adj_path)
    elif ext == '.csv':
        adj = np.loadtxt(adj_path, delimiter=',', dtype=np.float32)
    else:
        raise ValueError(f"不支持的邻接矩阵格式: {ext}")

    if adj.ndim != 2:
        raise ValueError(f"邻接矩阵必须是二维数组，实际 ndim={adj.ndim}")

    if adj.shape != (num_nodes, num_nodes):
        raise ValueError(
            f"邻接矩阵形状应为 ({num_nodes}, {num_nodes})，实际为 {adj.shape}。\n"
            f"如果你的 CSV 里还有额外的行名/列名，请先去掉；\n"
            f"如果它本来就是纯数字无表头矩阵，那就检查文件里是否多出空列或分隔符。"
        )

    adj = (adj != 0).astype(np.float32)
    np.fill_diagonal(adj, 1.0)

    deg = np.sum(adj, axis=1)
    deg_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
    deg_inv_sqrt[deg == 0] = 0.0
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt
    return adj.astype(np.float32), adj_norm.astype(np.float32)


def build_graph_windows(values, masks, dts_raw, exo, seq_len, stride):
    """
    输入：
        values:  (T, N, F)
        masks:   (T, N, F)
        dts_raw: (T, N, F)  原始 dt（还没 log1p）
        exo:     (T, N, F_exo)
    输出：list，每个元素是一个全站点窗口：
        values_win: (seq_len, N, F)
        masks_win:  (seq_len, N, F)
        dts_win:    (seq_len, N, F)
        exo_win:    (seq_len, N, F_exo)
    """
    T_local = values.shape[0]
    windows = []
    for start in range(0, T_local - seq_len + 1, stride):
        end = start + seq_len
        windows.append((
            values[start:end].copy(),
            masks[start:end].copy(),
            dts_raw[start:end].copy(),
            exo[start:end].copy(),
        ))
    return windows


def mask_single_station_graph_window(values, masks, dts_raw, exo, block_lengths, num_blocks, rng, use_log1p_dt=True):
    """
    对“全站点窗口”中的一个随机站点做人为连续块遮挡。

    values:  (T, N, F)
    masks:   (T, N, F)
    dts_raw: (T, N, F)
    exo:     (T, N, F_exo)

    返回：
        x_in:        (T, N, F)
        m_in:        (T, N, F)
        d_in:        (T, N, F)
        exo_in:      (T, N, F_exo)
        target_x:    (T, N, F)
        target_mask: (T, N, F) 仅被人工遮挡位置为1
    """
    T_local, N_local, F = values.shape
    x_in = values.copy().astype(np.float32)
    m_in = masks.copy().astype(np.float32)
    d_raw = dts_raw.copy().astype(np.float32)
    exo_in = exo.copy().astype(np.float32)
    target_x = values.copy().astype(np.float32)
    target_mask = np.zeros((T_local, N_local, F), dtype=np.float32)

    candidate_stations = []
    station_valid_all = []
    for s in range(N_local):
        valid_all = np.all(masks[:, s, :] > 0.5, axis=1)
        possible = False
        for L in block_lengths:
            if L <= T_local:
                for start in range(0, T_local - L + 1):
                    if np.all(valid_all[start:start + L]):
                        possible = True
                        break
            if possible:
                break
        if possible:
            candidate_stations.append(s)
            station_valid_all.append(valid_all)

    if len(candidate_stations) == 0:
        d_in = transform_dt(d_raw, use_log1p_dt).astype(np.float32)
        x_in[m_in < 0.5] = 0.0
        return x_in, m_in, d_in, exo_in, target_x, target_mask

    choice_idx = int(rng.integers(len(candidate_stations)))
    s = candidate_stations[choice_idx]
    valid_all = station_valid_all[choice_idx]
    blocks = choose_non_overlapping_blocks(valid_all, block_lengths, num_blocks, rng)

    for start, end in blocks:
        x_in[start:end, s, :] = 0.0
        m_in[start:end, s, :] = 0.0
        target_mask[start:end, s, :] = 1.0

    d_raw[:, s, :] = recompute_dt_from_mask(m_in[:, s, :])
    d_in = transform_dt(d_raw, use_log1p_dt).astype(np.float32)
    x_in[m_in < 0.5] = 0.0

    return x_in, m_in, d_in, exo_in, target_x, target_mask


class GRINExoTrainDataset(Dataset):
    def __init__(self, windows, block_lengths, num_blocks_per_sample, seed=42, use_log1p_dt=True):
        self.windows = windows
        self.block_lengths = list(block_lengths)
        self.num_blocks_per_sample = num_blocks_per_sample
        self.base_seed = int(seed)
        self.epoch = 0
        self.use_log1p_dt = use_log1p_dt

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        values, masks, dts_raw, exo = self.windows[idx]
        rng = np.random.default_rng(self.base_seed + self.epoch * 1000003 + idx)
        x_in, m_in, d_in, exo_in, target_x, target_mask = mask_single_station_graph_window(
            values, masks, dts_raw, exo, self.block_lengths, self.num_blocks_per_sample, rng, self.use_log1p_dt
        )
        return (
            torch.tensor(x_in, dtype=torch.float32),
            torch.tensor(m_in, dtype=torch.float32),
            torch.tensor(d_in, dtype=torch.float32),
            torch.tensor(exo_in, dtype=torch.float32),
            torch.tensor(target_x, dtype=torch.float32),
            torch.tensor(target_mask, dtype=torch.float32),
        )


class GRINExoEvalDataset(Dataset):
    def __init__(self, windows, block_length, num_blocks_per_sample=1, seed=123, use_log1p_dt=True):
        self.samples = []
        for idx, (values, masks, dts_raw, exo) in enumerate(windows):
            rng = np.random.default_rng(seed + idx)
            x_in, m_in, d_in, exo_in, target_x, target_mask = mask_single_station_graph_window(
                values, masks, dts_raw, exo, [block_length], num_blocks_per_sample, rng, use_log1p_dt
            )
            self.samples.append((x_in, m_in, d_in, exo_in, target_x, target_mask))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x_in, m_in, d_in, exo_in, target_x, target_mask = self.samples[idx]
        return (
            torch.tensor(x_in, dtype=torch.float32),
            torch.tensor(m_in, dtype=torch.float32),
            torch.tensor(d_in, dtype=torch.float32),
            torch.tensor(exo_in, dtype=torch.float32),
            torch.tensor(target_x, dtype=torch.float32),
            torch.tensor(target_mask, dtype=torch.float32),
        )


# ==================== GRIN-style 模型 ====================
class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, adj_norm):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.register_buffer('adj_norm', torch.tensor(adj_norm, dtype=torch.float32))

    def forward(self, x):
        # x: (B, N, F)
        ax = torch.einsum('ij,bjf->bif', self.adj_norm, x)
        return self.linear(ax)


class TemporalDecay(nn.Module):
    def __init__(self, input_dim, output_dim, diag=False):
        super().__init__()
        self.diag = diag
        self.W = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.b = nn.Parameter(torch.Tensor(output_dim))
        if diag:
            assert input_dim == output_dim
            mask = torch.eye(input_dim)
            self.register_buffer('diag_mask', mask)
        else:
            self.diag_mask = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.zeros_(self.b)

    def forward(self, d):
        # d: (..., F)
        W = self.W
        if self.diag:
            W = W * self.diag_mask
        gamma = torch.exp(-torch.relu(torch.matmul(d, W.t()) + self.b))
        return gamma


class FeatureRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.W = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.b = nn.Parameter(torch.Tensor(input_dim))
        mask = torch.ones(input_dim, input_dim) - torch.eye(input_dim)
        self.register_buffer('m', mask)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.zeros_(self.b)

    def forward(self, x):
        return torch.matmul(x, (self.W * self.m).t()) + self.b


class GraphRITSExo(nn.Module):
    def __init__(self, input_dim, exo_dim, hidden_size, adj_norm, exo_hidden=32, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.exo_dim = exo_dim
        self.hidden_size = hidden_size
        self.exo_hidden = exo_hidden

        self.temp_decay_h = TemporalDecay(input_dim, hidden_size, diag=False)
        self.temp_decay_x = TemporalDecay(input_dim, input_dim, diag=True)

        self.hist_graph = GraphConv(hidden_size, hidden_size, adj_norm)
        self.hist_reg = nn.Linear(hidden_size, input_dim)

        self.feat_reg = FeatureRegression(input_dim)
        self.weight_combine = nn.Linear(input_dim * 2, input_dim)

        self.input_graph = GraphConv(input_dim, input_dim, adj_norm)
        self.exo_proj = nn.Sequential(
            nn.Linear(exo_dim, exo_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)
        self.rnn_cell = nn.GRUCell(input_dim * 3 + exo_hidden, hidden_size)

    def forward(self, x, m, d, exo, target_x=None, target_mask=None):
        # x,m,d: (B, T, N, F); exo: (B, T, N, F_exo)
        B, T, N, F = x.size()
        h = torch.zeros(B, N, self.hidden_size, device=x.device)

        imputations = []
        loss = torch.tensor(0.0, device=x.device)

        for t in range(T):
            x_t = x[:, t, :, :]
            m_t = m[:, t, :, :]
            d_t = d[:, t, :, :]
            exo_t = exo[:, t, :, :]

            gamma_h = self.temp_decay_h(d_t.reshape(B * N, F)).view(B, N, self.hidden_size)
            gamma_x = self.temp_decay_x(d_t.reshape(B * N, F)).view(B, N, F)
            h = h * gamma_h

            h_graph = torch.relu(self.hist_graph(h))
            h_graph = self.dropout(h_graph)
            x_hist = self.hist_reg(h_graph)

            x_c = m_t * x_t + (1.0 - m_t) * x_hist

            z_h = self.feat_reg(x_c.reshape(B * N, F)).view(B, N, F)
            alpha = torch.sigmoid(
                self.weight_combine(torch.cat([gamma_x, m_t], dim=-1).reshape(B * N, 2 * F))
            ).view(B, N, F)
            c_h = alpha * z_h + (1.0 - alpha) * x_hist
            c_c = m_t * x_t + (1.0 - m_t) * c_h

            x_graph = torch.relu(self.input_graph(c_c))
            x_graph = self.dropout(x_graph)

            exo_graph = self.exo_proj(exo_t.reshape(B * N, self.exo_dim)).view(B, N, self.exo_hidden)

            gru_in = torch.cat([c_c, m_t, x_graph, exo_graph], dim=-1).reshape(B * N, 3 * F + self.exo_hidden)
            h = self.rnn_cell(gru_in, h.reshape(B * N, self.hidden_size)).view(B, N, self.hidden_size)
            imputations.append(c_c.unsqueeze(1))

            if target_x is not None and target_mask is not None:
                tgt_t = target_x[:, t, :, :]
                eval_t = target_mask[:, t, :, :]
                denom = eval_t.sum() + 1e-8
                loss = loss + (((x_hist - tgt_t) ** 2) * eval_t).sum() / denom
                loss = loss + (((z_h - tgt_t) ** 2) * eval_t).sum() / denom
                loss = loss + (((c_h - tgt_t) ** 2) * eval_t).sum() / denom

        imputations = torch.cat(imputations, dim=1)
        return imputations, loss / max(T, 1)


class GRINExo(nn.Module):
    def __init__(self, input_dim, exo_dim, hidden_size, adj_norm, exo_hidden=32, consistency_weight=0.1, dropout=0.1):
        super().__init__()
        self.fwd = GraphRITSExo(input_dim, exo_dim, hidden_size, adj_norm, exo_hidden, dropout)
        self.bwd = GraphRITSExo(input_dim, exo_dim, hidden_size, adj_norm, exo_hidden, dropout)
        self.consistency_weight = consistency_weight

    @staticmethod
    def reverse_time(x):
        idx = torch.arange(x.size(1) - 1, -1, -1, device=x.device)
        return x.index_select(1, idx)

    def forward(self, x, m, d, exo, target_x=None, target_mask=None):
        imp_f, loss_f = self.fwd(x, m, d, exo, target_x, target_mask)

        x_rev = self.reverse_time(x)
        m_rev = self.reverse_time(m)
        d_rev = self.reverse_time(d)
        exo_rev = self.reverse_time(exo)
        target_x_rev = self.reverse_time(target_x) if target_x is not None else None
        target_mask_rev = self.reverse_time(target_mask) if target_mask is not None else None

        imp_b_rev, loss_b = self.bwd(x_rev, m_rev, d_rev, exo_rev, target_x_rev, target_mask_rev)
        imp_b = self.reverse_time(imp_b_rev)

        imputations = 0.5 * (imp_f + imp_b)
        consistency = torch.mean(torch.abs(imp_f - imp_b))
        total_loss = loss_f + loss_b + self.consistency_weight * consistency
        return imputations, total_loss


# ==================== 数据加载与预处理 ====================
print("加载数据...")
full_data = np.load(DATA_PATH).astype(np.float32)
with open(FEATURE_NAMES_PATH, 'rb') as f:
    all_feature_names = pickle.load(f)

T, N, F_full = full_data.shape
print(f"完整数据形状: {full_data.shape}, 站点数: {N}, 特征数: {F_full}")
print(f"完整特征名: {all_feature_names}")

value_idxs, mask_idxs, dt_idxs = extract_target_triplets(all_feature_names, TARGET_NAMES, MASK_SUFFIX, DT_SUFFIX)
exo_idxs = extract_exogenous_indices(all_feature_names, EXOGENOUS_FEATURE_NAMES)
values_full = full_data[:, :, value_idxs]
masks_full = full_data[:, :, mask_idxs]
dts_full = full_data[:, :, dt_idxs]
exo_full = full_data[:, :, exo_idxs]

adj_raw, adj_norm = load_adjacency(ADJ_PATH, N)
print(f"邻接矩阵已加载，形状: {adj_raw.shape}, 非零边数: {int(np.sum(adj_raw))}")

# 时间切分
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
train_dts_raw = dts_full[train_slice]
val_dts_raw = dts_full[val_slice]
test_dts_raw = dts_full[test_slice]
train_exo_raw = exo_full[train_slice]
val_exo_raw = exo_full[val_slice]
test_exo_raw = exo_full[test_slice]

# 标准化
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
    'value_means': value_means,
    'value_stds': value_stds,
    'exo_means': exo_means,
    'exo_stds': exo_stds,
    'use_log1p_dt': USE_LOG1P_DT,
}
with open(os.path.join(OUTPUT_DIR, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler_params, f)
print("标准化参数已保存到 scaler.pkl")

train_windows = build_graph_windows(train_values_std, train_masks_raw, train_dts_raw, train_exo_std, SEQ_LEN, WINDOW_STRIDE)
val_windows = build_graph_windows(val_values_std, val_masks_raw, val_dts_raw, val_exo_std, SEQ_LEN, WINDOW_STRIDE)
test_windows = build_graph_windows(test_values_std, test_masks_raw, test_dts_raw, test_exo_std, SEQ_LEN, WINDOW_STRIDE)
print(f"训练窗口数: {len(train_windows)}, 验证窗口数: {len(val_windows)}, 测试窗口数: {len(test_windows)}")

train_dataset = GRINExoTrainDataset(
    train_windows,
    TRAIN_BLOCK_LENGTHS,
    TRAIN_NUM_BLOCKS_PER_SAMPLE,
    TRAIN_BLOCK_SEED,
    USE_LOG1P_DT,
)
val_dataset_dict = {
    L: GRINExoEvalDataset(val_windows, L, VAL_NUM_BLOCKS_PER_SAMPLE, VAL_BLOCK_SEED + i, USE_LOG1P_DT)
    for i, L in enumerate(VAL_BLOCK_LENGTHS)
}
test_dataset_dict = {
    L: GRINExoEvalDataset(test_windows, L, TEST_NUM_BLOCKS_PER_SAMPLE, TEST_BLOCK_SEED + i, USE_LOG1P_DT)
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
model = GRINExo(
    input_dim=len(TARGET_NAMES),
    exo_dim=len(EXOGENOUS_FEATURE_NAMES),
    hidden_size=HIDDEN_SIZE,
    adj_norm=adj_norm,
    exo_hidden=EXO_HIDDEN,
    consistency_weight=CONSISTENCY_WEIGHT,
    dropout=DROPOUT,
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

    for x_in, m_in, d_in, exo_in, target_x, target_mask in loader:
        x_in = x_in.to(device, non_blocking=True)
        m_in = m_in.to(device, non_blocking=True)
        d_in = d_in.to(device, non_blocking=True)
        exo_in = exo_in.to(device, non_blocking=True)
        target_x = target_x.to(device, non_blocking=True)
        target_mask = target_mask.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad()

        imputations, internal_loss = model(x_in, m_in, d_in, exo_in, target_x, target_mask)
        recon_loss = (((imputations - target_x) ** 2) * target_mask).sum() / (target_mask.sum() + 1e-8)
        loss = recon_loss + internal_loss

        if training:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(total_batches, 1)


print("\n开始训练 GRIN + exogenous baseline...")
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
        for x_in, m_in, d_in, exo_in, target_x, target_mask in loader:
            x_in = x_in.to(device, non_blocking=True)
            m_in = m_in.to(device, non_blocking=True)
            d_in = d_in.to(device, non_blocking=True)
            exo_in = exo_in.to(device, non_blocking=True)
            target_x = target_x.to(device, non_blocking=True)
            target_mask = target_mask.to(device, non_blocking=True)

            imputations, _ = model(x_in, m_in, d_in, exo_in, target_x, target_mask)
            all_pred.append(imputations.cpu().numpy())
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

print("====== GRIN + exogenous 人工大块遮挡 block-imputation 测试 ======")
for L in sorted(all_results.keys()):
    m = all_results[L]
    if m is None:
        print(f"[block={L}] 没有可评估样本")
    else:
        print(f"[block={L}] R2={m['R2']:.4f}, RMSE={m['RMSE']:.4f}, MAE={m['MAE']:.4f}, n_valid={m['n_valid']}")

with open(os.path.join(OUTPUT_DIR, 'Configs and results.txt'), 'w', encoding='utf-8') as f:
    f.write("========== GRIN Exogenous Block Imputation Configuration ==========\n")
    f.write(f"SEQ_LEN: {SEQ_LEN}\n")
    f.write(f"WINDOW_STRIDE: {WINDOW_STRIDE}\n")
    f.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
    f.write(f"HIDDEN_SIZE: {HIDDEN_SIZE}\n")
    f.write(f"EXO_HIDDEN: {EXO_HIDDEN}\n")
    f.write(f"DROPOUT: {DROPOUT}\n")
    f.write(f"LEARNING_RATE: {LEARNING_RATE}\n")
    f.write(f"PATIENCE: {PATIENCE}\n")
    f.write(f"MAX_EPOCHS: {MAX_EPOCHS}\n")
    f.write(f"TRAIN_RATIO: {TRAIN_RATIO}\n")
    f.write(f"VAL_RATIO: {VAL_RATIO}\n")
    f.write(f"TEST_RATIO: {TEST_RATIO}\n")
    f.write(f"SEED: {SEED}\n")
    f.write(f"CONSISTENCY_WEIGHT: {CONSISTENCY_WEIGHT}\n")
    f.write(f"USE_LOG1P_DT: {USE_LOG1P_DT}\n")
    f.write(f"NUM_WORKERS: {NUM_WORKERS}\n")
    f.write(f"PERSISTENT_WORKERS: {PERSISTENT_WORKERS}\n")
    f.write(f"DATA_PATH: {DATA_PATH}\n")
    f.write(f"FEATURE_NAMES_PATH: {FEATURE_NAMES_PATH}\n")
    f.write(f"ADJ_PATH: {ADJ_PATH}\n")
    f.write(f"OUTPUT_DIR: {OUTPUT_DIR}\n")
    f.write(f"TARGET_NAMES: {TARGET_NAMES}\n")
    f.write(f"EXOGENOUS_FEATURE_NAMES: {EXOGENOUS_FEATURE_NAMES}\n")
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




