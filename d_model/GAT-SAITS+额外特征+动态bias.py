import gc
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

CONFIG_PATH = "configs.yaml"
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config_all = yaml.safe_load(f)

cfg = config_all.get('SPATIAL_FIRST_STATIONWISE_GAT_SAITS_training', {})

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
ADJ_PATH = cfg.get('ADJ_PATH')
OUTPUT_DIR = cfg.get('OUTPUT_DIR')
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

GAT_HIDDEN = cfg.get('GAT_HIDDEN', 32)
GAT_HEADS = cfg.get('GAT_HEADS', 4)

block_cfg = cfg.get('block_masking', {})
TRAIN_BLOCK_LENGTHS = block_cfg.get('train_block_lengths', [3, 6, 12, 18, 42, 180])
TRAIN_NUM_BLOCKS_PER_SAMPLE = block_cfg.get('train_num_blocks_per_sample', 1)
TRAIN_BLOCK_SEED = block_cfg.get('train_block_seed', 42)
VAL_BLOCK_LENGTHS = block_cfg.get('val_block_lengths', [3, 6, 12, 18, 42, 180])
VAL_NUM_BLOCKS_PER_SAMPLE = block_cfg.get('val_num_blocks_per_sample', 1)
VAL_BLOCK_SEED = block_cfg.get('val_block_seed', 202)

# evaluation groups
EVAL_CFG = cfg.get('evaluation_protocols', {})
RANDOM_MISSING_RATIOS = EVAL_CFG.get('random_missing_ratios', [0.1, 0.3, 0.5, 0.7])
RANDOM_MISSING_SEED = EVAL_CFG.get('random_missing_seed', 2026)
SINGLE_STATION_BLOCK_LENGTHS = EVAL_CFG.get('single_station_block_lengths', [3, 6, 12, 18, 42, 180])
SINGLE_STATION_NUM_BLOCKS_PER_SAMPLE = EVAL_CFG.get('single_station_num_blocks_per_sample', 1)
SINGLE_STATION_BLOCK_SEED = EVAL_CFG.get('single_station_block_seed', 3031)
SYNC_ALL_STATION_BLOCK_LENGTHS = EVAL_CFG.get('sync_all_station_block_lengths', [3, 6, 12, 18, 42, 180])
SYNC_ALL_STATION_NUM_BLOCKS_PER_SAMPLE = EVAL_CFG.get('sync_all_station_num_blocks_per_sample', 1)
SYNC_ALL_STATION_BLOCK_SEED = EVAL_CFG.get('sync_all_station_block_seed', 4041)

if OUTPUT_DIR is None:
    raise ValueError('OUTPUT_DIR 不能为空')

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


def extract_target_triplets(all_feature_names, target_names, mask_suffix='_mask', dt_suffix='_dt'):
    value_indices, mask_indices, dt_indices = [], [], []
    for name in target_names:
        if name not in all_feature_names:
            raise ValueError(f'找不到目标变量: {name}')
        mask_name = name + mask_suffix
        dt_name = name + dt_suffix
        if mask_name not in all_feature_names:
            raise ValueError(f'找不到掩码列: {mask_name}')
        if dt_name not in all_feature_names:
            raise ValueError(f'找不到 dt 列: {dt_name}')
        value_indices.append(all_feature_names.index(name))
        mask_indices.append(all_feature_names.index(mask_name))
        dt_indices.append(all_feature_names.index(dt_name))
    return value_indices, mask_indices, dt_indices


def extract_feature_indices(all_feature_names, feature_names, feature_group_name='features'):
    if not feature_names:
        raise ValueError(f'{feature_group_name} 不能为空')
    idxs, missing = [], []
    for name in feature_names:
        if name not in all_feature_names:
            missing.append(name)
        else:
            idxs.append(all_feature_names.index(name))
    if missing:
        raise ValueError(f'以下 {feature_group_name} 不存在于特征中: {missing}')
    return idxs


def fit_target_scaler(values, masks):
    f_dim = values.shape[-1]
    means = np.zeros(f_dim, dtype=np.float32)
    stds = np.ones(f_dim, dtype=np.float32)
    for f in range(f_dim):
        valid = (masks[..., f] > 0.5) & (~np.isnan(values[..., f]))
        vals = values[..., f][valid]
        if vals.size == 0:
            raise ValueError(f'目标第 {f} 维在训练集中没有有效观测，无法标准化')
        means[f] = np.mean(vals)
        std = np.std(vals)
        stds[f] = std if std > 0 else 1.0
    return means, stds


def apply_value_standardization(values, masks, means, stds):
    out = values.astype(np.float32, copy=True)
    for f in range(values.shape[-1]):
        valid = (masks[..., f] > 0.5) & (~np.isnan(out[..., f]))
        out[..., f][valid] = (out[..., f][valid] - means[f]) / stds[f]
    return out


def fit_exo_scaler(exo):
    f_dim = exo.shape[-1]
    means = np.zeros(f_dim, dtype=np.float32)
    stds = np.ones(f_dim, dtype=np.float32)
    for f in range(f_dim):
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
    out = exo.astype(np.float32, copy=True)
    for f in range(exo.shape[-1]):
        valid = ~np.isnan(out[..., f])
        out[..., f][valid] = (out[..., f][valid] - means[f]) / stds[f]
    return out


def choose_non_overlapping_blocks(valid_all, candidate_lengths, num_blocks, rng):
    t_local = len(valid_all)
    occupied = np.zeros(t_local, dtype=bool)
    blocks = []
    lengths = list(candidate_lengths)
    for _ in range(num_blocks):
        rng.shuffle(lengths)
        chosen = False
        for l_val in lengths:
            if l_val > t_local:
                continue
            starts = []
            for start in range(0, t_local - l_val + 1):
                end = start + l_val
                if occupied[start:end].any():
                    continue
                if np.all(valid_all[start:end]):
                    starts.append(start)
            if starts:
                start = int(rng.choice(starts))
                end = start + l_val
                blocks.append((start, end))
                occupied[start:end] = True
                chosen = True
                break
        if not chosen:
            break
    return blocks


def load_adjacency(adj_path, num_nodes):
    ext = os.path.splitext(adj_path)[1].lower()
    if ext == '.npy':
        adj = np.load(adj_path)
    elif ext == '.csv':
        adj = np.loadtxt(adj_path, delimiter=',', dtype=np.float32)
    else:
        raise ValueError(f'不支持的邻接矩阵格式: {ext}')
    if adj.ndim != 2:
        raise ValueError(f'邻接矩阵必须是二维数组，实际 ndim={adj.ndim}')
    if adj.shape != (num_nodes, num_nodes):
        raise ValueError(f'邻接矩阵形状应为 ({num_nodes}, {num_nodes})，实际为 {adj.shape}')
    adj = (adj != 0).astype(np.float32)
    np.fill_diagonal(adj, 1.0)
    return adj.astype(np.float32)


def build_stationwise_sample_specs(num_timesteps, num_stations, seq_len, stride):
    specs = []
    for station_idx in range(num_stations):
        for start in range(0, num_timesteps - seq_len + 1, stride):
            specs.append((int(start), int(station_idx)))
    return specs


def slice_window(arr, start, seq_len):
    end = start + seq_len
    return arr[start:end]


def make_single_station_block_sample(values, masks, exo, time_feats, target_station, block_lengths, num_blocks, rng):
    t_local, n_local, f_dim = values.shape
    x_graph = values.astype(np.float32, copy=True)
    m_graph = masks.astype(np.float32, copy=True)
    target_x = values[:, target_station, :].astype(np.float32, copy=True)
    target_mask = np.zeros((t_local, f_dim), dtype=np.float32)

    valid_all = np.all(masks[:, target_station, :] > 0.5, axis=1)
    blocks = choose_non_overlapping_blocks(valid_all, block_lengths, num_blocks, rng)
    for start, end in blocks:
        x_graph[start:end, target_station, :] = 0.0
        m_graph[start:end, target_station, :] = 0.0
        target_mask[start:end, :] = 1.0

    x_graph[m_graph < 0.5] = 0.0
    return (
        x_graph,
        m_graph,
        exo.astype(np.float32, copy=False),
        time_feats.astype(np.float32, copy=False),
        target_x,
        target_mask.astype(np.float32, copy=False),
        int(target_station),
    )


def make_sync_all_station_block_sample(values, masks, exo, time_feats, target_station, block_lengths, num_blocks, rng):
    t_local, n_local, f_dim = values.shape
    x_graph = values.astype(np.float32, copy=True)
    m_graph = masks.astype(np.float32, copy=True)
    target_x = values[:, target_station, :].astype(np.float32, copy=True)
    target_mask = np.zeros((t_local, f_dim), dtype=np.float32)

    valid_all = np.all(masks[:, target_station, :] > 0.5, axis=1)
    blocks = choose_non_overlapping_blocks(valid_all, block_lengths, num_blocks, rng)
    for start, end in blocks:
        x_graph[start:end, :, :] = 0.0
        m_graph[start:end, :, :] = 0.0
        target_mask[start:end, :] = 1.0

    x_graph[m_graph < 0.5] = 0.0
    return (
        x_graph,
        m_graph,
        exo.astype(np.float32, copy=False),
        time_feats.astype(np.float32, copy=False),
        target_x,
        target_mask.astype(np.float32, copy=False),
        int(target_station),
    )


def make_random_missing_sample(values, masks, exo, time_feats, target_station, missing_ratio, rng):
    t_local, n_local, f_dim = values.shape
    x_graph = values.astype(np.float32, copy=True)
    m_graph = masks.astype(np.float32, copy=True)
    target_x = values[:, target_station, :].astype(np.float32, copy=True)
    target_mask = np.zeros((t_local, f_dim), dtype=np.float32)

    for f in range(f_dim):
        valid_idx = np.where(masks[:, target_station, f] > 0.5)[0]
        if len(valid_idx) == 0:
            continue
        k = int(np.floor(len(valid_idx) * float(missing_ratio)))
        if k <= 0:
            continue
        chosen = rng.choice(valid_idx, size=k, replace=False)
        x_graph[chosen, target_station, f] = 0.0
        m_graph[chosen, target_station, f] = 0.0
        target_mask[chosen, f] = 1.0

    x_graph[m_graph < 0.5] = 0.0
    return (
        x_graph,
        m_graph,
        exo.astype(np.float32, copy=False),
        time_feats.astype(np.float32, copy=False),
        target_x,
        target_mask.astype(np.float32, copy=False),
        int(target_station),
    )


class BaseStationwiseWindowDataset(Dataset):
    def __init__(self, values, masks, exo, time_feats, seq_len, sample_specs):
        self.values = values
        self.masks = masks
        self.exo = exo
        self.time_feats = time_feats
        self.seq_len = int(seq_len)
        self.sample_specs = sample_specs

    def __len__(self):
        return len(self.sample_specs)

    def _get_window(self, idx):
        start, target_station = self.sample_specs[idx]
        values = slice_window(self.values, start, self.seq_len)
        masks = slice_window(self.masks, start, self.seq_len)
        exo = slice_window(self.exo, start, self.seq_len)
        time_feats = slice_window(self.time_feats, start, self.seq_len)
        return values, masks, exo, time_feats, target_station


class SpatialFirstStationwiseTrainDataset(BaseStationwiseWindowDataset):
    def __init__(self, values, masks, exo, time_feats, seq_len, sample_specs, block_lengths, num_blocks_per_sample, seed=42):
        super().__init__(values, masks, exo, time_feats, seq_len, sample_specs)
        self.block_lengths = list(block_lengths)
        self.num_blocks_per_sample = int(num_blocks_per_sample)
        self.base_seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, idx):
        values, masks, exo, time_feats, target_station = self._get_window(idx)
        rng = np.random.default_rng(self.base_seed + self.epoch * 1000003 + idx)
        x_graph, m_graph, exo_in, time_feats_in, target_x, target_mask, target_station = make_single_station_block_sample(
            values, masks, exo, time_feats, target_station, self.block_lengths, self.num_blocks_per_sample, rng
        )
        return (
            torch.from_numpy(x_graph),
            torch.from_numpy(m_graph),
            torch.from_numpy(exo_in),
            torch.from_numpy(time_feats_in),
            torch.tensor(target_station, dtype=torch.long),
            torch.from_numpy(target_x),
            torch.from_numpy(target_mask),
        )


class SpatialFirstStationwiseBlockEvalDataset(BaseStationwiseWindowDataset):
    def __init__(self, values, masks, exo, time_feats, seq_len, sample_specs, block_length, num_blocks_per_sample=1, seed=123, mode='single_station'):
        super().__init__(values, masks, exo, time_feats, seq_len, sample_specs)
        self.block_length = int(block_length)
        self.num_blocks_per_sample = int(num_blocks_per_sample)
        self.seed = int(seed)
        self.mode = mode

    def __getitem__(self, idx):
        values, masks, exo, time_feats, target_station = self._get_window(idx)
        rng = np.random.default_rng(self.seed + idx)
        if self.mode == 'single_station':
            sample = make_single_station_block_sample(values, masks, exo, time_feats, target_station, [self.block_length], self.num_blocks_per_sample, rng)
        elif self.mode == 'sync_all_station':
            sample = make_sync_all_station_block_sample(values, masks, exo, time_feats, target_station, [self.block_length], self.num_blocks_per_sample, rng)
        else:
            raise ValueError(f'未知 block eval mode: {self.mode}')
        x_graph, m_graph, exo_in, time_feats_in, target_x, target_mask, target_station = sample
        return (
            torch.from_numpy(x_graph),
            torch.from_numpy(m_graph),
            torch.from_numpy(exo_in),
            torch.from_numpy(time_feats_in),
            torch.tensor(target_station, dtype=torch.long),
            torch.from_numpy(target_x),
            torch.from_numpy(target_mask),
        )


class SpatialFirstStationwiseRandomEvalDataset(BaseStationwiseWindowDataset):
    def __init__(self, values, masks, exo, time_feats, seq_len, sample_specs, missing_ratio, seed=2026):
        super().__init__(values, masks, exo, time_feats, seq_len, sample_specs)
        self.missing_ratio = float(missing_ratio)
        self.seed = int(seed)

    def __getitem__(self, idx):
        values, masks, exo, time_feats, target_station = self._get_window(idx)
        rng = np.random.default_rng(self.seed + idx)
        x_graph, m_graph, exo_in, time_feats_in, target_x, target_mask, target_station = make_random_missing_sample(
            values, masks, exo, time_feats, target_station, self.missing_ratio, rng
        )
        return (
            torch.from_numpy(x_graph),
            torch.from_numpy(m_graph),
            torch.from_numpy(exo_in),
            torch.from_numpy(time_feats_in),
            torch.tensor(target_station, dtype=torch.long),
            torch.from_numpy(target_x),
            torch.from_numpy(target_mask),
        )


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
        t_local = x.size(1)
        diag_mask = torch.eye(t_local, dtype=torch.bool, device=x.device)
        x = self.pos_encoder(x)
        x = self.encoder(x, mask=diag_mask)
        return x


class MultiHeadGraphAttention(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads=4, dropout=0.1, negative_slope=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.proj = nn.Linear(in_dim, hidden_dim * num_heads, bias=False)
        self.attn_src = nn.Parameter(torch.empty(num_heads, hidden_dim))
        self.attn_dst = nn.Parameter(torch.empty(num_heads, hidden_dim))
        self.out_proj = nn.Linear(hidden_dim * num_heads, hidden_dim)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, x, adj_mask):
        bsz, n_nodes, _ = x.shape
        h = self.proj(x).view(bsz, n_nodes, self.num_heads, self.hidden_dim).permute(0, 2, 1, 3)
        src = (h * self.attn_src[None, :, None, :]).sum(dim=-1)
        dst = (h * self.attn_dst[None, :, None, :]).sum(dim=-1)
        e = self.leaky_relu(src.unsqueeze(-1) + dst.unsqueeze(-2))
        if adj_mask.dim() == 2:
            mask = adj_mask[None, None, :, :].expand(bsz, self.num_heads, n_nodes, n_nodes)
        else:
            mask = adj_mask[:, None, :, :].expand(bsz, self.num_heads, n_nodes, n_nodes)
        e = e.masked_fill(mask < 0.5, -1e9)
        alpha = torch.softmax(e, dim=-1)
        alpha = self.dropout(alpha)
        out = torch.matmul(alpha, h)
        out = out.permute(0, 2, 1, 3).reshape(bsz, n_nodes, self.num_heads * self.hidden_dim)
        return self.out_proj(out)


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
                gate_ctx = torch.cat(gate_parts, dim=-1)
                gate = self.fusion_gate(gate_ctx)
                time_bias = gate * time_bias
            bias = bias + time_bias
        bias_reg = (bias ** 2).mean()
        return bias, bias_reg


class SpatialFirstStationwiseGATSAITS(nn.Module):
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
        gat_hidden=32,
        gat_heads=4,
        bias_hidden=64,
        bias_dropout=0.1,
        bias_init_scale=1.0,
        use_variable_bias=True,
        use_station_bias=True,
        use_time_bias=True,
        use_time_gate=True,
        static_adj=None,
    ):
        super().__init__()
        if static_adj is None:
            raise ValueError('static_adj 不能为空')
        self.input_dim = input_dim
        self.exo_proj = nn.Sequential(
            nn.Linear(exo_dim, exo_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        graph_in_dim = input_dim * 2 + exo_hidden
        self.gat = MultiHeadGraphAttention(graph_in_dim, gat_hidden, num_heads=gat_heads, dropout=dropout)
        self.graph_dropout = nn.Dropout(dropout)

        total_in_dim = input_dim * 2 + exo_hidden + gat_hidden
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
        self.register_buffer('static_adj', torch.tensor(static_adj, dtype=torch.float32))

    def _spatial_encode(self, x_graph, m_graph, exo_graph_h):
        bsz, t_local, n_local, _ = x_graph.shape
        outs = []
        for t in range(t_local):
            node_in = torch.cat([x_graph[:, t], m_graph[:, t], exo_graph_h[:, t]], dim=-1)
            h_t = self.gat(node_in, self.static_adj)
            h_t = torch.relu(h_t)
            h_t = self.graph_dropout(h_t)
            outs.append(h_t.unsqueeze(1))
        return torch.cat(outs, dim=1)

    def _gather_target_station(self, tensor, target_station):
        bsz, t_local, _, d_dim = tensor.shape
        idx = target_station.view(bsz, 1, 1, 1).expand(-1, t_local, 1, d_dim)
        return torch.gather(tensor, dim=2, index=idx).squeeze(2)

    def forward(self, x_graph, m_graph, exo_graph, time_graph, target_station):
        exo_graph_h = self.exo_proj(exo_graph)
        spatial_graph_h = self._spatial_encode(x_graph, m_graph, exo_graph_h)

        x_tgt = self._gather_target_station(x_graph, target_station)
        m_tgt = self._gather_target_station(m_graph, target_station)
        exo_tgt_h = self._gather_target_station(exo_graph_h, target_station)
        spatial_tgt_h = self._gather_target_station(spatial_graph_h, target_station)
        time_tgt = self._gather_target_station(time_graph, target_station)

        inp1 = torch.cat([x_tgt, m_tgt, exo_tgt_h, spatial_tgt_h], dim=-1)
        h1 = self.encoder_1(self.input_proj_1(inp1))
        x_tilde_1 = self.output_proj_1(h1)
        x_hat_1 = m_tgt * x_tgt + (1.0 - m_tgt) * x_tilde_1

        inp2 = torch.cat([x_hat_1, m_tgt, exo_tgt_h, spatial_tgt_h], dim=-1)
        h2 = self.encoder_2(self.input_proj_2(inp2))
        x_tilde_2 = self.output_proj_2(h2)
        x_hat_2 = m_tgt * x_tgt + (1.0 - m_tgt) * x_tilde_2

        gate = self.combine_gate(torch.cat([x_tilde_1, x_tilde_2], dim=-1))
        x_comb = gate * x_tilde_1 + (1.0 - gate) * x_tilde_2

        bias_term, bias_reg = self.bias_branch(target_station, time_tgt)
        x_bias = x_comb + self.bias_scale * bias_term
        x_final = m_tgt * x_tgt + (1.0 - m_tgt) * x_bias

        return {
            'imputation_1': x_hat_1,
            'imputation_2': x_hat_2,
            'imputation_no_bias': m_tgt * x_tgt + (1.0 - m_tgt) * x_comb,
            'bias_term': bias_term,
            'imputation_final': x_final,
            'bias_reg': bias_reg,
        }


def spatial_first_stationwise_loss(outputs, target_x, target_mask, bias_reg_weight=1e-4):
    denom = target_mask.sum() + 1e-8
    l1 = (((outputs['imputation_1'] - target_x) ** 2) * target_mask).sum() / denom
    l2 = (((outputs['imputation_2'] - target_x) ** 2) * target_mask).sum() / denom
    lf = (((outputs['imputation_final'] - target_x) ** 2) * target_mask).sum() / denom
    return 0.5 * l1 + 0.5 * l2 + lf + bias_reg_weight * outputs['bias_reg']


print('加载数据...')
full_data = np.load(DATA_PATH).astype(np.float32)
with open(FEATURE_NAMES_PATH, 'rb') as f:
    all_feature_names = pickle.load(f)

total_t, n_stations, f_full = full_data.shape
print(f'完整数据形状: {full_data.shape}, 站点数: {n_stations}, 特征数: {f_full}')

value_idxs, mask_idxs, _ = extract_target_triplets(all_feature_names, TARGET_NAMES, MASK_SUFFIX, DT_SUFFIX)
exo_idxs = extract_feature_indices(all_feature_names, EXOGENOUS_FEATURE_NAMES, 'exogenous_feature_names')
time_feature_idxs = extract_feature_indices(all_feature_names, TIME_FEATURE_NAMES, 'time_feature_names')

values_full = full_data[:, :, value_idxs]
masks_full = full_data[:, :, mask_idxs]
exo_full = full_data[:, :, exo_idxs]
time_feats_full = full_data[:, :, time_feature_idxs]
time_feats_full = np.nan_to_num(time_feats_full, nan=0.0).astype(np.float32, copy=False)

adj_raw = load_adjacency(ADJ_PATH, n_stations)
print(f'目标变量: {TARGET_NAMES}')
print(f'额外特征数: {len(EXOGENOUS_FEATURE_NAMES)}')
print(f'时间 bias 特征: {TIME_FEATURE_NAMES}')
print(f'邻接矩阵路径: {ADJ_PATH}')

train_len = int(total_t * TRAIN_RATIO)
val_len = int(total_t * VAL_RATIO)

test_len = total_t - train_len - val_len
train_slice = slice(0, train_len)
val_slice = slice(train_len, train_len + val_len)
test_slice = slice(train_len + val_len, total_t)

train_values_raw = values_full[train_slice]
val_values_raw = values_full[val_slice]
test_values_raw = values_full[test_slice]
train_masks_raw = masks_full[train_slice].astype(np.float32, copy=False)
val_masks_raw = masks_full[val_slice].astype(np.float32, copy=False)
test_masks_raw = masks_full[test_slice].astype(np.float32, copy=False)
train_exo_raw = exo_full[train_slice]
val_exo_raw = exo_full[val_slice]
test_exo_raw = exo_full[test_slice]
train_time_feats = time_feats_full[train_slice].astype(np.float32, copy=False)
val_time_feats = time_feats_full[val_slice].astype(np.float32, copy=False)
test_time_feats = time_feats_full[test_slice].astype(np.float32, copy=False)

value_means, value_stds = fit_target_scaler(train_values_raw, train_masks_raw)
train_values_std = apply_value_standardization(train_values_raw, train_masks_raw, value_means, value_stds)
val_values_std = apply_value_standardization(val_values_raw, val_masks_raw, value_means, value_stds)
test_values_std = apply_value_standardization(test_values_raw, test_masks_raw, value_means, value_stds)

exo_means, exo_stds = fit_exo_scaler(train_exo_raw)
train_exo_std = apply_exo_standardization(train_exo_raw, exo_means, exo_stds)
val_exo_std = apply_exo_standardization(val_exo_raw, exo_means, exo_stds)
test_exo_std = apply_exo_standardization(test_exo_raw, exo_means, exo_stds)

train_values_std = np.nan_to_num(train_values_std, nan=0.0).astype(np.float32, copy=False)
val_values_std = np.nan_to_num(val_values_std, nan=0.0).astype(np.float32, copy=False)
test_values_std = np.nan_to_num(test_values_std, nan=0.0).astype(np.float32, copy=False)
train_exo_std = np.nan_to_num(train_exo_std, nan=0.0).astype(np.float32, copy=False)
val_exo_std = np.nan_to_num(val_exo_std, nan=0.0).astype(np.float32, copy=False)
test_exo_std = np.nan_to_num(test_exo_std, nan=0.0).astype(np.float32, copy=False)

scaler_params = {
    'target_names': TARGET_NAMES,
    'exogenous_feature_names': EXOGENOUS_FEATURE_NAMES,
    'time_feature_names': TIME_FEATURE_NAMES,
    'value_means': value_means,
    'value_stds': value_stds,
    'exo_means': exo_means,
    'exo_stds': exo_stds,
}
with open(os.path.join(OUTPUT_DIR, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler_params, f)
print('标准化参数已保存到 scaler.pkl')

del full_data, values_full, masks_full, exo_full, time_feats_full
del train_values_raw, val_values_raw, test_values_raw
del train_exo_raw, val_exo_raw, test_exo_raw
gc.collect()

train_sample_specs = build_stationwise_sample_specs(train_values_std.shape[0], n_stations, SEQ_LEN, WINDOW_STRIDE)
val_sample_specs = build_stationwise_sample_specs(val_values_std.shape[0], n_stations, SEQ_LEN, WINDOW_STRIDE)
test_sample_specs = build_stationwise_sample_specs(test_values_std.shape[0], n_stations, SEQ_LEN, WINDOW_STRIDE)
print(f'训练样本数: {len(train_sample_specs)}, 验证样本数: {len(val_sample_specs)}, 测试样本数: {len(test_sample_specs)}')

train_dataset = SpatialFirstStationwiseTrainDataset(
    train_values_std, train_masks_raw, train_exo_std, train_time_feats, SEQ_LEN,
    train_sample_specs, TRAIN_BLOCK_LENGTHS, TRAIN_NUM_BLOCKS_PER_SAMPLE, TRAIN_BLOCK_SEED
)
val_dataset_dict = {
    l_val: SpatialFirstStationwiseBlockEvalDataset(
        val_values_std, val_masks_raw, val_exo_std, val_time_feats, SEQ_LEN,
        val_sample_specs, l_val, VAL_NUM_BLOCKS_PER_SAMPLE, VAL_BLOCK_SEED + i, mode='single_station'
    )
    for i, l_val in enumerate(VAL_BLOCK_LENGTHS)
}

random_test_dataset_dict = {
    ratio: SpatialFirstStationwiseRandomEvalDataset(
        test_values_std, test_masks_raw, test_exo_std, test_time_feats, SEQ_LEN,
        test_sample_specs, ratio, RANDOM_MISSING_SEED + i
    )
    for i, ratio in enumerate(RANDOM_MISSING_RATIOS)
}
single_station_test_dataset_dict = {
    l_val: SpatialFirstStationwiseBlockEvalDataset(
        test_values_std, test_masks_raw, test_exo_std, test_time_feats, SEQ_LEN,
        test_sample_specs, l_val, SINGLE_STATION_NUM_BLOCKS_PER_SAMPLE,
        SINGLE_STATION_BLOCK_SEED + i, mode='single_station'
    )
    for i, l_val in enumerate(SINGLE_STATION_BLOCK_LENGTHS)
}
sync_all_station_test_dataset_dict = {
    l_val: SpatialFirstStationwiseBlockEvalDataset(
        test_values_std, test_masks_raw, test_exo_std, test_time_feats, SEQ_LEN,
        test_sample_specs, l_val, SYNC_ALL_STATION_NUM_BLOCKS_PER_SAMPLE,
        SYNC_ALL_STATION_BLOCK_SEED + i, mode='sync_all_station'
    )
    for i, l_val in enumerate(SYNC_ALL_STATION_BLOCK_LENGTHS)
}

test_dataset_groups = {
    'random_missing': random_test_dataset_dict,
    'single_station_block_missing': single_station_test_dataset_dict,
    'sync_all_station_block_missing': sync_all_station_test_dataset_dict,
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
    l_val: DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin_memory,
        num_workers=NUM_WORKERS, persistent_workers=use_persistent
    )
    for l_val, ds in val_dataset_dict.items()
}
test_loader_groups = {
    group_name: {
        key: DataLoader(
            ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin_memory,
            num_workers=NUM_WORKERS, persistent_workers=use_persistent
        )
        for key, ds in ds_dict.items()
    }
    for group_name, ds_dict in test_dataset_groups.items()
}

model = SpatialFirstStationwiseGATSAITS(
    input_dim=len(TARGET_NAMES),
    exo_dim=len(EXOGENOUS_FEATURE_NAMES),
    num_stations=n_stations,
    time_feat_dim=len(TIME_FEATURE_NAMES),
    d_model=D_MODEL,
    n_head=N_HEAD,
    n_layers=N_LAYERS,
    d_ff=D_FF,
    dropout=DROPOUT,
    seq_len=SEQ_LEN,
    exo_hidden=EXO_HIDDEN,
    gat_hidden=GAT_HIDDEN,
    gat_heads=GAT_HEADS,
    bias_hidden=BIAS_HIDDEN,
    bias_dropout=BIAS_DROPOUT,
    bias_init_scale=BIAS_INIT_SCALE,
    use_variable_bias=USE_VARIABLE_BIAS,
    use_station_bias=USE_STATION_BIAS,
    use_time_bias=USE_TIME_BIAS,
    use_time_gate=USE_TIME_GATE,
    static_adj=adj_raw,
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
    for x_graph, m_graph, exo_in, time_feats, target_station, target_x, target_mask in loader:
        x_graph = x_graph.to(device, non_blocking=True)
        m_graph = m_graph.to(device, non_blocking=True)
        exo_in = exo_in.to(device, non_blocking=True)
        time_feats = time_feats.to(device, non_blocking=True)
        target_station = target_station.to(device, non_blocking=True)
        target_x = target_x.to(device, non_blocking=True)
        target_mask = target_mask.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad()
        outputs = model(x_graph, m_graph, exo_in, time_feats, target_station)
        loss = spatial_first_stationwise_loss(outputs, target_x, target_mask, bias_reg_weight=BIAS_REG_WEIGHT)
        if training:
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        total_batches += 1
    return total_loss / max(total_batches, 1)


print('\n开始训练 Spatial-first Station-wise GAT-SAITS ...')
for epoch in range(MAX_EPOCHS):
    train_loss = run_epoch(train_loader, training=True, epoch=epoch)
    val_losses = [run_epoch(loader, training=False) for _, loader in val_loader_dict.items()]
    val_loss = float(np.mean(val_losses)) if val_losses else np.nan
    scheduler.step(val_loss)
    print(f'Epoch {epoch + 1}/{MAX_EPOCHS} - Train Loss: {train_loss:.6f}, Val Mean Loss: {val_loss:.6f}')
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model.pth'))
        counter = 0
    else:
        counter += 1
        if counter >= PATIENCE:
            print(f'Early stopping at epoch {epoch + 1}')
            break


def _compute_basic_metrics(pred_valid, target_valid):
    return {
        'R2': float(r2_score(target_valid, pred_valid)),
        'RMSE': float(np.sqrt(mean_squared_error(target_valid, pred_valid))),
        'MAE': float(mean_absolute_error(target_valid, pred_valid)),
        'n_valid': int(len(pred_valid)),
    }


def evaluate_loader_with_breakdown(loader):
    model.eval()
    all_pred, all_target, all_evalmask, all_bias, all_station = [], [], [], [], []
    with torch.no_grad():
        for x_graph, m_graph, exo_in, time_feats, target_station, target_x, target_mask in loader:
            x_graph = x_graph.to(device, non_blocking=True)
            m_graph = m_graph.to(device, non_blocking=True)
            exo_in = exo_in.to(device, non_blocking=True)
            time_feats = time_feats.to(device, non_blocking=True)
            target_station = target_station.to(device, non_blocking=True)
            target_x = target_x.to(device, non_blocking=True)
            target_mask = target_mask.to(device, non_blocking=True)
            outputs = model(x_graph, m_graph, exo_in, time_feats, target_station)
            pred = outputs['imputation_final'].cpu().numpy()
            target = target_x.cpu().numpy()
            evalmask = target_mask.cpu().numpy()
            bias = outputs['bias_term'].cpu().numpy()
            station = target_station.cpu().numpy()
            all_pred.append(pred)
            all_target.append(target)
            all_evalmask.append(evalmask)
            all_bias.append(bias)
            all_station.append(station)

    pred = np.concatenate(all_pred, axis=0)
    target = np.concatenate(all_target, axis=0)
    evalmask = np.concatenate(all_evalmask, axis=0)
    bias = np.concatenate(all_bias, axis=0)
    station_ids = np.concatenate(all_station, axis=0)

    valid = evalmask > 0.5
    if np.sum(valid) == 0:
        return None, pred, target, evalmask, bias, {}, {}

    pred_valid = pred[valid]
    target_valid = target[valid]
    overall_metrics = _compute_basic_metrics(pred_valid, target_valid)

    per_variable_metrics = {}
    for f in range(len(TARGET_NAMES)):
        valid_f = valid[:, :, f]
        if np.sum(valid_f) == 0:
            continue
        pred_f = pred[:, :, f][valid_f]
        target_f = target[:, :, f][valid_f]
        per_variable_metrics[TARGET_NAMES[f]] = _compute_basic_metrics(pred_f, target_f)

    per_station_metrics = {}
    for s in range(n_stations):
        mask_s = valid & (station_ids[:, None] == s)
        if np.sum(mask_s) == 0:
            continue
        pred_s = pred[mask_s]
        target_s = target[mask_s]
        per_station_metrics[f'station_{s}'] = _compute_basic_metrics(pred_s, target_s)

    return overall_metrics, pred, target, evalmask, bias, per_variable_metrics, per_station_metrics


print('\n加载最佳模型进行三组测试...')
model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_model.pth'), map_location=device))
model.eval()

all_eval_results = {}
for group_name, loader_dict in test_loader_groups.items():
    print(f'\n==================== {group_name} ====================')
    all_eval_results[group_name] = {}
    for key, loader in loader_dict.items():
        overall_metrics, pred, target, evalmask, bias, per_var, per_station = evaluate_loader_with_breakdown(loader)
        all_eval_results[group_name][str(key)] = {
            'overall': overall_metrics,
            'per_variable': per_var,
            'per_station': per_station,
        }
        prefix = f'{group_name}_{key}'
        np.save(os.path.join(OUTPUT_DIR, f'{prefix}_pred.npy'), pred)
        np.save(os.path.join(OUTPUT_DIR, f'{prefix}_target.npy'), target)
        np.save(os.path.join(OUTPUT_DIR, f'{prefix}_evalmask.npy'), evalmask)
        np.save(os.path.join(OUTPUT_DIR, f'{prefix}_bias.npy'), bias)
        if overall_metrics is None:
            print(f'[{key}] 没有可评估样本')
        else:
            print(f'[{key}] R2={overall_metrics["R2"]:.4f}, RMSE={overall_metrics["RMSE"]:.4f}, MAE={overall_metrics["MAE"]:.4f}, n_valid={overall_metrics["n_valid"]}')
        del pred, target, evalmask, bias
        gc.collect()

with open(os.path.join(OUTPUT_DIR, 'Configs and results.txt'), 'w', encoding='utf-8') as f:
    f.write('========== Spatial-first Station-wise GAT-SAITS Configuration ==========\n')
    f.write(f'SEQ_LEN: {SEQ_LEN}\n')
    f.write(f'WINDOW_STRIDE: {WINDOW_STRIDE}\n')
    f.write(f'BATCH_SIZE: {BATCH_SIZE}\n')
    f.write(f'D_MODEL: {D_MODEL}\n')
    f.write(f'N_HEAD: {N_HEAD}\n')
    f.write(f'N_LAYERS: {N_LAYERS}\n')
    f.write(f'D_FF: {D_FF}\n')
    f.write(f'DROPOUT: {DROPOUT}\n')
    f.write(f'LEARNING_RATE: {LEARNING_RATE}\n')
    f.write(f'PATIENCE: {PATIENCE}\n')
    f.write(f'MAX_EPOCHS: {MAX_EPOCHS}\n')
    f.write(f'TRAIN_RATIO: {TRAIN_RATIO}\n')
    f.write(f'VAL_RATIO: {VAL_RATIO}\n')
    f.write(f'TEST_RATIO: {TEST_RATIO}\n')
    f.write(f'SEED: {SEED}\n')
    f.write(f'NUM_WORKERS: {NUM_WORKERS}\n')
    f.write(f'PERSISTENT_WORKERS: {PERSISTENT_WORKERS}\n')
    f.write(f'DATA_PATH: {DATA_PATH}\n')
    f.write(f'FEATURE_NAMES_PATH: {FEATURE_NAMES_PATH}\n')
    f.write(f'ADJ_PATH: {ADJ_PATH}\n')
    f.write(f'OUTPUT_DIR: {OUTPUT_DIR}\n')
    f.write(f'TARGET_NAMES: {TARGET_NAMES}\n')
    f.write(f'EXOGENOUS_FEATURE_NAMES: {EXOGENOUS_FEATURE_NAMES}\n')
    f.write(f'TIME_FEATURE_NAMES: {TIME_FEATURE_NAMES}\n')
    f.write(f'RANDOM_MISSING_RATIOS: {RANDOM_MISSING_RATIOS}\n')
    f.write(f'SINGLE_STATION_BLOCK_LENGTHS: {SINGLE_STATION_BLOCK_LENGTHS}\n')
    f.write(f'SYNC_ALL_STATION_BLOCK_LENGTHS: {SYNC_ALL_STATION_BLOCK_LENGTHS}\n')
    f.write(f'best_epoch: {best_epoch}\n')
    f.write(f'best_val_loss: {best_val_loss:.6f}\n')
    f.write(f'learned_bias_scale: {model.bias_scale.item():.6f}\n\n')
    for group_name, group_results in all_eval_results.items():
        f.write(f'========== {group_name} ==========\n')
        for key, result_dict in group_results.items():
            overall = result_dict['overall']
            f.write(f'-- scenario={key} --\n')
            if overall is None:
                f.write('overall: no valid samples\n\n')
                continue
            f.write(f'overall: R2={overall["R2"]:.4f}, RMSE={overall["RMSE"]:.4f}, MAE={overall["MAE"]:.4f}, n_valid={overall["n_valid"]}\n')
            f.write('per_variable:\n')
            for name, m in result_dict['per_variable'].items():
                f.write(f'  {name}: R2={m["R2"]:.4f}, RMSE={m["RMSE"]:.4f}, MAE={m["MAE"]:.4f}, n_valid={m["n_valid"]}\n')
            f.write('per_station:\n')
            for name, m in result_dict['per_station'].items():
                f.write(f'  {name}: R2={m["R2"]:.4f}, RMSE={m["RMSE"]:.4f}, MAE={m["MAE"]:.4f}, n_valid={m["n_valid"]}\n')
            f.write('\n')

print(f'\n三组评估结果已保存到 {OUTPUT_DIR}')

