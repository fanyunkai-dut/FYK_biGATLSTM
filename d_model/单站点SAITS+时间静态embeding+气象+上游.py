#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
水质主分支 + 气象条件分支 + 空间邻站分支 WQ-Met-Spatial Context-SAITS

本脚本实现前三个分支：
- 强 SAITS 水质主分支
- 气象条件分支
- 邻接矩阵约束的空间邻站分支
- 不使用 GAT，只用 spatial cross-attention
- 利用：5 个水质目标变量 + 时间特征 + 静态站点特征 + gap/mask 特征 + 动态气象特征

核心逻辑：
1. 读取 preprocessed_data.npy 和 feature_names.pkl
2. 根据 target_names 找 5 个水质变量
3. 根据 target_mask 列或 NaN 构造原始缺失 M_orig
4. 按时间 7:2:1 划分 train/val/test
5. 只用训练集原始观测位置计算水质标准化参数
6. station-wise 滑动窗口构造样本
7. 训练时动态制造连续人工缺失 M_art
8. 用 M_in = M_orig * (1 - M_art) 构造模型实际输入
9. 基于 M_in 生成 gap 特征
10. 构造水质分支输入：
    h_val = Linear([X_in, M_in])
    z = LayerNorm(h_val + alpha_time*time + alpha_static*static + alpha_gap*gap + alpha_station*station)
11. 保留强 SAITS：两阶段插补 + diagonal masked encoder + 窗口内位置编码
12. 输出 5 个水质变量，使用 stage1/stage2/final 多重 loss，只在 M_art=1 的人工缺失位置计算
13. 训练/验证/测试输出 RMSE、MAE、R2、NSE
14. 训练结束后按缺失长度评估测试集，并为 5 个水质指标绘制预测-真实散点图
"""

import os
import json
import math
import random
import pickle
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np
import yaml

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 0. 配置读取：按你的习惯，全部放在代码最前面
# ============================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用的设备: {device}')

CONFIG_PATH = os.environ.get('CONFIG_PATH', 'saits_configs.yaml')
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config_all = yaml.safe_load(f)

# 优先读取专门的水质分支配置；没有就兼容你原来的 GAT_SAITS 配置名。
cfg = config_all.get(
    'WQ_MET_SPATIAL_CONTEXT_SAITS_training',
    config_all.get(
        'WQ_MET_CONTEXT_SAITS_training',
        config_all.get(
            'WQ_SAITS_BRANCH_training',
            config_all.get(
                'GAT_SAITS_block_imputation_training',
                config_all.get('GAT_SAITS_training', {}),
            ),
        ),
    ),
)

# -------------------------
# 路径与基础参数
# -------------------------
DATA_PATH = cfg.get('DATA_PATH')
FEATURE_NAMES_PATH = cfg.get('FEATURE_NAMES_PATH')
OUTPUT_DIR = cfg.get('OUTPUT_DIR')
ADJ_PATH = cfg.get('ADJ_PATH')

SEQ_LEN = cfg.get('SEQ_LEN', 192)
WINDOW_STRIDE = cfg.get('WINDOW_STRIDE', 1)
BATCH_SIZE = cfg.get('BATCH_SIZE', 8)
D_MODEL = cfg.get('D_MODEL', 128)
N_HEAD = cfg.get('N_HEAD', 4)
N_LAYERS = cfg.get('N_LAYERS', 2)
D_FF = cfg.get('D_FF', 256)
DROPOUT = cfg.get('DROPOUT', 0.1)
LEARNING_RATE = cfg.get('LEARNING_RATE', 1e-3)
WEIGHT_DECAY = cfg.get('WEIGHT_DECAY', 1e-4)
PATIENCE = cfg.get('PATIENCE', 10)
MAX_EPOCHS = cfg.get('MAX_EPOCHS', 100)
TRAIN_RATIO = cfg.get('TRAIN_RATIO', 0.7)
VAL_RATIO = cfg.get('VAL_RATIO', 0.15)
TEST_RATIO = cfg.get('TEST_RATIO', 0.15)
SEED = cfg.get('SEED', 42)
NUM_WORKERS = cfg.get('NUM_WORKERS', 4)
PERSISTENT_WORKERS = cfg.get('PERSISTENT_WORKERS', True)

MASK_SUFFIX = cfg.get('MASK_SUFFIX', '_mask')
DT_SUFFIX = cfg.get('DT_SUFFIX', '_dt')

# -------------------------
# 变量名配置
# -------------------------
TARGET_NAMES = cfg.get('target_names', ['总氮', '总磷', '水温', 'pH', '溶解氧'])
EXOGENOUS_FEATURE_NAMES = cfg.get('exogenous_feature_names', [])
TIME_FEATURE_NAMES = cfg.get('time_feature_names', ['year_id', 'month_id', 'day_id', 'hour_id'])

# 水质分支暂时不使用气象变量。
# 如果 configs.yaml 里没有 static_feature_names，就从你给出的 exogenous_feature_names 中自动识别静态站点特征。
STATIC_CANDIDATES = [
    '平均dem', '坡度', '面积', '上游面积总和', '河流等级', '主河道长度', '是否干流',
    '耕地', '林地', '城镇', '农村',
]
STATIC_FEATURE_NAMES = cfg.get(
    'static_feature_names',
    [name for name in EXOGENOUS_FEATURE_NAMES if name in STATIC_CANDIDATES],
)

# 第二分支：气象/外部动态条件特征。
# 注意：这里不要包含 year_id/month_id/day_id/hour_id，也不要包含静态站点属性。
DEFAULT_METEOROLOGICAL_FEATURE_NAMES = [
    'P4', 'P8', 'P12', 'P24', 'P48', 'P72', 'Imax24', 'API72',
    'T_now', 'T24_mean', 'RH_now', 'RH24_mean',
    'SWD_now', 'SWD24_mean', 'LWD24_mean', 'Wind24_mean', 'Pres_now',
]
METEOROLOGICAL_FEATURE_NAMES = cfg.get(
    'meteorological_feature_names',
    [name for name in EXOGENOUS_FEATURE_NAMES if name in DEFAULT_METEOROLOGICAL_FEATURE_NAMES] or DEFAULT_METEOROLOGICAL_FEATURE_NAMES,
)

# 气象分支设置：强 SAITS 主干不动，气象单独编码后通过 cross-attention 注入两个 SAITS stage。
USE_MET_BRANCH = bool(cfg.get('USE_MET_BRANCH', True))
MET_N_HEAD = int(cfg.get('MET_N_HEAD', N_HEAD))
MET_N_LAYERS = int(cfg.get('MET_N_LAYERS', 1))
MET_D_FF = int(cfg.get('MET_D_FF', D_FF))
MET_DROPOUT = float(cfg.get('MET_DROPOUT', DROPOUT))
MET_CROSS_HEADS = int(cfg.get('MET_CROSS_HEADS', N_HEAD))
MET_ALPHA_INIT = float(cfg.get('MET_ALPHA_INIT', 0.1))
MET_CONTEXT_ALPHA_INIT = float(cfg.get('MET_CONTEXT_ALPHA_INIT', 0.1))
MET_USE_TIME_CONTEXT = bool(cfg.get('MET_USE_TIME_CONTEXT', True))
MET_USE_STATIC_CONTEXT = bool(cfg.get('MET_USE_STATIC_CONTEXT', True))
MET_USE_STATION_CONTEXT = bool(cfg.get('MET_USE_STATION_CONTEXT', True))

# 第三分支：上下游/邻站空间条件分支。
# 第一版不做 GAT，只使用邻接矩阵约束的 spatial cross-attention。
USE_SPATIAL_BRANCH = bool(cfg.get('USE_SPATIAL_BRANCH', True))
SPATIAL_N_HEAD = int(cfg.get('SPATIAL_N_HEAD', N_HEAD))
SPATIAL_N_LAYERS = int(cfg.get('SPATIAL_N_LAYERS', 1))
SPATIAL_D_FF = int(cfg.get('SPATIAL_D_FF', D_FF))
SPATIAL_DROPOUT = float(cfg.get('SPATIAL_DROPOUT', DROPOUT))
SPATIAL_CROSS_HEADS = int(cfg.get('SPATIAL_CROSS_HEADS', N_HEAD))
SPATIAL_ALPHA_INIT = float(cfg.get('SPATIAL_ALPHA_INIT', 0.1))
SPATIAL_CONTEXT_ALPHA_INIT = float(cfg.get('SPATIAL_CONTEXT_ALPHA_INIT', 0.1))
SPATIAL_USE_TIME_CONTEXT = bool(cfg.get('SPATIAL_USE_TIME_CONTEXT', True))
SPATIAL_USE_STATIC_CONTEXT = bool(cfg.get('SPATIAL_USE_STATIC_CONTEXT', True))
SPATIAL_USE_STATION_CONTEXT = bool(cfg.get('SPATIAL_USE_STATION_CONTEXT', True))
SPATIAL_EXCLUDE_SELF = bool(cfg.get('SPATIAL_EXCLUDE_SELF', True))
SPATIAL_FALLBACK_TO_ALL = bool(cfg.get('SPATIAL_FALLBACK_TO_ALL', True))
SPATIAL_ADJ_TRANSPOSE = bool(cfg.get('SPATIAL_ADJ_TRANSPOSE', False))

# -------------------------
# 人工连续缺失配置
# -------------------------
block_cfg = cfg.get('block_masking', {})
TRAIN_BLOCK_LENGTHS = block_cfg.get('train_block_lengths', [3, 6, 12, 18, 42, 180])
TRAIN_NUM_BLOCKS_PER_SAMPLE = block_cfg.get('train_num_blocks_per_sample', 1)
TRAIN_BLOCK_SEED = block_cfg.get('train_block_seed', 42)

VAL_BLOCK_LENGTHS = block_cfg.get('val_block_lengths', [3, 6, 12, 18, 42, 180])
VAL_NUM_BLOCKS_PER_SAMPLE = block_cfg.get('val_num_blocks_per_sample', 1)
VAL_BLOCK_SEED = block_cfg.get('val_block_seed', 202)

TEST_BLOCK_LENGTHS = block_cfg.get('test_block_lengths', VAL_BLOCK_LENGTHS)
TEST_NUM_BLOCKS_PER_SAMPLE = block_cfg.get('test_num_blocks_per_sample', VAL_NUM_BLOCKS_PER_SAMPLE)
TEST_BLOCK_SEED = block_cfg.get('test_block_seed', 303)

# 训练结束后，专门按这些连续缺失长度分别在测试集评估。
# 注意：这里默认包含 9，即使训练 block 里没有 9，也可以单独测试 9 小时缺失泛化效果。
TEST_EVAL_BLOCK_LENGTHS = cfg.get('TEST_EVAL_BLOCK_LENGTHS', [3, 6, 9, 12, 18, 42, 180])
TEST_EVAL_BLOCK_SEED = cfg.get('TEST_EVAL_BLOCK_SEED', 5051)
MAX_SCATTER_POINTS_PER_TARGET = int(cfg.get('MAX_SCATTER_POINTS_PER_TARGET', 20000))
EVAL_IN_ORIGINAL_SCALE = bool(cfg.get('EVAL_IN_ORIGINAL_SCALE', False))

# all_targets：一次连续缺失同时遮住 5 个水质变量，更像多参数探头故障。
# random_variable：每次只遮住一个水质变量。
# random_subset：每次随机遮住一部分水质变量。
MASK_MODE = block_cfg.get('mask_mode', 'all_targets')
MIN_WINDOW_OBSERVED_RATIO = block_cfg.get('min_window_observed_ratio', 0.8)
ALLOW_FALLBACK_SHORTER = block_cfg.get('allow_fallback_shorter', True)

# 长缺失必须给足左右上下文；180 如果 SEQ_LEN=192，基本不会被有效采样，这是有意限制。
MIN_CONTEXT_BY_LENGTH = block_cfg.get(
    'min_context_by_length',
    {3: 6, 6: 12, 12: 24, 18: 24, 42: 48, 180: 72},
)
MIN_CONTEXT_BY_LENGTH = {int(k): int(v) for k, v in MIN_CONTEXT_BY_LENGTH.items()}
# 补齐常用长度的默认上下文限制，特别是 9 小时测试场景。
for _length, _ctx in {3: 12, 6: 12, 9: 24, 12: 24, 18: 24, 42: 48, 180: 72}.items():
    MIN_CONTEXT_BY_LENGTH.setdefault(_length, _ctx)

# -------------------------
# 模型输入融合设置
# -------------------------
CONTEXT_ALPHA_INIT = cfg.get('CONTEXT_ALPHA_INIT', 0.1)
USE_STATION_ID_EMBEDDING = cfg.get('USE_STATION_ID_EMBEDDING', True)
OBS_RECON_LOSS_WEIGHT = cfg.get('OBS_RECON_LOSS_WEIGHT', 0.0)

# 日历时间 embedding。你的预处理应提供 year_id / month_id / day_id / hour_id 四列类别编号。
# year_id 的类别数会根据数据自动推断，也可在配置中显式指定 NUM_YEARS。
USE_CALENDAR_TIME_EMBEDDING = bool(cfg.get('USE_CALENDAR_TIME_EMBEDDING', True))
NUM_YEARS = cfg.get('NUM_YEARS', None)
YEAR_EMB_DIM = int(cfg.get('YEAR_EMB_DIM', 4))
MONTH_EMB_DIM = int(cfg.get('MONTH_EMB_DIM', 4))
DAY_EMB_DIM = int(cfg.get('DAY_EMB_DIM', 4))
HOUR_EMB_DIM = int(cfg.get('HOUR_EMB_DIM', 4))

# 如果你后续要加气象或上游分支，再新增配置；这里刻意不读取 ADJ / GAT。

if DATA_PATH is None:
    raise ValueError('DATA_PATH 不能为空')
if FEATURE_NAMES_PATH is None:
    raise ValueError('FEATURE_NAMES_PATH 不能为空')
if OUTPUT_DIR is None:
    raise ValueError('OUTPUT_DIR 不能为空')
if USE_SPATIAL_BRANCH and ADJ_PATH is None:
    raise ValueError('USE_SPATIAL_BRANCH=True 时 ADJ_PATH 不能为空')

os.makedirs(OUTPUT_DIR, exist_ok=True)

print('========== 当前水质分支配置 ==========', flush=True)
print(f'CONFIG_PATH: {CONFIG_PATH}')
print(f'DATA_PATH: {DATA_PATH}')
print(f'FEATURE_NAMES_PATH: {FEATURE_NAMES_PATH}')
print(f'OUTPUT_DIR: {OUTPUT_DIR}')
print(f'ADJ_PATH: {ADJ_PATH}')
print(f'SEQ_LEN: {SEQ_LEN}, WINDOW_STRIDE: {WINDOW_STRIDE}, BATCH_SIZE: {BATCH_SIZE}')
print(f'D_MODEL: {D_MODEL}, N_HEAD: {N_HEAD}, N_LAYERS: {N_LAYERS}, D_FF: {D_FF}')
print(f'TARGET_NAMES: {TARGET_NAMES}')
print(f'TIME_FEATURE_NAMES: {TIME_FEATURE_NAMES}')
print(f'STATIC_FEATURE_NAMES: {STATIC_FEATURE_NAMES}')
print(f'METEOROLOGICAL_FEATURE_NAMES: {METEOROLOGICAL_FEATURE_NAMES}')
print(f'USE_MET_BRANCH: {USE_MET_BRANCH}, MET_N_LAYERS: {MET_N_LAYERS}, MET_CROSS_HEADS: {MET_CROSS_HEADS}')
print(f'USE_SPATIAL_BRANCH: {USE_SPATIAL_BRANCH}, SPATIAL_N_LAYERS: {SPATIAL_N_LAYERS}, SPATIAL_CROSS_HEADS: {SPATIAL_CROSS_HEADS}, SPATIAL_ADJ_TRANSPOSE: {SPATIAL_ADJ_TRANSPOSE}')
print(f'TRAIN_BLOCK_LENGTHS: {TRAIN_BLOCK_LENGTHS}')
print(f'TEST_EVAL_BLOCK_LENGTHS: {TEST_EVAL_BLOCK_LENGTHS}')
print(f'EVAL_IN_ORIGINAL_SCALE: {EVAL_IN_ORIGINAL_SCALE}')
print(f'MIN_CONTEXT_BY_LENGTH: {MIN_CONTEXT_BY_LENGTH}')
print('=====================================', flush=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 训练速度优先；如果你要完全复现，可以把 benchmark=False、deterministic=True。
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


set_seed(SEED)


# ============================================================
# 1. 基础工具
# ============================================================

def load_feature_names(path: str) -> List[str]:
    with open(path, 'rb') as f:
        obj = pickle.load(f)

    if isinstance(obj, (list, tuple)):
        return list(obj)

    if isinstance(obj, dict):
        for key in ['feature_names', 'features', 'columns', 'names']:
            if key in obj:
                return list(obj[key])
        if all(isinstance(k, int) for k in obj.keys()):
            return [obj[i] for i in sorted(obj.keys())]
        if all(isinstance(v, int) for v in obj.values()):
            return [k for k, _ in sorted(obj.items(), key=lambda kv: kv[1])]

    raise ValueError(f'无法从 {path} 解析 feature_names，实际类型：{type(obj)}')


def load_adjacency_matrix(path: str, expected_size: int) -> np.ndarray:
    """读取邻接矩阵 CSV，返回 [S,S] float32。非零视为可参考邻接关系。"""
    import pandas as pd
    df = pd.read_csv(path, header=None)
    mat = df.values
    if mat.shape[0] == expected_size + 1 and mat.shape[1] == expected_size + 1:
        mat = mat[1:, 1:]
    elif mat.shape[0] == expected_size and mat.shape[1] == expected_size + 1:
        mat = mat[:, 1:]
    elif mat.shape[0] == expected_size + 1 and mat.shape[1] == expected_size:
        mat = mat[1:, :]
    if mat.shape != (expected_size, expected_size):
        raise ValueError(f'邻接矩阵 shape={mat.shape}，但站点数 S={expected_size}，无法对应。')
    mat = mat.astype(np.float32)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    mat = (mat != 0).astype(np.float32)
    if SPATIAL_ADJ_TRANSPOSE:
        mat = mat.T.copy()
    return mat


def get_indices(feature_names: Sequence[str], wanted: Sequence[str], group_name: str) -> List[int]:
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    missing = [name for name in wanted if name not in name_to_idx]
    if missing:
        raise ValueError(
            f'{group_name} 中这些特征名不在 feature_names.pkl 里：{missing}\n'
            f'请检查 configs.yaml 中的名字是否和 pkl 完全一致。'
        )
    return [name_to_idx[name] for name in wanted]


def get_optional_indices(feature_names: Sequence[str], wanted: Sequence[str]) -> Tuple[List[int], List[str]]:
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    idx, found = [], []
    for name in wanted:
        if name in name_to_idx:
            idx.append(name_to_idx[name])
            found.append(name)
    return idx, found


def robust_std(x: np.ndarray, axis=None, keepdims=False, eps: float = 1e-6) -> np.ndarray:
    s = np.nanstd(x, axis=axis, keepdims=keepdims)
    return np.where(s < eps, 1.0, s)


def split_by_time(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    T = arr.shape[0]
    n_train = int(T * TRAIN_RATIO)
    n_val = int(T * VAL_RATIO)
    train = arr[:n_train]
    val = arr[n_train:n_train + n_val]
    test = arr[n_train + n_val:]
    return train, val, test


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum()
    if denom.item() < 1.0:
        return pred.sum() * 0.0
    return (((pred - target) ** 2) * mask).sum() / denom.clamp_min(1.0)


def finalize_metric_sums(total_abs: float, total_sq: float, total_y: float, total_y2: float, total_count: int) -> Dict[str, float]:
    """
    基于人工缺失位置的累加量计算 MAE / RMSE / R2 / NSE。
    这里 R2 和 NSE 都采用 1 - SSE/SST；对整体 flatten 后计算时二者数值相同。
    """
    if total_count <= 0:
        return {'loss': float('nan'), 'mae': float('nan'), 'rmse': float('nan'), 'r2': float('nan'), 'nse': float('nan')}

    mse = total_sq / max(total_count, 1)
    mae = total_abs / max(total_count, 1)
    rmse = math.sqrt(mse)
    sst = total_y2 - (total_y * total_y) / max(total_count, 1)
    if sst <= 1e-12:
        r2 = float('nan')
        nse = float('nan')
    else:
        r2 = 1.0 - total_sq / sst
        nse = 1.0 - total_sq / sst
    return {'loss': mse, 'mae': mae, 'rmse': rmse, 'r2': r2, 'nse': nse}


def compute_metrics_np(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    n = int(y_true.size)
    if n == 0:
        return {'loss': float('nan'), 'mae': float('nan'), 'rmse': float('nan'), 'r2': float('nan'), 'nse': float('nan'), 'count': 0}
    err = y_pred - y_true
    sse = float(np.sum(err ** 2))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if sst <= 1e-12:
        r2 = float('nan')
        nse = float('nan')
    else:
        r2 = 1.0 - sse / sst
        nse = 1.0 - sse / sst
    return {'loss': sse / n, 'mae': mae, 'rmse': rmse, 'r2': r2, 'nse': nse, 'count': n}


def safe_filename(name: str) -> str:
    keep = []
    for ch in str(name):
        if ch.isalnum() or ch in ('_', '-', '.'):
            keep.append(ch)
        else:
            keep.append('_')
    return ''.join(keep)


def move_batch(batch: Dict[str, torch.Tensor], dev: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(dev, non_blocking=True) for k, v in batch.items()}


# ============================================================
# 2. 数据准备
# ============================================================

class PreparedArrays:
    def __init__(
        self,
        target_true_std: np.ndarray,
        m_orig: np.ndarray,
        time_raw: np.ndarray,
        static_std: np.ndarray,
        met_std: np.ndarray,
        target_mean: np.ndarray,
        target_std_val: np.ndarray,
        static_mean: np.ndarray,
        static_std_val: np.ndarray,
        met_mean: np.ndarray,
        met_std_val: np.ndarray,
        feature_names: List[str],
    ):
        self.target_true_std = target_true_std
        self.m_orig = m_orig
        self.time_raw = time_raw
        self.static_std = static_std
        self.met_std = met_std
        self.target_mean = target_mean
        self.target_std_val = target_std_val
        self.static_mean = static_mean
        self.static_std_val = static_std_val
        self.met_mean = met_mean
        self.met_std_val = met_std_val
        self.feature_names = feature_names


def prepare_arrays() -> PreparedArrays:
    data = np.load(DATA_PATH).astype(np.float32)
    if data.ndim == 2:
        data = data[:, None, :]
    if data.ndim != 3:
        raise ValueError(f'preprocessed_data.npy 需要是 [T,S,F] 或 [T,F]，实际 shape={data.shape}')

    T, S, F_all = data.shape
    feature_names = load_feature_names(FEATURE_NAMES_PATH)
    if len(feature_names) != F_all:
        raise ValueError(f'feature_names 数量 {len(feature_names)} 与 data 最后一维 {F_all} 不一致。')

    target_idx = get_indices(feature_names, TARGET_NAMES, 'target_names')
    time_idx = get_indices(feature_names, TIME_FEATURE_NAMES, 'time_feature_names') if TIME_FEATURE_NAMES else []
    static_idx = get_indices(feature_names, STATIC_FEATURE_NAMES, 'static_feature_names') if STATIC_FEATURE_NAMES else []
    met_idx = get_indices(feature_names, METEOROLOGICAL_FEATURE_NAMES, 'meteorological_feature_names') if (USE_MET_BRANCH and METEOROLOGICAL_FEATURE_NAMES) else []

    target_raw = data[:, :, target_idx].astype(np.float32)  # [T,S,5]

    # 优先使用 “变量名 + MASK_SUFFIX” 列构造原始缺失 mask。
    target_mask_names = [name + MASK_SUFFIX for name in TARGET_NAMES]
    mask_idx, found_mask_names = get_optional_indices(feature_names, target_mask_names)
    if len(mask_idx) == len(TARGET_NAMES):
        m_orig = (data[:, :, mask_idx] > 0.5).astype(np.float32)
        print(f'[INFO] 使用 mask 列构造 M_orig: {found_mask_names}')
    else:
        m_orig = np.isfinite(target_raw).astype(np.float32)
        print('[INFO] 未找到完整 target mask 列，使用 np.isfinite(target) 构造 M_orig。')

    # 原始缺失位置设为 NaN，避免预处理填充值污染标准化。
    target_for_stat = target_raw.copy()
    target_for_stat[m_orig < 0.5] = np.nan

    n_train = int(T * TRAIN_RATIO)
    train_target = target_for_stat[:n_train]

    # 每个水质变量单独标准化；只用训练集的原始观测位置。
    target_mean = np.nanmean(train_target, axis=(0, 1)).astype(np.float32)
    target_std_val = robust_std(train_target, axis=(0, 1)).astype(np.float32)

    target_std = (target_for_stat - target_mean.reshape(1, 1, -1)) / target_std_val.reshape(1, 1, -1)
    target_true_std = np.nan_to_num(target_std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if time_idx:
        time_raw = data[:, :, time_idx].astype(np.float32)
        time_raw = np.nan_to_num(time_raw, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        time_raw = np.zeros((T, S, 0), dtype=np.float32)

    if static_idx:
        static_raw_all_time = data[:, :, static_idx].astype(np.float32)  # [T,S,Fs]
        # 静态特征按站点取训练期均值，得到 [S,Fs]。
        static_station = np.nanmean(static_raw_all_time[:n_train], axis=0)
        static_mean = np.nanmean(static_station, axis=0).astype(np.float32)
        static_std_val = robust_std(static_station, axis=0).astype(np.float32)
        static_std = (static_station - static_mean.reshape(1, -1)) / static_std_val.reshape(1, -1)
        static_std = np.nan_to_num(static_std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    else:
        static_std = np.zeros((S, 0), dtype=np.float32)
        static_mean = np.zeros((0,), dtype=np.float32)
        static_std_val = np.ones((0,), dtype=np.float32)

    # 气象/外部动态特征：按 train 时间段计算均值方差，随后对全时段 transform。
    # 这里把气象视为完整外部条件；若存在 NaN，则用训练均值标准化后再填 0。
    if met_idx:
        met_raw = data[:, :, met_idx].astype(np.float32)  # [T,S,Fm]
        met_train = met_raw[:n_train]
        met_mean = np.nanmean(met_train, axis=(0, 1)).astype(np.float32)
        met_std_val = robust_std(met_train, axis=(0, 1)).astype(np.float32)
        met_std = (met_raw - met_mean.reshape(1, 1, -1)) / met_std_val.reshape(1, 1, -1)
        met_std = np.nan_to_num(met_std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    else:
        met_std = np.zeros((T, S, 0), dtype=np.float32)
        met_mean = np.zeros((0,), dtype=np.float32)
        met_std_val = np.ones((0,), dtype=np.float32)

    print('[INFO] data shape:', data.shape)
    print('[INFO] target observed ratio:', float(m_orig.mean()))
    print('[INFO] time_dim:', time_raw.shape[-1], '| static_dim:', static_std.shape[-1], '| met_dim:', met_std.shape[-1])

    return PreparedArrays(
        target_true_std=target_true_std,
        m_orig=m_orig.astype(np.float32),
        time_raw=time_raw.astype(np.float32),
        static_std=static_std.astype(np.float32),
        met_std=met_std.astype(np.float32),
        target_mean=target_mean,
        target_std_val=target_std_val,
        static_mean=static_mean,
        static_std_val=static_std_val,
        met_mean=met_mean,
        met_std_val=met_std_val,
        feature_names=feature_names,
    )


# ============================================================
# 3. Dataset：station-wise 滑窗 + 动态人工连续缺失
# ============================================================

class WQWindowDataset(Dataset):
    def __init__(
        self,
        x_true: np.ndarray,       # [T,S,Fq]，标准化后，原始缺失填 0
        m_orig: np.ndarray,       # [T,S,Fq]
        time_raw: np.ndarray,     # [T,S,Ft]
        static_std: np.ndarray,   # [S,Fs]
        met_std: np.ndarray,      # [T,S,Fm]
        mode: str,
        block_lengths_override: Optional[Sequence[int]] = None,
        num_blocks_override: Optional[int] = None,
        base_seed_override: Optional[int] = None,
    ):
        super().__init__()
        assert mode in {'train', 'val', 'test'}
        self.x_true = x_true.astype(np.float32)
        self.m_orig = m_orig.astype(np.float32)
        self.time_raw = time_raw.astype(np.float32)
        self.static_std = static_std.astype(np.float32)
        self.met_std = met_std.astype(np.float32)
        self.mode = mode
        self.seq_len = int(SEQ_LEN)
        self.stride = int(WINDOW_STRIDE)
        self.epoch = 0

        if mode == 'train':
            self.block_lengths = [int(x) for x in TRAIN_BLOCK_LENGTHS]
            self.num_blocks = int(TRAIN_NUM_BLOCKS_PER_SAMPLE)
            self.base_seed = int(TRAIN_BLOCK_SEED)
            self.dynamic = True
        elif mode == 'val':
            self.block_lengths = [int(x) for x in VAL_BLOCK_LENGTHS]
            self.num_blocks = int(VAL_NUM_BLOCKS_PER_SAMPLE)
            self.base_seed = int(VAL_BLOCK_SEED)
            self.dynamic = False
        else:
            self.block_lengths = [int(x) for x in TEST_BLOCK_LENGTHS]
            self.num_blocks = int(TEST_NUM_BLOCKS_PER_SAMPLE)
            self.base_seed = int(TEST_BLOCK_SEED)
            self.dynamic = False

        if block_lengths_override is not None:
            self.block_lengths = [int(x) for x in block_lengths_override]
        if num_blocks_override is not None:
            self.num_blocks = int(num_blocks_override)
        if base_seed_override is not None:
            self.base_seed = int(base_seed_override)

        T, S, Fq = self.x_true.shape
        if T < self.seq_len:
            raise ValueError(f'{mode} split 长度 T={T} 小于 SEQ_LEN={self.seq_len}')
        self.T, self.S, self.Fq = T, S, Fq

        # station-wise 样本：每个窗口起点 + 每个站点 = 一个样本。
        self.indices = []
        for start in range(0, T - self.seq_len + 1, self.stride):
            end = start + self.seq_len
            for s in range(S):
                m_win = self.m_orig[start:end, s, :]
                if float(m_win.mean()) >= float(MIN_WINDOW_OBSERVED_RATIO):
                    self.indices.append((start, s))

        print(f'[INFO] {mode} dataset windows: {len(self.indices)} | T={T}, S={S}, stride={self.stride}')

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.indices)

    def _rng_for_index(self, idx: int) -> np.random.Generator:
        # train 动态：同一样本不同 epoch 挖法不同。
        # val/test 固定：同一样本每次挖法一致，便于公平比较。
        if self.dynamic:
            seed = self.base_seed + self.epoch * 1_000_003 + idx * 97
        else:
            seed = self.base_seed + idx * 97
        return np.random.default_rng(seed)

    def _select_variables(self, rng: np.random.Generator) -> np.ndarray:
        if MASK_MODE == 'all_targets':
            return np.arange(self.Fq, dtype=np.int64)
        if MASK_MODE == 'random_variable':
            return np.array([rng.integers(0, self.Fq)], dtype=np.int64)
        if MASK_MODE == 'random_subset':
            k = int(rng.integers(1, self.Fq + 1))
            return np.sort(rng.choice(np.arange(self.Fq), size=k, replace=False)).astype(np.int64)
        raise ValueError(f'未知 MASK_MODE={MASK_MODE}')

    def _valid_starts_for_block(self, m_available: np.ndarray, block_len: int, var_idx: np.ndarray) -> List[int]:
        """
        m_available: [L,F]，1 表示此位置目前仍可被人工遮住。
        block_len: 连续缺失长度。
        var_idx: 被遮住的变量索引。
        """
        L = m_available.shape[0]
        min_ctx = MIN_CONTEXT_BY_LENGTH.get(int(block_len), 0)
        if block_len + 2 * min_ctx > L:
            return []

        valid = []
        max_start = L - block_len - min_ctx
        for st in range(min_ctx, max_start + 1):
            seg = m_available[st:st + block_len, :][:, var_idx]
            if np.all(seg > 0.5):
                valid.append(st)
        return valid

    def _make_artificial_mask(self, m_orig: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        m_orig: [L,Fq]，1 原始有值，0 原始缺失。
        返回 M_art: [L,Fq]，1 表示人工遮住。
        """
        L, Fq = m_orig.shape
        m_art = np.zeros((L, Fq), dtype=np.float32)

        for _ in range(self.num_blocks):
            var_idx = self._select_variables(rng)
            desired = int(rng.choice(self.block_lengths))

            if ALLOW_FALLBACK_SHORTER:
                # 先尝试抽到的长度；如果放不了，再从长到短尝试其他长度。
                candidates = [desired] + [x for x in sorted(self.block_lengths, reverse=True) if int(x) != desired]
            else:
                candidates = [desired]

            placed = False
            for blen in candidates:
                m_available = m_orig * (1.0 - m_art)
                valid_starts = self._valid_starts_for_block(m_available, int(blen), var_idx)
                if valid_starts:
                    st = int(rng.choice(valid_starts))
                    m_art[st:st + int(blen), var_idx] = 1.0
                    placed = True
                    break

            # 如果放不了任何 block，保持 m_art 全 0；该样本 loss 为 0。
            # 如果这种情况很多，说明 SEQ_LEN / 观测率 / min_context 设置过严。
            if not placed:
                pass

        return m_art

    @staticmethod
    def _compute_gap_features(m_in: np.ndarray) -> np.ndarray:
        """
        m_in: [L,F]，1 observed，0 missing。
        返回 [L, F*3]：missing_flag, gap_len_norm, gap_pos_ratio。
        """
        L, F = m_in.shape
        missing_flag = (m_in < 0.5).astype(np.float32)
        gap_len = np.zeros((L, F), dtype=np.float32)
        gap_pos = np.zeros((L, F), dtype=np.float32)

        for f in range(F):
            t = 0
            while t < L:
                if missing_flag[t, f] < 0.5:
                    t += 1
                    continue
                start = t
                while t < L and missing_flag[t, f] > 0.5:
                    t += 1
                end = t
                length = end - start
                if length > 0:
                    gap_len[start:end, f] = float(length) / float(max(L, 1))
                    gap_pos[start:end, f] = np.arange(1, length + 1, dtype=np.float32) / float(length)

        return np.concatenate([missing_flag, gap_len, gap_pos], axis=-1).astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start, station = self.indices[idx]
        end = start + self.seq_len
        rng = self._rng_for_index(idx)

        # 目标站点窗口。
        x_true = self.x_true[start:end, station, :]       # [L,F]
        m_orig = self.m_orig[start:end, station, :]       # [L,F]
        time_raw = self.time_raw[start:end, station, :]   # [L,Ft]
        static = self.static_std[station, :]              # [Fs]
        met = self.met_std[start:end, station, :]         # [L,Fm]

        m_art = self._make_artificial_mask(m_orig, rng)   # [L,F]
        m_in = m_orig * (1.0 - m_art)
        x_in = x_true * m_in                              # 原始缺失和人工缺失都填 0
        gap = self._compute_gap_features(m_in)            # [L,F*3]

        # 空间分支：返回同一窗口内所有站点水质。
        # 人工缺失只施加在目标站点；其他站点只保留原始缺失。
        x_all_true = self.x_true[start:end, :, :].copy()  # [L,S,F]
        m_all_in = self.m_orig[start:end, :, :].copy()    # [L,S,F]
        m_all_in[:, station, :] = m_in
        x_all_in = x_all_true * m_all_in

        return {
            'x_in': torch.from_numpy(x_in).float(),
            'x_true': torch.from_numpy(x_true).float(),
            'm_in': torch.from_numpy(m_in).float(),
            'm_orig': torch.from_numpy(m_orig).float(),
            'm_art': torch.from_numpy(m_art).float(),
            'time_raw': torch.from_numpy(time_raw).float(),
            'static': torch.from_numpy(static).float(),
            'met': torch.from_numpy(met).float(),
            'gap': torch.from_numpy(gap).float(),
            'x_all': torch.from_numpy(x_all_in).float(),
            'm_all': torch.from_numpy(m_all_in).float(),
            'static_all': torch.from_numpy(self.static_std).float(),
            'station_id': torch.tensor(station, dtype=torch.long),
        }


# ============================================================
# 4. 模型：强 SAITS 主干 + 日历时间 embedding + 窗口内位置编码
# ============================================================

class PositionalEncoding(nn.Module):
    """标准窗口内位置编码：告诉 Transformer 当前 token 在窗口里的第几个时间步。"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1,max_len,D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,L,D]
        return x + self.pe[:, :x.size(1), :]


class DiagonalMaskedEncoder(nn.Module):
    """
    原版强 SAITS 的关键之一：
    - 先加窗口内 positional encoding；
    - attention 中禁止每个时间步直接看自己，避免简单复制输入。
    """
    def __init__(self, d_model: int, n_head: int, d_ff: int, dropout: float, n_layers: int, max_len: int):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,L,D]
        L = x.size(1)
        diag_mask = torch.eye(L, dtype=torch.bool, device=x.device)
        x = self.pos_encoder(x)
        return self.encoder(x, mask=diag_mask)


class CalendarTimeEmbedding(nn.Module):
    """
    针对预处理生成的 year_id / month_id / day_id / hour_id 做可训练 embedding。

    注意：
    - 输入不是 sin/cos，而是类别编号；
    - 输入顺序由 time_feature_names 指定；
    - 输出先 concat，再投影到 D_MODEL。
    """
    def __init__(
        self,
        time_feature_names: Sequence[str],
        num_years: int,
        year_dim: int = 4,
        month_dim: int = 4,
        day_dim: int = 4,
        hour_dim: int = 4,
        d_model: int = 128,
    ):
        super().__init__()
        self.time_feature_names = list(time_feature_names)
        required = ['year_id', 'month_id', 'day_id', 'hour_id']
        missing = [name for name in required if name not in self.time_feature_names]
        if missing:
            raise ValueError(
                f'USE_CALENDAR_TIME_EMBEDDING=True 时，time_feature_names 必须包含 {required}，缺少：{missing}'
            )
        self.name_to_pos = {name: i for i, name in enumerate(self.time_feature_names)}
        self.year_emb = nn.Embedding(int(num_years), int(year_dim))
        self.month_emb = nn.Embedding(12, int(month_dim))
        self.day_emb = nn.Embedding(31, int(day_dim))
        self.hour_emb = nn.Embedding(24, int(hour_dim))
        out_dim = int(year_dim) + int(month_dim) + int(day_dim) + int(hour_dim)
        self.proj = nn.Linear(out_dim, d_model)

    def forward(self, time_raw: torch.Tensor) -> torch.Tensor:
        # time_raw: [B,L,Ft]，里面是 float 存储的类别编号；这里转 long。
        year_id = time_raw[..., self.name_to_pos['year_id']].long().clamp_min(0).clamp_max(self.year_emb.num_embeddings - 1)
        month_id = time_raw[..., self.name_to_pos['month_id']].long().clamp_min(0).clamp_max(11)
        day_id = time_raw[..., self.name_to_pos['day_id']].long().clamp_min(0).clamp_max(30)
        hour_id = time_raw[..., self.name_to_pos['hour_id']].long().clamp_min(0).clamp_max(23)
        e = torch.cat([
            self.year_emb(year_id),
            self.month_emb(month_id),
            self.day_emb(day_id),
            self.hour_emb(hour_id),
        ], dim=-1)
        return self.proj(e)  # [B,L,D]


class ContextEmbedding(nn.Module):
    """把 time/static/gap/station 作为辅助 context 注入强 SAITS，而不是替代 SAITS 主干。"""
    def __init__(
        self,
        time_dim: int,
        static_dim: int,
        gap_dim: int,
        num_stations: int,
        num_years: int,
    ):
        super().__init__()
        self.use_calendar = bool(USE_CALENDAR_TIME_EMBEDDING and time_dim > 0)
        if self.use_calendar:
            self.time_proj = CalendarTimeEmbedding(
                time_feature_names=TIME_FEATURE_NAMES,
                num_years=num_years,
                year_dim=YEAR_EMB_DIM,
                month_dim=MONTH_EMB_DIM,
                day_dim=DAY_EMB_DIM,
                hour_dim=HOUR_EMB_DIM,
                d_model=D_MODEL,
            )
        else:
            self.time_proj = nn.Linear(time_dim, D_MODEL) if time_dim > 0 else None

        self.static_proj = nn.Linear(static_dim, D_MODEL) if static_dim > 0 else None
        self.gap_proj = nn.Sequential(
            nn.Linear(gap_dim, D_MODEL),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL, D_MODEL),
        ) if gap_dim > 0 else None
        self.station_emb = nn.Embedding(num_stations, D_MODEL) if USE_STATION_ID_EMBEDDING else None

        # 可学习上下文权重：保证 SAITS 的水质主信息仍是主体。
        self.alpha_time = nn.Parameter(torch.tensor(float(CONTEXT_ALPHA_INIT)))
        self.alpha_static = nn.Parameter(torch.tensor(float(CONTEXT_ALPHA_INIT)))
        self.alpha_gap = nn.Parameter(torch.tensor(float(CONTEXT_ALPHA_INIT)))
        self.alpha_station = nn.Parameter(torch.tensor(float(CONTEXT_ALPHA_INIT)))
        self.norm = nn.LayerNorm(D_MODEL)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(
        self,
        base: torch.Tensor,       # [B,L,D]，来自 SAITS input_proj 的水质主表示
        time_raw: torch.Tensor,   # [B,L,Ft]
        static: torch.Tensor,     # [B,Fs]
        gap: torch.Tensor,        # [B,L,Fg]
        station_id: torch.Tensor, # [B]
    ) -> torch.Tensor:
        z = base
        if self.time_proj is not None and time_raw.shape[-1] > 0:
            z = z + self.alpha_time * self.time_proj(time_raw)
        if self.static_proj is not None and static.shape[-1] > 0:
            z = z + self.alpha_static * self.static_proj(static).unsqueeze(1)
        if self.gap_proj is not None and gap.shape[-1] > 0:
            z = z + self.alpha_gap * self.gap_proj(gap)
        if self.station_emb is not None:
            z = z + self.alpha_station * self.station_emb(station_id).unsqueeze(1)
        return self.dropout(self.norm(z))


class PlainTemporalEncoder(nn.Module):
    """气象分支的普通 Transformer 编码器：不使用 diagonal mask，因为气象不是待补目标。"""
    def __init__(self, d_model: int, n_head: int, d_ff: int, dropout: float, n_layers: int, max_len: int):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos_encoder(x)
        return self.encoder(x)


class MeteorologicalEncoder(nn.Module):
    """
    气象/外部动态条件分支。
    输入 X_met: [B,L,Fm]，输出 H_met: [B,L,D_MODEL]。

    设计原则：
    - 气象是完整外部条件，不作为待补目标；
    - 气象先独立编码成条件表示，再通过 cross-attention 辅助 SAITS 两个 stage；
    - 可选择给气象编码器也注入 time/static/station context。
    """
    def __init__(self, met_dim: int, time_dim: int, static_dim: int, num_stations: int, num_years: int):
        super().__init__()
        self.met_dim = int(met_dim)
        self.input_proj = nn.Linear(self.met_dim, D_MODEL)

        self.use_calendar = bool(MET_USE_TIME_CONTEXT and USE_CALENDAR_TIME_EMBEDDING and time_dim > 0)
        if MET_USE_TIME_CONTEXT and time_dim > 0:
            if self.use_calendar:
                self.time_proj = CalendarTimeEmbedding(
                    time_feature_names=TIME_FEATURE_NAMES,
                    num_years=num_years,
                    year_dim=YEAR_EMB_DIM,
                    month_dim=MONTH_EMB_DIM,
                    day_dim=DAY_EMB_DIM,
                    hour_dim=HOUR_EMB_DIM,
                    d_model=D_MODEL,
                )
            else:
                self.time_proj = nn.Linear(time_dim, D_MODEL)
        else:
            self.time_proj = None

        self.static_proj = nn.Linear(static_dim, D_MODEL) if (MET_USE_STATIC_CONTEXT and static_dim > 0) else None
        self.station_emb = nn.Embedding(num_stations, D_MODEL) if MET_USE_STATION_CONTEXT else None

        self.alpha_time = nn.Parameter(torch.tensor(float(MET_CONTEXT_ALPHA_INIT)))
        self.alpha_static = nn.Parameter(torch.tensor(float(MET_CONTEXT_ALPHA_INIT)))
        self.alpha_station = nn.Parameter(torch.tensor(float(MET_CONTEXT_ALPHA_INIT)))

        self.norm = nn.LayerNorm(D_MODEL)
        self.dropout = nn.Dropout(MET_DROPOUT)
        self.encoder = PlainTemporalEncoder(D_MODEL, MET_N_HEAD, MET_D_FF, MET_DROPOUT, MET_N_LAYERS, max_len=SEQ_LEN + 8)

    def forward(
        self,
        met: torch.Tensor,         # [B,L,Fm]
        time_raw: torch.Tensor,    # [B,L,Ft]
        static: torch.Tensor,      # [B,Fs]
        station_id: torch.Tensor,  # [B]
    ) -> torch.Tensor:
        z = self.input_proj(met)
        if self.time_proj is not None and time_raw.shape[-1] > 0:
            z = z + self.alpha_time * self.time_proj(time_raw)
        if self.static_proj is not None and static.shape[-1] > 0:
            z = z + self.alpha_static * self.static_proj(static).unsqueeze(1)
        if self.station_emb is not None:
            z = z + self.alpha_station * self.station_emb(station_id).unsqueeze(1)
        z = self.dropout(self.norm(z))
        return self.encoder(z)


class GatedMeteorologicalCrossAttention(nn.Module):
    """
    用气象条件表示增强水质 SAITS hidden state。

    H_wq 做 Query，H_met 做 Key/Value：
        C_met = CrossAttention(Q=H_wq, K=H_met, V=H_met)
        H_out = LN(H_wq + alpha_met * gate(H_wq, C_met) * C_met)
    """
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=D_MODEL,
            num_heads=MET_CROSS_HEADS,
            dropout=DROPOUT,
            batch_first=True,
        )
        self.gate = nn.Sequential(
            nn.Linear(D_MODEL * 2, D_MODEL),
            nn.Sigmoid(),
        )
        self.alpha_met = nn.Parameter(torch.tensor(float(MET_ALPHA_INIT)))
        self.norm = nn.LayerNorm(D_MODEL)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, h_wq: torch.Tensor, h_met: Optional[torch.Tensor]) -> torch.Tensor:
        if h_met is None:
            return h_wq
        ctx, _ = self.attn(query=h_wq, key=h_met, value=h_met, need_weights=False)
        g = self.gate(torch.cat([h_wq, ctx], dim=-1))
        return self.norm(h_wq + self.alpha_met * self.dropout(g * ctx))


class SpatialContextEncoder(nn.Module):
    """所有站点先做共享 temporal encoder，得到邻站时序表示。"""
    def __init__(self, input_dim: int, time_dim: int, static_dim: int, num_stations: int, num_years: int):
        super().__init__()
        self.input_proj = nn.Linear(input_dim * 2, D_MODEL)
        if SPATIAL_USE_TIME_CONTEXT and time_dim > 0:
            if USE_CALENDAR_TIME_EMBEDDING:
                self.time_proj = CalendarTimeEmbedding(
                    time_feature_names=TIME_FEATURE_NAMES,
                    num_years=num_years,
                    year_dim=YEAR_EMB_DIM,
                    month_dim=MONTH_EMB_DIM,
                    day_dim=DAY_EMB_DIM,
                    hour_dim=HOUR_EMB_DIM,
                    d_model=D_MODEL,
                )
            else:
                self.time_proj = nn.Linear(time_dim, D_MODEL)
        else:
            self.time_proj = None
        self.static_proj = nn.Linear(static_dim, D_MODEL) if (SPATIAL_USE_STATIC_CONTEXT and static_dim > 0) else None
        self.station_emb = nn.Embedding(num_stations, D_MODEL) if SPATIAL_USE_STATION_CONTEXT else None
        self.alpha_time = nn.Parameter(torch.tensor(float(SPATIAL_CONTEXT_ALPHA_INIT)))
        self.alpha_static = nn.Parameter(torch.tensor(float(SPATIAL_CONTEXT_ALPHA_INIT)))
        self.alpha_station = nn.Parameter(torch.tensor(float(SPATIAL_CONTEXT_ALPHA_INIT)))
        self.norm = nn.LayerNorm(D_MODEL)
        self.dropout = nn.Dropout(SPATIAL_DROPOUT)
        self.encoder = PlainTemporalEncoder(D_MODEL, SPATIAL_N_HEAD, SPATIAL_D_FF, SPATIAL_DROPOUT, SPATIAL_N_LAYERS, max_len=SEQ_LEN + 8)

    def forward(self, x_all: torch.Tensor, m_all: torch.Tensor, time_raw: torch.Tensor, static_all: torch.Tensor) -> torch.Tensor:
        # x_all/m_all: [B,L,S,F], time_raw: [B,L,Ft], static_all: [B,S,Fs]
        B, L, S, Fq = x_all.shape
        z = self.input_proj(torch.cat([x_all, m_all], dim=-1))
        if self.time_proj is not None and time_raw.shape[-1] > 0:
            z = z + self.alpha_time * self.time_proj(time_raw).unsqueeze(2)
        if self.static_proj is not None and static_all.shape[-1] > 0:
            z = z + self.alpha_static * self.static_proj(static_all).unsqueeze(1)
        if self.station_emb is not None:
            station_ids = torch.arange(S, device=x_all.device, dtype=torch.long)
            z = z + self.alpha_station * self.station_emb(station_ids).view(1, 1, S, D_MODEL)
        z = self.dropout(self.norm(z))
        z_bs = z.permute(0, 2, 1, 3).contiguous().view(B * S, L, D_MODEL)
        h_bs = self.encoder(z_bs)
        return h_bs.view(B, S, L, D_MODEL).permute(0, 2, 1, 3).contiguous()


class GatedSpatialCrossAttention(nn.Module):
    """目标站 H_wq 作为 Query，邻接矩阵允许的邻站 H_all 作为 Key/Value。"""
    def __init__(self, adj_matrix: np.ndarray):
        super().__init__()
        self.register_buffer('adj', torch.tensor(adj_matrix.astype(np.float32), dtype=torch.float32))
        self.attn = nn.MultiheadAttention(D_MODEL, SPATIAL_CROSS_HEADS, dropout=DROPOUT, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(D_MODEL * 2, D_MODEL), nn.Sigmoid())
        self.alpha_spatial = nn.Parameter(torch.tensor(float(SPATIAL_ALPHA_INIT)))
        self.norm = nn.LayerNorm(D_MODEL)
        self.dropout = nn.Dropout(DROPOUT)

    def _build_key_padding_mask(self, station_id: torch.Tensor, L: int) -> torch.Tensor:
        B = station_id.shape[0]
        S = self.adj.shape[0]
        allowed = self.adj[station_id.long()] > 0.0
        if SPATIAL_EXCLUDE_SELF:
            row = torch.arange(B, device=station_id.device)
            allowed[row, station_id.long()] = False
        if SPATIAL_FALLBACK_TO_ALL:
            no_neighbor = allowed.sum(dim=1) == 0
            if no_neighbor.any():
                allowed[no_neighbor, :] = True
                if SPATIAL_EXCLUDE_SELF:
                    rows = torch.where(no_neighbor)[0]
                    allowed[rows, station_id.long()[rows]] = False
        no_neighbor = allowed.sum(dim=1) == 0
        if no_neighbor.any():
            rows = torch.where(no_neighbor)[0]
            allowed[rows, station_id.long()[rows]] = True
        return (~allowed).unsqueeze(1).expand(B, L, S).reshape(B * L, S)

    def forward(self, h_wq: torch.Tensor, h_all: Optional[torch.Tensor], station_id: torch.Tensor) -> torch.Tensor:
        if h_all is None:
            return h_wq
        B, L, D = h_wq.shape
        S = h_all.shape[2]
        q = h_wq.reshape(B * L, 1, D)
        k = h_all.reshape(B * L, S, D)
        key_padding_mask = self._build_key_padding_mask(station_id, L)
        ctx, _ = self.attn(q, k, k, key_padding_mask=key_padding_mask, need_weights=False)
        ctx = ctx.reshape(B, L, D)
        g = self.gate(torch.cat([h_wq, ctx], dim=-1))
        return self.norm(h_wq + self.alpha_spatial * self.dropout(g * ctx))


class ContextSAITS(nn.Module):
    """
    强 SAITS：两阶段插补 + gate 融合 + 多重 loss。

    与原版 SAITS 的区别只有一个：
    在 stage1/stage2 的 input_proj 之后，额外注入 time/static/gap/station context。
    """
    def __init__(
        self,
        input_dim: int,
        time_dim: int,
        static_dim: int,
        gap_dim: int,
        met_dim: int,
        num_stations: int,
        num_years: int,
        adj_matrix: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.met_dim = int(met_dim)
        self.context = ContextEmbedding(time_dim, static_dim, gap_dim, num_stations, num_years)
        self.met_encoder = MeteorologicalEncoder(met_dim, time_dim, static_dim, num_stations, num_years) if (USE_MET_BRANCH and met_dim > 0) else None
        self.met_cross_1 = GatedMeteorologicalCrossAttention() if self.met_encoder is not None else None
        self.met_cross_2 = GatedMeteorologicalCrossAttention() if self.met_encoder is not None else None

        self.spatial_encoder = None
        self.spatial_cross_1 = None
        self.spatial_cross_2 = None
        if USE_SPATIAL_BRANCH:
            if adj_matrix is None:
                raise ValueError('USE_SPATIAL_BRANCH=True 时必须传入 adj_matrix')
            self.spatial_encoder = SpatialContextEncoder(input_dim, time_dim, static_dim, num_stations, num_years)
            self.spatial_cross_1 = GatedSpatialCrossAttention(adj_matrix)
            self.spatial_cross_2 = GatedSpatialCrossAttention(adj_matrix)

        self.input_proj_1 = nn.Linear(input_dim * 2, D_MODEL)
        self.encoder_1 = DiagonalMaskedEncoder(D_MODEL, N_HEAD, D_FF, DROPOUT, N_LAYERS, max_len=SEQ_LEN + 8)
        self.output_proj_1 = nn.Linear(D_MODEL, input_dim)

        self.input_proj_2 = nn.Linear(input_dim * 2, D_MODEL)
        self.encoder_2 = DiagonalMaskedEncoder(D_MODEL, N_HEAD, D_FF, DROPOUT, N_LAYERS, max_len=SEQ_LEN + 8)
        self.output_proj_2 = nn.Linear(D_MODEL, input_dim)

        self.combine_gate = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x_in: torch.Tensor,       # [B,L,F]
        m_in: torch.Tensor,       # [B,L,F]
        time_raw: torch.Tensor,   # [B,L,Ft]
        static: torch.Tensor,     # [B,Fs]
        gap: torch.Tensor,        # [B,L,Fg]
        station_id: torch.Tensor, # [B]
        met: Optional[torch.Tensor] = None,  # [B,L,Fm]
        x_all: Optional[torch.Tensor] = None, # [B,L,S,F]
        m_all: Optional[torch.Tensor] = None, # [B,L,S,F]
        static_all: Optional[torch.Tensor] = None, # [B,S,Fs]
    ) -> Dict[str, torch.Tensor]:
        h_met = None
        if self.met_encoder is not None:
            if met is None or met.shape[-1] != self.met_dim:
                raise ValueError(f'USE_MET_BRANCH=True 但 met 输入不正确，期望最后一维 {self.met_dim}')
            h_met = self.met_encoder(met, time_raw, static, station_id)

        h_spatial = None
        if self.spatial_encoder is not None:
            if x_all is None or m_all is None or static_all is None:
                raise ValueError('USE_SPATIAL_BRANCH=True 但 x_all/m_all/static_all 输入不完整')
            h_spatial = self.spatial_encoder(x_all, m_all, time_raw, static_all)

        # stage 1
        inp1 = torch.cat([x_in, m_in], dim=-1)
        base1 = self.input_proj_1(inp1)
        h1 = self.encoder_1(self.context(base1, time_raw, static, gap, station_id))
        if self.met_cross_1 is not None:
            h1 = self.met_cross_1(h1, h_met)
        if self.spatial_cross_1 is not None:
            h1 = self.spatial_cross_1(h1, h_spatial, station_id)
        x_tilde_1 = self.output_proj_1(h1)
        x_hat_1 = m_in * x_in + (1.0 - m_in) * x_tilde_1

        # stage 2
        inp2 = torch.cat([x_hat_1, m_in], dim=-1)
        base2 = self.input_proj_2(inp2)
        h2 = self.encoder_2(self.context(base2, time_raw, static, gap, station_id))
        if self.met_cross_2 is not None:
            h2 = self.met_cross_2(h2, h_met)
        if self.spatial_cross_2 is not None:
            h2 = self.spatial_cross_2(h2, h_spatial, station_id)
        x_tilde_2 = self.output_proj_2(h2)
        x_hat_2 = m_in * x_in + (1.0 - m_in) * x_tilde_2

        # gated combination
        gate = self.combine_gate(torch.cat([x_tilde_1, x_tilde_2], dim=-1))
        x_comb = gate * x_tilde_1 + (1.0 - gate) * x_tilde_2
        x_final = m_in * x_in + (1.0 - m_in) * x_comb

        return {
            'imputation_1': x_hat_1,
            'imputation_2': x_hat_2,
            'imputation_final': x_final,
            'reconstruction_raw': x_comb,
        }


def saits_loss(outputs: Dict[str, torch.Tensor], target_x: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    """原版 SAITS 风格多重 loss：stage1 + stage2 + final。"""
    denom = target_mask.sum().clamp_min(1.0)
    l1 = (((outputs['imputation_1'] - target_x) ** 2) * target_mask).sum() / denom
    l2 = (((outputs['imputation_2'] - target_x) ** 2) * target_mask).sum() / denom
    lf = (((outputs['imputation_final'] - target_x) ** 2) * target_mask).sum() / denom
    return 0.5 * l1 + 0.5 * l2 + lf

# ============================================================
# 5. 训练与评估
# ============================================================

def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[optim.Optimizer],
    obs_recon_weight: float = 0.0,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_abs = 0.0
    total_sq = 0.0
    total_y = 0.0
    total_y2 = 0.0
    total_count = 0
    total_art_count = 0
    total_optim_loss = 0.0

    for batch in loader:
        batch = move_batch(batch, device)

        outputs = model(
            x_in=batch['x_in'],
            m_in=batch['m_in'],
            time_raw=batch['time_raw'],
            static=batch['static'],
            gap=batch['gap'],
            station_id=batch['station_id'],
            met=batch['met'],
            x_all=batch.get('x_all'),
            m_all=batch.get('m_all'),
            static_all=batch.get('static_all'),
        )

        x_true = batch['x_true']
        m_art = batch['m_art']
        m_in = batch['m_in']

        loss = saits_loss(outputs, x_true, m_art)
        pred = outputs['imputation_final']
        if is_train and obs_recon_weight > 0:
            # 可选辅助项：用未mask的 reconstruction_raw 轻微约束已观测点重构。
            loss = loss + obs_recon_weight * masked_mse_loss(outputs['reconstruction_raw'], x_true, m_in)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        total_optim_loss += float(loss.item())

        with torch.no_grad():
            count = int(m_art.sum().item())
            total_art_count += count
            if count > 0:
                err = (pred - x_true) * m_art
                y_masked = x_true * m_art
                total_abs += float(err.abs().sum().item())
                total_sq += float((err ** 2).sum().item())
                total_y += float(y_masked.sum().item())
                total_y2 += float(((x_true ** 2) * m_art).sum().item())
                total_count += count

    metrics = finalize_metric_sums(total_abs, total_sq, total_y, total_y2, total_count)
    metrics['optim_loss'] = total_optim_loss / max(len(loader), 1)
    metrics['artificial_count'] = float(total_art_count)
    return metrics


def make_loader(ds: Dataset, shuffle: bool, dynamic_train: bool = False) -> DataLoader:
    # 注意：如果训练集动态造 mask，persistent_workers=True 时 worker 内的 dataset.epoch 不会更新。
    # 所以这里强制关闭 persistent_workers，保证每个 epoch 的随机人工缺失确实变化。
    persistent = bool(PERSISTENT_WORKERS) and int(NUM_WORKERS) > 0
    if dynamic_train and persistent:
        print('[WARN] 训练集动态人工缺失与 persistent_workers=True 冲突，已自动关闭 persistent_workers。')
        persistent = False

    return DataLoader(
        ds,
        batch_size=int(BATCH_SIZE),
        shuffle=shuffle,
        num_workers=int(NUM_WORKERS),
        pin_memory=(device.type == 'cuda'),
        persistent_workers=persistent,
        drop_last=False,
    )



def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    target_mean: Optional[np.ndarray] = None,
    target_std_val: Optional[np.ndarray] = None,
    inverse_transform: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray], Dict[str, float], List[Dict[str, float]]]:
    """
    收集测试/验证集中人工缺失位置的预测和真值。
    返回：
    - y_true_by_target: 长度 F 的 list，每个元素是一维数组
    - y_pred_by_target: 长度 F 的 list，每个元素是一维数组
    - overall_metrics: 所有变量 flatten 后整体指标
    - per_target_metrics: 每个变量单独指标
    """
    model.eval()
    f_count = len(TARGET_NAMES)
    y_true_parts = [[] for _ in range(f_count)]
    y_pred_parts = [[] for _ in range(f_count)]

    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            outputs = model(
                x_in=batch['x_in'],
                m_in=batch['m_in'],
                time_raw=batch['time_raw'],
                static=batch['static'],
                gap=batch['gap'],
                station_id=batch['station_id'],
                met=batch['met'],
                x_all=batch.get('x_all'),
                m_all=batch.get('m_all'),
                static_all=batch.get('static_all'),
            )
            pred = outputs['imputation_final']
            true = batch['x_true']
            m_art = batch['m_art']

            pred_np = pred.detach().cpu().numpy()
            true_np = true.detach().cpu().numpy()
            mask_np = m_art.detach().cpu().numpy() > 0.5

            for f in range(f_count):
                m = mask_np[:, :, f]
                if not np.any(m):
                    continue
                yt = true_np[:, :, f][m]
                yp = pred_np[:, :, f][m]
                if inverse_transform and target_mean is not None and target_std_val is not None:
                    yt = yt * float(target_std_val[f]) + float(target_mean[f])
                    yp = yp * float(target_std_val[f]) + float(target_mean[f])
                y_true_parts[f].append(yt.astype(np.float64))
                y_pred_parts[f].append(yp.astype(np.float64))

    y_true_by_target = []
    y_pred_by_target = []
    for f in range(f_count):
        if y_true_parts[f]:
            y_true_by_target.append(np.concatenate(y_true_parts[f], axis=0))
            y_pred_by_target.append(np.concatenate(y_pred_parts[f], axis=0))
        else:
            y_true_by_target.append(np.array([], dtype=np.float64))
            y_pred_by_target.append(np.array([], dtype=np.float64))

    all_true = np.concatenate([x for x in y_true_by_target if x.size > 0], axis=0) if any(x.size > 0 for x in y_true_by_target) else np.array([])
    all_pred = np.concatenate([x for x in y_pred_by_target if x.size > 0], axis=0) if any(x.size > 0 for x in y_pred_by_target) else np.array([])
    overall_metrics = compute_metrics_np(all_true, all_pred)

    per_target_metrics = []
    for name, yt, yp in zip(TARGET_NAMES, y_true_by_target, y_pred_by_target):
        row = compute_metrics_np(yt, yp)
        row['target'] = name
        per_target_metrics.append(row)

    return y_true_by_target, y_pred_by_target, overall_metrics, per_target_metrics


def save_metrics_csv(rows: List[Dict[str, float]], path: Path, fieldnames: Sequence[str]):
    import csv
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in fieldnames})


def plot_test_scatter_by_target(
    y_true_by_target: List[np.ndarray],
    y_pred_by_target: List[np.ndarray],
    output_dir: Path,
    prefix: str = 'test_scatter',
    original_scale: bool = False,
):
    plot_dir = output_dir / 'scatter_plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED + 12345)
    scale_note = 'original scale' if original_scale else 'standardized scale'

    for name, yt, yp in zip(TARGET_NAMES, y_true_by_target, y_pred_by_target):
        if yt.size == 0:
            print(f'[WARN] {name} 没有可绘制的人工缺失测试点，跳过散点图。')
            continue

        n = yt.size
        if n > MAX_SCATTER_POINTS_PER_TARGET:
            idx = rng.choice(np.arange(n), size=MAX_SCATTER_POINTS_PER_TARGET, replace=False)
            yt_plot = yt[idx]
            yp_plot = yp[idx]
        else:
            yt_plot = yt
            yp_plot = yp

        metrics = compute_metrics_np(yt, yp)
        lo = float(np.nanmin([np.nanmin(yt_plot), np.nanmin(yp_plot)]))
        hi = float(np.nanmax([np.nanmax(yt_plot), np.nanmax(yp_plot)]))
        if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
            lo, hi = -1.0, 1.0
        pad = 0.05 * (hi - lo)
        lo -= pad
        hi += pad

        plt.figure(figsize=(6, 6))
        plt.scatter(yp_plot, yt_plot, s=8, alpha=0.35)
        plt.plot([lo, hi], [lo, hi], linestyle='--', linewidth=1)
        plt.xlim(lo, hi)
        plt.ylim(lo, hi)
        plt.xlabel(f'Predicted ({scale_note})')
        plt.ylabel(f'Observed / True ({scale_note})')
        plt.title(
            f'{name}\nRMSE={metrics["rmse"]:.4f}, MAE={metrics["mae"]:.4f}, '
            f'R2={metrics["r2"]:.4f}, NSE={metrics["nse"]:.4f}, n={metrics["count"]}'
        )
        plt.tight_layout()
        out_path = plot_dir / f'{prefix}_{safe_filename(name)}.png'
        plt.savefig(out_path, dpi=200)
        plt.close()


def evaluate_test_by_block_lengths(
    model: nn.Module,
    x_test: np.ndarray,
    m_test: np.ndarray,
    t_test: np.ndarray,
    static_std: np.ndarray,
    met_test: np.ndarray,
    output_dir: Path,
    arrays: PreparedArrays,
) -> List[Dict[str, float]]:
    rows = []
    for length in [int(x) for x in TEST_EVAL_BLOCK_LENGTHS]:
        try:
            ds = WQWindowDataset(
                x_test,
                m_test,
                t_test,
                static_std,
                met_test,
                mode='test',
                block_lengths_override=[length],
                num_blocks_override=TEST_NUM_BLOCKS_PER_SAMPLE,
                base_seed_override=TEST_EVAL_BLOCK_SEED + int(length) * 1009,
            )
            if len(ds) == 0:
                raise RuntimeError('没有可用窗口')
            loader = make_loader(ds, shuffle=False, dynamic_train=False)
            _, _, overall, per_target = collect_predictions(
                model,
                loader,
                target_mean=arrays.target_mean,
                target_std_val=arrays.target_std_val,
                inverse_transform=EVAL_IN_ORIGINAL_SCALE,
            )
            row = {'block_length': length, **overall}
            rows.append(row)
            print(
                f"[TEST-L{length}] loss {overall['loss']:.6f} | mae {overall['mae']:.6f} | "
                f"rmse {overall['rmse']:.6f} | r2 {overall['r2']:.6f} | nse {overall['nse']:.6f} | "
                f"count {int(overall['count'])}"
            )

            # 每个长度也保存一份五个水质指标的分项结果，便于后续分析。
            per_rows = []
            for r in per_target:
                rr = {'block_length': length, **r}
                per_rows.append(rr)
            save_metrics_csv(
                per_rows,
                output_dir / f'test_metrics_by_target_L{length}.csv',
                fieldnames=['block_length', 'target', 'loss', 'mae', 'rmse', 'r2', 'nse', 'count'],
            )
        except Exception as exc:
            print(f'[WARN] 测试缺失长度 {length} 评估失败：{exc}')
            rows.append({'block_length': length, 'loss': float('nan'), 'mae': float('nan'), 'rmse': float('nan'), 'r2': float('nan'), 'nse': float('nan'), 'count': 0})

    save_metrics_csv(
        rows,
        output_dir / 'test_metrics_by_block_length.csv',
        fieldnames=['block_length', 'loss', 'mae', 'rmse', 'r2', 'nse', 'count'],
    )
    with open(output_dir / 'test_metrics_by_block_length.json', 'w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    return rows

def main():
    arrays = prepare_arrays()
    adj_matrix = None
    if USE_SPATIAL_BRANCH:
        adj_matrix = load_adjacency_matrix(ADJ_PATH, expected_size=arrays.target_true_std.shape[1])
        print(f'[INFO] adjacency shape: {adj_matrix.shape}, edge_count={int(adj_matrix.sum())}')

    # 按时间切分
    x_train, x_val, x_test = split_by_time(arrays.target_true_std)
    m_train, m_val, m_test = split_by_time(arrays.m_orig)
    t_train, t_val, t_test = split_by_time(arrays.time_raw)
    met_train, met_val, met_test = split_by_time(arrays.met_std)

    train_ds = WQWindowDataset(x_train, m_train, t_train, arrays.static_std, met_train, mode='train')
    val_ds = WQWindowDataset(x_val, m_val, t_val, arrays.static_std, met_val, mode='val')

    if len(train_ds) == 0:
        raise RuntimeError('训练集没有可用窗口。请降低 MIN_WINDOW_OBSERVED_RATIO 或检查 SEQ_LEN/数据长度。')
    if len(val_ds) == 0:
        raise RuntimeError('验证集没有可用窗口。请降低 MIN_WINDOW_OBSERVED_RATIO 或检查 SEQ_LEN/数据长度。')

    train_loader = make_loader(train_ds, shuffle=True, dynamic_train=True)
    val_loader = make_loader(val_ds, shuffle=False, dynamic_train=False)

    # 测试集可能因为时间太短或 SEQ_LEN 太长而不可用，这里做兼容。
    test_loader = None
    try:
        test_ds = WQWindowDataset(x_test, m_test, t_test, arrays.static_std, met_test, mode='test')
        if len(test_ds) > 0:
            test_loader = make_loader(test_ds, shuffle=False, dynamic_train=False)
    except Exception as exc:
        print(f'[WARN] 测试集暂不评估：{exc}')

    if USE_CALENDAR_TIME_EMBEDDING and len(TIME_FEATURE_NAMES) > 0:
        year_idx = TIME_FEATURE_NAMES.index('year_id') if 'year_id' in TIME_FEATURE_NAMES else None
        if NUM_YEARS is not None:
            num_years = int(NUM_YEARS)
        elif year_idx is not None:
            num_years = int(np.nanmax(arrays.time_raw[..., year_idx])) + 1
        else:
            num_years = 1
        num_years = max(num_years, 1)
    else:
        num_years = 1
    print(f'[INFO] calendar num_years: {num_years}')

    model = ContextSAITS(
        input_dim=len(TARGET_NAMES),
        time_dim=len(TIME_FEATURE_NAMES),
        static_dim=len(STATIC_FEATURE_NAMES),
        gap_dim=len(TARGET_NAMES) * 3,
        met_dim=arrays.met_std.shape[-1],
        num_stations=arrays.target_true_std.shape[1],
        num_years=num_years,
        adj_matrix=adj_matrix,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=float(LEARNING_RATE), weight_decay=float(WEIGHT_DECAY))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    output_dir = Path(OUTPUT_DIR)
    shutil.copy2(CONFIG_PATH, output_dir / 'used_config.yaml')

    stats = {
        'target_names': TARGET_NAMES,
        'target_mean': arrays.target_mean.tolist(),
        'target_std': arrays.target_std_val.tolist(),
        'time_feature_names': TIME_FEATURE_NAMES,
        'static_feature_names': STATIC_FEATURE_NAMES,
        'static_mean': arrays.static_mean.tolist(),
        'static_std': arrays.static_std_val.tolist(),
        'meteorological_feature_names': METEOROLOGICAL_FEATURE_NAMES,
        'use_spatial_branch': USE_SPATIAL_BRANCH,
        'adj_path': ADJ_PATH,
        'spatial_note': '空间分支使用邻接矩阵约束的 spatial cross-attention；目标站作为 Query，邻站作为 Key/Value。',
        'met_mean': arrays.met_mean.tolist(),
        'met_std': arrays.met_std_val.tolist(),
        'note': '预测值处于标准化空间；反标准化用 target_mean 和 target_std。气象特征按训练集均值方差标准化。',
    }
    with open(output_dir / 'normalization_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    best_rmse = float('inf')
    best_epoch = -1
    history = []

    for epoch in range(1, int(MAX_EPOCHS) + 1):
        train_ds.set_epoch(epoch)

        train_metrics = run_one_epoch(
            model,
            train_loader,
            optimizer,
            obs_recon_weight=float(OBS_RECON_LOSS_WEIGHT),
        )
        val_metrics = run_one_epoch(model, val_loader, optimizer=None, obs_recon_weight=0.0)
        scheduler.step(val_metrics['rmse'])

        row = {
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_mae': train_metrics['mae'],
            'train_rmse': train_metrics['rmse'],
            'train_r2': train_metrics['r2'],
            'train_nse': train_metrics['nse'],
            'train_artificial_count': train_metrics['artificial_count'],
            'val_loss': val_metrics['loss'],
            'val_mae': val_metrics['mae'],
            'val_rmse': val_metrics['rmse'],
            'val_r2': val_metrics['r2'],
            'val_nse': val_metrics['nse'],
            'val_artificial_count': val_metrics['artificial_count'],
            'lr': optimizer.param_groups[0]['lr'],
            'alpha_time': float(model.context.alpha_time.detach().cpu()),
            'alpha_static': float(model.context.alpha_static.detach().cpu()),
            'alpha_gap': float(model.context.alpha_gap.detach().cpu()),
            'alpha_station': float(model.context.alpha_station.detach().cpu()),
            'alpha_met_stage1': float(model.met_cross_1.alpha_met.detach().cpu()) if getattr(model, 'met_cross_1', None) is not None else float('nan'),
            'alpha_met_stage2': float(model.met_cross_2.alpha_met.detach().cpu()) if getattr(model, 'met_cross_2', None) is not None else float('nan'),
            'alpha_spatial_stage1': float(model.spatial_cross_1.alpha_spatial.detach().cpu()) if getattr(model, 'spatial_cross_1', None) is not None else float('nan'),
            'alpha_spatial_stage2': float(model.spatial_cross_2.alpha_spatial.detach().cpu()) if getattr(model, 'spatial_cross_2', None) is not None else float('nan'),
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {row['train_loss']:.6f} rmse {row['train_rmse']:.6f} "
            f"r2 {row['train_r2']:.6f} nse {row['train_nse']:.6f} | "
            f"val loss {row['val_loss']:.6f} rmse {row['val_rmse']:.6f} "
            f"r2 {row['val_r2']:.6f} nse {row['val_nse']:.6f} | "
            f"art_count train/val {int(row['train_artificial_count'])}/{int(row['val_artificial_count'])} | "
            f"alpha t/s/g/st {row['alpha_time']:.3f}/{row['alpha_static']:.3f}/{row['alpha_gap']:.3f}/{row['alpha_station']:.3f} | "
            f"alpha_met {row['alpha_met_stage1']:.3f}/{row['alpha_met_stage2']:.3f} | "
            f"alpha_sp {row['alpha_spatial_stage1']:.3f}/{row['alpha_spatial_stage2']:.3f} | "
            f"lr {row['lr']:.2e}"
        )

        with open(output_dir / 'history.json', 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        if val_metrics['rmse'] < best_rmse:
            best_rmse = val_metrics['rmse']
            best_epoch = epoch
            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_rmse': best_rmse,
                'config': cfg,
                'stats': stats,
            }
            torch.save(ckpt, output_dir / 'best_model.pt')
            print(f'[INFO] saved best_model.pt at epoch {epoch}, val_rmse={best_rmse:.6f}')

        if epoch - best_epoch >= int(PATIENCE):
            print(f'[INFO] early stopping at epoch {epoch}. best_epoch={best_epoch}, best_rmse={best_rmse:.6f}')
            break

    # 用 best model 在 test 上评估一次；并画五个水质指标的预测-真实散点图。
    if test_loader is not None and (output_dir / 'best_model.pt').exists():
        ckpt = torch.load(output_dir / 'best_model.pt', map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])

        test_metrics = run_one_epoch(model, test_loader, optimizer=None, obs_recon_weight=0.0)
        print(
            f"[TEST] loss {test_metrics['loss']:.6f} | "
            f"mae {test_metrics['mae']:.6f} | rmse {test_metrics['rmse']:.6f} | "
            f"r2 {test_metrics['r2']:.6f} | nse {test_metrics['nse']:.6f} | "
            f"art_count {int(test_metrics['artificial_count'])}"
        )
        with open(output_dir / 'test_metrics.json', 'w', encoding='utf-8') as f:
            json.dump(test_metrics, f, ensure_ascii=False, indent=2)

        y_true_by_target, y_pred_by_target, overall, per_target = collect_predictions(
            model,
            test_loader,
            target_mean=arrays.target_mean,
            target_std_val=arrays.target_std_val,
            inverse_transform=EVAL_IN_ORIGINAL_SCALE,
        )
        with open(output_dir / 'test_metrics_overall_from_predictions.json', 'w', encoding='utf-8') as f:
            json.dump(overall, f, ensure_ascii=False, indent=2)
        save_metrics_csv(
            per_target,
            output_dir / 'test_metrics_by_target.csv',
            fieldnames=['target', 'loss', 'mae', 'rmse', 'r2', 'nse', 'count'],
        )
        plot_test_scatter_by_target(
            y_true_by_target,
            y_pred_by_target,
            output_dir=output_dir,
            prefix='test_scatter_mixed',
            original_scale=EVAL_IN_ORIGINAL_SCALE,
        )

        # 按 3/6/9/12/18/42/180 等不同连续缺失长度单独评估测试集。
        evaluate_test_by_block_lengths(
            model=model,
            x_test=x_test,
            m_test=m_test,
            t_test=t_test,
            static_std=arrays.static_std,
            met_test=met_test,
            output_dir=output_dir,
            arrays=arrays,
        )

    print('[DONE] best_epoch:', best_epoch, 'best_val_rmse:', best_rmse)
    print('[DONE] output_dir:', OUTPUT_DIR)


if __name__ == '__main__':
    main()