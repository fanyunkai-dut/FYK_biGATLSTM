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
CONFIG_PATH = "configs.yaml"

with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config_all = yaml.safe_load(f)

cfg = config_all.get('GATLSTM_block_training', {})

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
DT_SUFFIX = cfg.get('DT_SUFFIX', '_dt')
SHARED_BLOCK_LEN_NAME = cfg.get('shared_block_len_name', 'wq_block_len_shared')
SHARED_BLOCK_POS_NAME = cfg.get('shared_block_pos_name', 'wq_block_pos_shared')

# ---------- 大块遮挡配置 ----------
block_cfg = cfg.get('block_masking', {})
ENABLE_TRAIN_BLOCK_MASKING = block_cfg.get('enable_train_block_masking', True)
TRAIN_BLOCK_LENGTHS = block_cfg.get('train_block_lengths', [3, 6, 12, 18])
TRAIN_NUM_BLOCKS_PER_STATION = block_cfg.get('train_num_blocks_per_station', 2)
TRAIN_BLOCK_SEED = block_cfg.get('train_block_seed', 42)

ENABLE_BLOCKED_TEST = block_cfg.get('enable_blocked_test', True)
TEST_BLOCK_LENGTHS = block_cfg.get('test_block_lengths', [3, 6, 12, 18])
TEST_NUM_BLOCKS_PER_STATION = block_cfg.get('test_num_blocks_per_station', 1)
TEST_BLOCK_SEED = block_cfg.get('test_block_seed', 123)


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


# ==================== 工具函数：大块遮挡 ====================
def compute_dt_since_last_obs(mask_1d):
    dt = np.zeros_like(mask_1d, dtype=np.float32)
    gap = 0.0
    for t in range(len(mask_1d)):
        if mask_1d[t] > 0.5:
            gap = 0.0
            dt[t] = 0.0
        else:
            gap += 1.0
            dt[t] = gap
    return dt


def compute_block_len_pos_from_missing(missing_1d):
    """
    missing_1d: bool/0-1, 1 表示属于共享缺失块（当前站点时刻五个目标变量全部缺失）
    返回：
        block_len_1d: 缺失块长度，仅在共享缺失块位置非0
        block_pos_1d: 块内相对位置[0,1]，仅在共享缺失块位置非0
    """
    missing_1d = np.asarray(missing_1d).astype(bool)
    T = len(missing_1d)
    block_len = np.zeros(T, dtype=np.float32)
    block_pos = np.zeros(T, dtype=np.float32)

    t = 0
    while t < T:
        if not missing_1d[t]:
            t += 1
            continue

        start = t
        while t < T and missing_1d[t]:
            t += 1
        end = t
        L = end - start

        block_len[start:end] = float(L)
        if L == 1:
            block_pos[start:end] = 0.5
        else:
            block_pos[start:end] = np.linspace(0.0, 1.0, L, dtype=np.float32)

    return block_len, block_pos


def get_history_feature_indices(
    input_feature_names,
    target_names,
    mask_suffix='_mask',
    dt_suffix='_dt',
    shared_block_len_name='wq_block_len_shared',
    shared_block_pos_name='wq_block_pos_shared'
):
    """
    在输入特征里找到：
        - 每个目标变量本身
        - 每个目标变量对应 mask
        - 每个目标变量对应 dt
        - 可选的共享 block_len / block_pos
    """
    value_indices = []
    mask_indices = []
    dt_indices = []

    for name in target_names:
        if name not in input_feature_names:
            raise ValueError(f"做大块遮挡时，输入特征中缺少历史变量: {name}")

        mask_name = name + mask_suffix
        dt_name = name + dt_suffix

        if mask_name not in input_feature_names:
            raise ValueError(f"做大块遮挡时，输入特征中缺少 mask: {mask_name}")
        if dt_name not in input_feature_names:
            raise ValueError(f"做大块遮挡时，输入特征中缺少 dt: {dt_name}")

        value_indices.append(input_feature_names.index(name))
        mask_indices.append(input_feature_names.index(mask_name))
        dt_indices.append(input_feature_names.index(dt_name))

    shared_len_idx = input_feature_names.index(shared_block_len_name) if shared_block_len_name in input_feature_names else None
    shared_pos_idx = input_feature_names.index(shared_block_pos_name) if shared_block_pos_name in input_feature_names else None

    if (shared_len_idx is None) ^ (shared_pos_idx is None):
        raise ValueError(
            f"共享 block 特征不完整：{shared_block_len_name} / {shared_block_pos_name} 必须同时存在或同时不存在"
        )

    return value_indices, mask_indices, dt_indices, shared_len_idx, shared_pos_idx


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


def apply_block_masking_to_x(
    x_data,
    input_feature_names,
    target_names,
    mask_suffix='_mask',
    dt_suffix='_dt',
    block_lengths=(3, 6, 12, 18),
    num_blocks_per_station=2,
    seed=42,
    shared_block_len_name='wq_block_len_shared',
    shared_block_pos_name='wq_block_pos_shared'
):
    """
    对输入序列 x_data 做单站点大块连续遮挡。
    遮挡逻辑：
        - 单站点
        - 五个目标变量同时遮挡
        - 同步更新 value / mask / dt
        - 若存在共享 block 特征，则按“五个目标变量全部缺失”重算 shared_block_len / shared_block_pos
    返回：
        x_blocked: (T, N, F_in)
        artificial_block_map: (T, N) bool，表示该站点该时刻是否被人工遮挡
        block_summary: dict
    """
    rng = np.random.default_rng(seed)
    x_blocked = x_data.copy()
    T_local, N_local, _ = x_blocked.shape

    value_idxs, mask_idxs, dt_idxs, shared_len_idx, shared_pos_idx = get_history_feature_indices(
        input_feature_names,
        target_names,
        mask_suffix,
        dt_suffix,
        shared_block_len_name,
        shared_block_pos_name
    )

    artificial_block_map = np.zeros((T_local, N_local), dtype=bool)
    station_block_counts = []

    for s in range(N_local):
        station_masks = np.stack([x_blocked[:, s, idx] for idx in mask_idxs], axis=1)
        valid_all = np.all(station_masks > 0.5, axis=1)

        blocks = choose_non_overlapping_blocks(
            valid_all=valid_all,
            candidate_lengths=block_lengths,
            num_blocks=num_blocks_per_station,
            rng=rng
        )

        for start, end in blocks:
            x_blocked[start:end, s, value_idxs] = np.nan
            x_blocked[start:end, s, mask_idxs] = 0.0
            artificial_block_map[start:end, s] = True

        # 遮挡后重新计算每个变量的 dt
        for j, dt_idx in enumerate(dt_idxs):
            cur_mask = x_blocked[:, s, mask_idxs[j]]
            x_blocked[:, s, dt_idx] = compute_dt_since_last_obs(cur_mask)

        # 遮挡后重新计算共享缺失块特征：仅当五个目标变量同时缺失时 shared block 才成立
        if shared_len_idx is not None and shared_pos_idx is not None:
            station_masks_after = np.stack([x_blocked[:, s, idx] for idx in mask_idxs], axis=1)
            shared_missing = np.all(station_masks_after < 0.5, axis=1)
            block_len, block_pos = compute_block_len_pos_from_missing(shared_missing)
            x_blocked[:, s, shared_len_idx] = block_len
            x_blocked[:, s, shared_pos_idx] = block_pos

        station_block_counts.append(len(blocks))

    block_summary = {
        "total_blocks": int(np.sum(station_block_counts)),
        "mean_blocks_per_station": float(np.mean(station_block_counts)),
        "station_block_counts": station_block_counts,
        "total_blocked_points": int(np.sum(artificial_block_map)),
    }

    return x_blocked, artificial_block_map, block_summary


def build_affected_target_map(artificial_block_map, seq_len):
    T_local, N_local = artificial_block_map.shape
    num_samples = T_local - seq_len
    affected = np.zeros((num_samples, N_local), dtype=np.float32)

    for t in range(seq_len, T_local):
        affected[t - seq_len] = np.any(artificial_block_map[t - seq_len:t, :], axis=0).astype(np.float32)

    return affected


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

x_full = full_data[:, :, INPUT_IDXS]
y_full = full_data[:, :, TARGET_IDXS]
mask_full = full_data[:, :, MASK_IDXS]

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

# ==================== 训练集：人工大块遮挡 ====================
if ENABLE_TRAIN_BLOCK_MASKING:
    print("\n对训练集输入做人工大块连续遮挡...")
    train_x_raw, train_block_map, train_block_summary = apply_block_masking_to_x(
        x_data=train_x_raw,
        input_feature_names=INPUT_FEATURE_NAMES,
        target_names=TARGET_NAMES,
        mask_suffix=MASK_SUFFIX,
        dt_suffix=DT_SUFFIX,
        block_lengths=TRAIN_BLOCK_LENGTHS,
        num_blocks_per_station=TRAIN_NUM_BLOCKS_PER_STATION,
        seed=TRAIN_BLOCK_SEED,
        shared_block_len_name=SHARED_BLOCK_LEN_NAME,
        shared_block_pos_name=SHARED_BLOCK_POS_NAME
    )
    print("训练遮挡摘要:", train_block_summary)
else:
    train_block_map = np.zeros((train_x_raw.shape[0], train_x_raw.shape[1]), dtype=bool)

# ==================== 构造 blocked test 原始版本 ====================
blocked_test_raw_dict = {}
if ENABLE_BLOCKED_TEST:
    print("\n构造人工遮挡测试集...")
    for i, L in enumerate(TEST_BLOCK_LENGTHS):
        cur_seed = TEST_BLOCK_SEED + i
        x_blocked_test_raw, test_block_map, test_block_summary = apply_block_masking_to_x(
            x_data=test_x_raw,
            input_feature_names=INPUT_FEATURE_NAMES,
            target_names=TARGET_NAMES,
            mask_suffix=MASK_SUFFIX,
            dt_suffix=DT_SUFFIX,
            block_lengths=[L],
            num_blocks_per_station=TEST_NUM_BLOCKS_PER_STATION,
            seed=cur_seed,
            shared_block_len_name=SHARED_BLOCK_LEN_NAME,
            shared_block_pos_name=SHARED_BLOCK_POS_NAME
        )
        blocked_test_raw_dict[L] = {
            "x_raw": x_blocked_test_raw,
            "block_map": test_block_map,
            "summary": test_block_summary
        }
        print(f"测试遮挡长度 {L} 步摘要: {test_block_summary}")

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

print(f"\nX 中需要标准化的特征 ({len(standardize_names_used)} 个): {standardize_names_used}")


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


x_means, x_stds, standardize_names_used = fit_x_scaler_nanaware(train_x_raw, norm_indices, INPUT_FEATURE_NAMES)


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


def apply_standardization_nanaware(data, norm_indices, means, stds):
    data_norm = data.copy()
    for idx, mean, std in zip(norm_indices, means, stds):
        valid = ~np.isnan(data_norm[..., idx])
        data_norm[..., idx][valid] = (data_norm[..., idx][valid] - mean) / std
    return data_norm


def apply_target_standardization_nanaware(y_data, mask_data, means, stds):
    y_norm = y_data.copy()
    for i, (mean, std) in enumerate(zip(means, stds)):
        valid = (mask_data[..., i] > 0.5) & (~np.isnan(y_norm[..., i]))
        y_norm[..., i][valid] = (y_norm[..., i][valid] - mean) / std
    return y_norm


train_x = apply_standardization_nanaware(train_x_raw, norm_indices, x_means, x_stds)
val_x   = apply_standardization_nanaware(val_x_raw,   norm_indices, x_means, x_stds)
test_x  = apply_standardization_nanaware(test_x_raw,  norm_indices, x_means, x_stds)

blocked_test_x_dict = {}
for L, info in blocked_test_raw_dict.items():
    blocked_test_x_dict[L] = apply_standardization_nanaware(info["x_raw"], norm_indices, x_means, x_stds)

train_y = apply_target_standardization_nanaware(train_y_raw, train_mask_raw, y_means, y_stds)
val_y   = apply_target_standardization_nanaware(val_y_raw,   val_mask_raw,   y_means, y_stds)
test_y  = apply_target_standardization_nanaware(test_y_raw,  test_mask_raw,  y_means, y_stds)

print("X 与 y 标准化完成（目前仍保留 NaN）。")


def replace_nan_with_zero(arr):
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


train_x = replace_nan_with_zero(train_x)
val_x   = replace_nan_with_zero(val_x)
test_x  = replace_nan_with_zero(test_x)

for L in blocked_test_x_dict:
    blocked_test_x_dict[L] = replace_nan_with_zero(blocked_test_x_dict[L])

train_y = replace_nan_with_zero(train_y)
val_y   = replace_nan_with_zero(val_y)
test_y  = replace_nan_with_zero(test_y)

print("已将标准化后的 X / y 中所有 NaN 统一替换为 0。")

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


def build_samples(x_data, y_data, mask_data, seq_len):
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
print("构建自然测试集样本...")
test_X, test_y_samples, test_mask = build_samples(test_x, test_y, test_mask_raw, SEQ_LEN)

blocked_test_dataset_dict = {}
for L, x_blocked_std in blocked_test_x_dict.items():
    X_b, y_b, mask_b = build_samples(x_blocked_std, test_y, test_mask_raw, SEQ_LEN)
    affected_b = build_affected_target_map(blocked_test_raw_dict[L]["block_map"], SEQ_LEN)
    blocked_test_dataset_dict[L] = {
        "X": X_b,
        "y": y_b,
        "mask": mask_b,
        "affected": affected_b
    }

print(f"训练样本数: {train_X.shape[0]}, 验证样本数: {val_X.shape[0]}, 自然测试样本数: {test_X.shape[0]}")
print(f"X 形状: {train_X.shape}, y 形状: {train_y_samples.shape}, mask 形状: {train_mask.shape}")

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

blocked_test_loader_dict = {}
for L, ds in blocked_test_dataset_dict.items():
    cur_dataset = TensorDataset(
        torch.tensor(ds["X"], dtype=torch.float32),
        torch.tensor(ds["y"], dtype=torch.float32),
        torch.tensor(ds["mask"], dtype=torch.float32)
    )
    blocked_test_loader_dict[L] = DataLoader(cur_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin_memory)

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

print("\n开始训练...")
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


def run_prediction(loader):
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
    return pred_concat, target_concat, mask_concat


def compute_metrics(pred, target, mask, affected=None):
    valid = mask > 0.5

    if affected is not None:
        if affected.ndim == 2:
            affected = affected[:, :, None]
        valid = valid & (affected > 0.5)

    if np.sum(valid) == 0:
        return None

    pred_valid = pred[valid]
    target_valid = target[valid]

    r2 = r2_score(target_valid, pred_valid)
    rmse = np.sqrt(mean_squared_error(target_valid, pred_valid))
    mae = mean_absolute_error(target_valid, pred_valid)

    return {
        "R2": float(r2),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "n_valid": int(np.sum(valid))
    }


def save_prediction_triplet(output_dir, prefix, pred, target, mask):
    np.save(os.path.join(output_dir, f'{prefix}_pred.npy'), pred)
    np.save(os.path.join(output_dir, f'{prefix}_target.npy'), target)
    np.save(os.path.join(output_dir, f'{prefix}_mask.npy'), mask)


print("\n加载最佳模型进行测试...")
model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_model.pth'), map_location=device))
model.eval()

test_pred, test_target, test_mask_eval = run_prediction(test_loader)
save_prediction_triplet(OUTPUT_DIR, 'test', test_pred, test_target, test_mask_eval)

natural_metrics = compute_metrics(test_pred, test_target, test_mask_eval, affected=None)

print("====== 自然测试集性能（overall）======")
print(f"R2: {natural_metrics['R2']:.4f}")
print(f"RMSE: {natural_metrics['RMSE']:.4f}")
print(f"MAE: {natural_metrics['MAE']:.4f}")
print(f"n_valid: {natural_metrics['n_valid']}")


def save_predictions(loader, name):
    pred_concat, target_concat, mask_concat = run_prediction(loader)
    save_prediction_triplet(OUTPUT_DIR, name, pred_concat, target_concat, mask_concat)
    print(f"{name} 集预测结果已保存到 {OUTPUT_DIR}")


save_predictions(train_loader, 'train')
save_predictions(val_loader, 'val')

blocked_metrics_all = {}

if ENABLE_BLOCKED_TEST:
    print("\n====== 人工大块遮挡测试（blocked-only）======")
    for L, loader in blocked_test_loader_dict.items():
        pred_b, target_b, mask_b = run_prediction(loader)
        affected_b = blocked_test_dataset_dict[L]["affected"]

        save_prediction_triplet(OUTPUT_DIR, f'test_block{L}', pred_b, target_b, mask_b)
        np.save(os.path.join(OUTPUT_DIR, f'test_block{L}_affected.npy'), affected_b)

        metrics_blocked = compute_metrics(pred_b, target_b, mask_b, affected=affected_b)
        blocked_metrics_all[L] = metrics_blocked

        if metrics_blocked is None:
            print(f"[block={L}] 没有可评估的 blocked-only 样本")
        else:
            print(
                f"[block={L}] R2={metrics_blocked['R2']:.4f}, "
                f"RMSE={metrics_blocked['RMSE']:.4f}, "
                f"MAE={metrics_blocked['MAE']:.4f}, "
                f"n_valid={metrics_blocked['n_valid']}"
            )

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
    f.write(f"SHARED_BLOCK_LEN_NAME: {SHARED_BLOCK_LEN_NAME}\n")
    f.write(f"SHARED_BLOCK_POS_NAME: {SHARED_BLOCK_POS_NAME}\n")
    f.write("\n========== Block Masking ==========\n")
    f.write(f"ENABLE_TRAIN_BLOCK_MASKING: {ENABLE_TRAIN_BLOCK_MASKING}\n")
    f.write(f"TRAIN_BLOCK_LENGTHS: {TRAIN_BLOCK_LENGTHS}\n")
    f.write(f"TRAIN_NUM_BLOCKS_PER_STATION: {TRAIN_NUM_BLOCKS_PER_STATION}\n")
    f.write(f"TRAIN_BLOCK_SEED: {TRAIN_BLOCK_SEED}\n")
    f.write(f"ENABLE_BLOCKED_TEST: {ENABLE_BLOCKED_TEST}\n")
    f.write(f"TEST_BLOCK_LENGTHS: {TEST_BLOCK_LENGTHS}\n")
    f.write(f"TEST_NUM_BLOCKS_PER_STATION: {TEST_NUM_BLOCKS_PER_STATION}\n")
    f.write(f"TEST_BLOCK_SEED: {TEST_BLOCK_SEED}\n")
    f.write("===================================\n\n")

    f.write("========== Natural Test Metrics ==========\n")
    f.write(f"R2: {natural_metrics['R2']:.4f}\n")
    f.write(f"RMSE: {natural_metrics['RMSE']:.4f}\n")
    f.write(f"MAE: {natural_metrics['MAE']:.4f}\n")
    f.write(f"n_valid: {natural_metrics['n_valid']}\n")
    f.write("==========================================\n\n")

    if ENABLE_BLOCKED_TEST:
        f.write("========== Blocked Test Metrics (blocked-only) ==========\n")
        for L in sorted(blocked_metrics_all.keys()):
            m = blocked_metrics_all[L]
            if m is None:
                f.write(f"block={L}: no valid blocked-only samples\n")
            else:
                f.write(
                    f"block={L}: R2={m['R2']:.4f}, RMSE={m['RMSE']:.4f}, "
                    f"MAE={m['MAE']:.4f}, n_valid={m['n_valid']}\n"
                )
        f.write("========================================================\n")

valid_idx = test_mask_eval > 0.5
pred_valid = test_pred[valid_idx]
target_valid = test_target[valid_idx]

plt.figure(figsize=(6, 6))
plt.scatter(target_valid, pred_valid, alpha=0.3, s=1)
plt.plot(
    [target_valid.min(), target_valid.max()],
    [target_valid.min(), target_valid.max()],
    'r--', lw=2
)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title(
    f'Natural Test Overall\n'
    f'R2={natural_metrics["R2"]:.3f}, '
    f'RMSE={natural_metrics["RMSE"]:.3f}, '
    f'MAE={natural_metrics["MAE"]:.3f}'
)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'scatter_natural_test_overall.png'), dpi=150)
plt.close()

print(f"\n散点图已保存到 {OUTPUT_DIR}")

