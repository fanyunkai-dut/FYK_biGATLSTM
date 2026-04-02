"""
生成四种缺失模式的测试集（预处理后数据）
使用方法：在代码开头设置 MODE = 1/2/3/4，运行后生成对应模式的测试数据。
输出文件：
    test_data_modeX.npy          : 预处理后的完整测试数据 (T_test, N, F)
    test_original_mask_modeX.npy : 原始掩码（注入人为缺失前） (T_test, N, F_wq)
    test_final_mask_modeX.npy    : 最终掩码（注入人为缺失后） (T_test, N, F_wq)
"""

import numpy as np
import pandas as pd
import yaml
import pickle
import os
import random
from sklearn.preprocessing import StandardScaler

# ==================== 配置区 ====================
CONFIG_PATH = "/home/fanyunkai/FYK_GCNLSTM/configs.yaml"  # 请根据实际路径修改
MODE = 3  # 1: 随机缺失25%  2: 独立连续缺失  3: 站内同步连续缺失  4: 全流域同步连续缺失
RANDOM_SEED = 42  # 用于缺失注入的可复现性

# 加载配置
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config_all = yaml.safe_load(f)

cfg = config_all.get('data_preprocessing', {})

# 训练时的超参数（需要与预处理脚本一致）
START_DATE = cfg.get('start_time', '2020-11-09 00:00:00')
FREQ = cfg.get('time_frequency', '4H')
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# 气象配置
METEOROLOGY_DIR = cfg.get('input_qx_path', '/home/fanyunkai/FYK_GCNLSTM/xiangjiang/QX')
meteo_config = cfg.get('qx_names', {
    'precip': 'xiangjiang_processed_precipitation_data.npy',
    'temp': 'xiangjiang_processed_temp_data.npy',
    'wind': 'xiangjiang_processed_wind_data.npy',
    'pres': 'xiangjiang_processed_pres_data.npy',
    'rh': 'xiangjiang_processed_rh_data.npy',
    'lwd': 'xiangjiang_processed_lwd_data.npy',
    'swd': 'xiangjiang_processed_swd_data.npy',
})

# 水质配置
WATER_QUALITY_DIR = cfg.get('input_wq_path', '/home/fanyunkai/FYK_GCNLSTM/a_original_dataset_processed/WQ_processed_dataset/')
site_names_ordered = cfg.get('sites_names')
wq_fields = cfg.get('wq_names', ['总氮', '总磷', '水温', 'pH', '溶解氧'])  # 注意：包含全部水质指标
fill_strategies = cfg.get('completion method', {
    '总氮': 'forward',
    '总磷': 'forward',
    '水温': 'forward',
    'pH': 'forward',
    '溶解氧': 'forward',
})
log_features = cfg.get('wq_log', ['precip'])
LOG_OFFSET = cfg.get('LOG_OFFSET', 0.01)

# 输出目录
OUTPUT_DIR = cfg.get('output_path', '/home/fanyunkai/FYK_GCNLSTM/xiangjiang')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 训练时保存的文件路径（用于加载 scaler 和特征列表）
SCALER_PATH = os.path.join(OUTPUT_DIR, 'scaler.pkl')
FEATURE_NAMES_PATH = os.path.join(OUTPUT_DIR, 'feature_names.pkl')

# ==================== 辅助函数 ====================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def generate_time_features(start_date, periods, freq='4H'):
    """生成时间特征（与预处理脚本完全一致）"""
    dates = pd.date_range(start=start_date, periods=periods, freq=freq)
    hour = dates.hour
    month = dates.month
    year = dates.year - dates.year[0]
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    year_offset = year.astype(np.float32)
    # 顺序：hour_sin, hour_cos, month_sin, month_cos, year_offset
    time_features = np.column_stack([hour_sin, hour_cos, month_sin, month_cos, year_offset]).astype(np.float32)
    time_feature_names = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'year_offset']
    return time_features, time_feature_names

def load_meteorology(meteo_config, base_dir, expected_sites):
    """加载气象数据，返回 (T, S, F_met) 和特征名称列表"""
    meteo_arrays = []
    meteo_names = []
    T, S = None, None
    for name, filename in meteo_config.items():
        path = os.path.join(base_dir, filename)
        arr = np.load(path).astype(np.float32)
        if T is None:
            T, S = arr.shape
        else:
            assert arr.shape == (T, S), f"{name} 形状不一致"
        meteo_arrays.append(arr)
        meteo_names.append(name)
    if S != len(expected_sites):
        raise ValueError(f"气象数据站点数 {S} 与配置的站点数 {len(expected_sites)} 不一致")
    meteo_data = np.stack(meteo_arrays, axis=-1)  # (T, S, F_met)
    return meteo_data, meteo_names

def load_water_quality(data_dir, site_names, wq_fields, T):
    """加载水质数据，返回 (T, S, F_wq) 和原始掩码 (T, S, F_wq)"""
    S = len(site_names)
    F_wq = len(wq_fields)
    wq_array = np.full((T, S, F_wq), np.nan, dtype=np.float32)
    mask_array = np.zeros((T, S, F_wq), dtype=np.float32)
    for i, site in enumerate(site_names):
        csv_path = os.path.join(data_dir, f"{site}.csv")
        df = pd.read_csv(csv_path, parse_dates=['监测时间'])
        for j, field in enumerate(wq_fields):
            values = df[field].values.astype(np.float32)
            if len(values) != T:
                if len(values) > T:
                    values = values[:T]
                else:
                    values = np.concatenate([values, [np.nan]*(T-len(values))])
            wq_array[:, i, j] = values
            mask_array[:, i, j] = (~np.isnan(values)).astype(np.float32)
    return wq_array, mask_array

def fill_missing(data, feature_names, fill_strategies, mask_suffix='_mask'):
    """缺失值填充（与训练脚本一致）"""
    filled = data.copy()
    for i, name in enumerate(feature_names):
        if name.endswith(mask_suffix):
            continue
        strategy = fill_strategies.get(name, 'forward')  # 默认 forward
        col = filled[..., i]
        if strategy == 'zero':
            col = np.nan_to_num(col, nan=0.0)
        elif strategy == 'mean':
            mean_val = np.nanmean(col)
            col = np.where(np.isnan(col), mean_val, col)
        elif strategy == 'forward':
            for s in range(col.shape[1]):
                series = col[:, s]
                mask_nan = np.isnan(series)
                if mask_nan.any():
                    # 使用 ffill 替代 fillna(method='ffill')
                    series = pd.Series(series).ffill().values
                    if np.isnan(series).any():
                        # 如果开头仍是 NaN，用该列均值填充
                        mean_val = np.nanmean(col[:, s])
                        if not np.isnan(mean_val):
                            series = np.where(np.isnan(series), mean_val, series)
                        else:
                            # 整列全为 NaN，用 0 填充
                            series = np.zeros_like(series)
                col[:, s] = series
        else:
            raise ValueError(f"未知填充策略: {strategy}")
        filled[..., i] = col
    return filled

def log_transform(data, feature_names, log_features, offset=0.01):
    """对数变换（与训练脚本一致）"""
    for name in log_features:
        if name not in feature_names:
            print(f"警告：特征 {name} 不存在，跳过对数变换")
            continue
        idx = feature_names.index(name)
        data[..., idx] = np.log10(data[..., idx] + offset)
    return data

def normalize_with_scaler(data, feature_names, scaler_params, exclude_suffix='_mask', exclude_names=None):
    """
    使用训练时保存的 scaler 参数进行标准化。
    scaler_params: 字典，包含 'means', 'stds', 'norm_indices', 'feature_names'
    """
    if exclude_names is None:
        exclude_names = []
    # 需要标准化的特征索引（与训练时一致）
    norm_indices = scaler_params['norm_indices']
    means = scaler_params['means']
    stds = scaler_params['stds']
    train_feature_names = scaler_params['feature_names']

    # 检查特征顺序是否一致
    if feature_names != train_feature_names:
        print("警告：当前特征列表与训练时不一致，请检查！")
        print("训练特征列表前10:", train_feature_names[:10])
        print("当前特征列表前10:", feature_names[:10])
        # 此处也可以根据实际需求调整，但最好保证一致

    normalized = data.copy()
    # 只对训练时标准化的特征进行标准化
    for i, idx in enumerate(norm_indices):
        # 注意：这里 i 对应 means 和 stds 的索引，idx 是特征在整体中的索引
        col = data[..., idx].reshape(-1, 1)   # 展平为 (T*S, 1)
        col_norm = (col - means[i]) / stds[i]
        normalized[..., idx] = col_norm.reshape(data.shape[:-1])
    return normalized

# ==================== 缺失模式注入函数 ====================
def inject_random_missing(wq, mask_orig, missing_ratio=0.25, seed=None):
    """模式1：随机缺失25%"""
    if seed is not None:
        np.random.seed(seed)
    new_wq = wq.copy()
    new_mask = mask_orig.copy()
    T, S, F = wq.shape
    for i in range(S):
        for j in range(F):
            valid_positions = np.where(mask_orig[:, i, j] == 1)[0]
            n_missing = int(len(valid_positions) * missing_ratio)
            if n_missing == 0:
                continue
            chosen = np.random.choice(valid_positions, size=n_missing, replace=False)
            new_wq[chosen, i, j] = np.nan
            new_mask[chosen, i, j] = 0
    return new_wq, new_mask

def inject_continuous_missing(wq, mask_orig, missing_ratio=0.25, min_len=6, max_len=180, seed=None):
    """模式2：独立连续缺失（每个指标每个站点独立）"""
    if seed is not None:
        np.random.seed(seed)
    new_wq = wq.copy()
    new_mask = mask_orig.copy()
    T, S, F = wq.shape
    for i in range(S):
        for j in range(F):
            valid_positions = np.where(mask_orig[:, i, j] == 1)[0]
            if len(valid_positions) == 0:
                continue
            total_missing_needed = int(len(valid_positions) * missing_ratio)
            if total_missing_needed == 0:
                continue
            missing_positions = set()
            while len(missing_positions) < total_missing_needed:
                seg_len = np.random.randint(min_len, max_len+1)
                possible_starts = [p for p in valid_positions if p+seg_len-1 <= T-1 and all((p <= q <= p+seg_len-1) for q in valid_positions)]
                if not possible_starts:
                    seg_len = 1
                    possible_starts = valid_positions
                start = np.random.choice(possible_starts)
                seg_indices = range(start, start+seg_len)
                for idx in seg_indices:
                    if idx in valid_positions and idx not in missing_positions:
                        missing_positions.add(idx)
            missing_list = sorted(missing_positions)
            new_wq[missing_list, i, j] = np.nan
            new_mask[missing_list, i, j] = 0
    return new_wq, new_mask

def inject_station_sync_missing(wq, mask_orig, missing_ratio=0.25, min_len=6, max_len=180, seed=None):
    """模式3：站内所有指标同时连续缺失（每个站点独立）"""
    if seed is not None:
        np.random.seed(seed)
    new_wq = wq.copy()
    new_mask = mask_orig.copy()
    T, S, F = wq.shape
    for i in range(S):
        common_mask = np.all(mask_orig[:, i, :] == 1, axis=1)
        valid_positions = np.where(common_mask)[0]
        if len(valid_positions) == 0:
            continue
        total_missing_needed = int(len(valid_positions) * missing_ratio)
        if total_missing_needed == 0:
            continue
        missing_positions = set()
        while len(missing_positions) < total_missing_needed:
            seg_len = np.random.randint(min_len, max_len+1)
            possible_starts = [p for p in valid_positions if p+seg_len-1 <= T-1 and all((p <= q <= p+seg_len-1) for q in valid_positions)]
            if not possible_starts:
                seg_len = 1
                possible_starts = valid_positions
            start = np.random.choice(possible_starts)
            seg_indices = range(start, start+seg_len)
            for idx in seg_indices:
                if idx in valid_positions and idx not in missing_positions:
                    missing_positions.add(idx)
        missing_list = sorted(missing_positions)
        for j in range(F):
            new_wq[missing_list, i, j] = np.nan
            new_mask[missing_list, i, j] = 0
    return new_wq, new_mask

def inject_global_sync_missing(wq, mask_orig, missing_ratio=0.25, min_len=6, max_len=180, seed=None):
    """模式4：全流域所有指标同时连续缺失"""
    if seed is not None:
        np.random.seed(seed)
    new_wq = wq.copy()
    new_mask = mask_orig.copy()
    T, S, F = wq.shape
    common_mask = np.all(mask_orig == 1, axis=(1,2))
    valid_positions = np.where(common_mask)[0]
    if len(valid_positions) == 0:
        return new_wq, new_mask
    total_missing_needed = int(len(valid_positions) * missing_ratio)
    if total_missing_needed == 0:
        return new_wq, new_mask
    missing_positions = set()
    while len(missing_positions) < total_missing_needed:
        seg_len = np.random.randint(min_len, max_len+1)
        possible_starts = [p for p in valid_positions if p+seg_len-1 <= T-1 and all((p <= q <= p+seg_len-1) for q in valid_positions)]
        if not possible_starts:
            seg_len = 1
            possible_starts = valid_positions
        start = np.random.choice(possible_starts)
        seg_indices = range(start, start+seg_len)
        for idx in seg_indices:
            if idx in valid_positions and idx not in missing_positions:
                missing_positions.add(idx)
    missing_list = sorted(missing_positions)
    for i in range(S):
        for j in range(F):
            new_wq[missing_list, i, j] = np.nan
            new_mask[missing_list, i, j] = 0
    return new_wq, new_mask

# ==================== 主程序 ====================
def main():
    set_seed(RANDOM_SEED)
    print(f"生成缺失模式 {MODE} 的测试集...")

    # 1. 加载原始数据（完整时间序列）
    print("加载气象数据...")
    meteo_data, meteo_names = load_meteorology(meteo_config, METEOROLOGY_DIR, site_names_ordered)
    T_total, S, F_met = meteo_data.shape
    print("加载水质数据...")
    wq_data, mask_orig = load_water_quality(WATER_QUALITY_DIR, site_names_ordered, wq_fields, T_total)
    F_wq = len(wq_fields)

    # 2. 提取测试集（时间最后 TEST_RATIO）
    test_len = int(T_total * TEST_RATIO)
    test_start = T_total - test_len
    print(f"总时间步数: {T_total}, 测试集步数: {test_len} (从 {test_start} 开始)")

    meteo_test = meteo_data[test_start:, :, :]          # (test_len, S, F_met)
    wq_test = wq_data[test_start:, :, :]                # (test_len, S, F_wq)
    mask_orig_test = mask_orig[test_start:, :, :]       # (test_len, S, F_wq)

    # 3. 生成时间特征
    print("生成时间特征...")
    time_features_full, time_feature_names = generate_time_features(START_DATE, T_total, freq=FREQ)
    time_features_test = time_features_full[test_start:, :]          # (test_len, 5)
    # 扩展为站点维度 (test_len, S, 5)
    time_features_test = np.repeat(time_features_test[:, np.newaxis, :], S, axis=1)

    # 4. 注入人为缺失
    print(f"注入人为缺失（模式 {MODE}）...")
    if MODE == 1:
        wq_test_missing, mask_final = inject_random_missing(wq_test, mask_orig_test, missing_ratio=0.25, seed=RANDOM_SEED)
    elif MODE == 2:
        wq_test_missing, mask_final = inject_continuous_missing(wq_test, mask_orig_test, missing_ratio=0.25, seed=RANDOM_SEED)
    elif MODE == 3:
        wq_test_missing, mask_final = inject_station_sync_missing(wq_test, mask_orig_test, missing_ratio=0.25, seed=RANDOM_SEED)
    elif MODE == 4:
        wq_test_missing, mask_final = inject_global_sync_missing(wq_test, mask_orig_test, missing_ratio=0.25, seed=RANDOM_SEED)
    else:
        raise ValueError("MODE 必须是 1,2,3,4")

    # 5. 按照训练时的特征顺序构建完整特征数组
    # 训练时特征顺序：时间特征(5) + 气象特征 + 水质特征 + 掩码特征
    all_features = [
        time_features_test,          # (test_len, S, 5) 顺序为 [hour_sin, hour_cos, month_sin, month_cos, year_offset]
        meteo_test,                  # (test_len, S, F_met)
        wq_test_missing,             # (test_len, S, F_wq)
        mask_final                   # (test_len, S, F_wq)
    ]
    all_data = np.concatenate(all_features, axis=-1)  # (test_len, S, F_total)

    # 特征名称列表（必须与训练时完全一致）
    feature_names = (
        time_feature_names +   # ['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'year_offset']
        meteo_names +
        wq_fields +
        [f"{f}_mask" for f in wq_fields]
    )
    print(f"构建的特征列表长度: {len(feature_names)}")
    print("前10个特征:", feature_names[:10])

    # 6. 预处理：填充、对数变换
    print("执行填充...")
    all_data_filled = fill_missing(all_data, feature_names, fill_strategies, mask_suffix='_mask')

    print("执行对数变换...")
    all_data_log = log_transform(all_data_filled, feature_names, log_features, LOG_OFFSET)

    # 7. 加载训练时的 scaler 参数
    print("加载 scaler...")
    with open(SCALER_PATH, 'rb') as f:
        scaler_params = pickle.load(f)

    # 8. 标准化（使用训练时的统计量）
    print("执行标准化...")
    all_data_norm = normalize_with_scaler(
        all_data_log, feature_names, scaler_params,
        exclude_suffix='_mask', exclude_names=['hour_sin', 'hour_cos', 'month_sin', 'month_cos']
    )

    # 9. 保存结果
    output_prefix = os.path.join(OUTPUT_DIR, f"test_data_mode{MODE}")
    np.save(output_prefix + ".npy", all_data_norm)
    np.save(os.path.join(OUTPUT_DIR, f"test_original_mask_mode{MODE}.npy"), mask_orig_test)
    np.save(os.path.join(OUTPUT_DIR, f"test_final_mask_mode{MODE}.npy"), mask_final)
    print(f"完成！结果保存在 {output_prefix}.npy 及相应掩码文件。")

if __name__ == "__main__":
    main()