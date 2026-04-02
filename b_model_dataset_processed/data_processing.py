"""
GCN-LSTM / GAT-LSTM 数据预处理流水线
输入：
    - 气象数据：多个 .npy 文件，每个形状 (T, S)
    - 水质数据：每个站点一个 CSV，文件名格式为 站点名.csv
    - 子流域静态特征：一个 CSV，按站点名匹配
输出：
    - preprocessed_data.npy : 形状 (T, S, F) 的原始尺度数组
    - feature_names.pkl     : 特征名称列表

说明：
    1. 特征大类开关只用于“本预处理脚本里决定拼哪些特征”，不服务于训练代码。
    2. 最终拼接顺序固定为：气象 -> 时间 -> 静态 -> 水质 -> 掩码
"""

import os
import pickle
import numpy as np
import pandas as pd
import yaml

# ==================== 从 configs.yaml 读取配置 ====================
CONFIG_PATH = "/home/fanyunkai/FYK_GCNLSTM/configs.yaml"

with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config_all = yaml.safe_load(f)

cfg = config_all.get('data_preprocessing', {})

# ---------- 路径 ----------
METEOROLOGY_DIR = cfg.get('input_qx_path')
WATER_QUALITY_DIR = cfg.get('input_wq_path')
OUTPUT_DIR = cfg.get('output_path')
STATIC_FEATURE_PATH = cfg.get('static_feature_path')

# ---------- 站点 ----------
site_names_ordered = cfg.get('sites_names', [])

# ---------- 气象 / 水质 ----------
meteo_config = cfg.get('qx_names', {})
wq_fields = cfg.get('wq_names', [])
fill_strategies = cfg.get('completion method', {})
log_features = cfg.get('wq_log', [])
LOG_OFFSET = cfg.get('LOG_OFFSET', 0.01)
MASK_SUFFIX = cfg.get('MASK_SUFFIX', '_mask')

# ---------- 时间 ----------
START_DATE = cfg.get('start_time')
FREQ = cfg.get('time_frequency', '4H')

# ---------- 大类选择（仅在本预处理脚本中使用） ----------
include_groups = cfg.get('include_feature_groups', {})
INCLUDE_METEO = include_groups.get('meteorology', True)
INCLUDE_TIME = include_groups.get('time', True)
INCLUDE_STATIC = include_groups.get('static', False)
INCLUDE_WQ = include_groups.get('water_quality', True)
INCLUDE_MASK = include_groups.get('mask', True)

# 向后兼容：若旧配置还保留 include_mask，则仅在新开关未显式提供 mask 时兜底
if 'mask' not in include_groups:
    INCLUDE_MASK = cfg.get('include_mask', True)

# ---------- 静态特征 ----------
STATIC_SITE_COL = cfg.get('static_site_col', '流域名称')
STATIC_IGNORE_COLS = cfg.get('static_ignore_cols', ['流域名称', '子流域顺序'])
STATIC_FEATURE_NAMES = cfg.get('static_feature_names', None)  # None 表示自动选取可用列


# ==================================================
def generate_time_features(start_date, periods, freq='4H'):
    """
    生成时间特征（小时、月、年）
    返回：
        time_features: (T, F_time)
        time_feature_names: list[str]
    """
    dates = pd.date_range(start=start_date, periods=periods, freq=freq)

    hour = dates.hour
    month = dates.month
    year = dates.year - dates.year[0]

    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    year_offset = year.astype(np.float32)

    time_features = np.column_stack([
        hour_sin, hour_cos,
        month_sin, month_cos,
        year_offset
    ]).astype(np.float32)

    time_feature_names = [
        'hour_sin', 'hour_cos',
        'month_sin', 'month_cos',
        'year_offset'
    ]
    return time_features, time_feature_names


def load_meteorology(meteo_config, base_dir, expected_sites):
    """
    加载所有气象特征
    返回：
        meteo_data: (T, S, F_met)
        meteo_names: list[str]
    """
    if not meteo_config:
        raise ValueError("qx_names 为空，无法加载气象特征。")

    meteo_arrays = []
    meteo_names = []
    T, S = None, None

    for name, filename in meteo_config.items():
        path = os.path.join(base_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"气象文件不存在: {path}")

        arr = np.load(path).astype(np.float32)
        if arr.ndim != 2:
            raise ValueError(f"气象文件 {path} 形状应为 (T, S)，实际为 {arr.shape}")

        if T is None:
            T, S = arr.shape
        else:
            if arr.shape != (T, S):
                raise ValueError(f"气象特征 {name} 形状不一致，期望 {(T, S)}，实际 {arr.shape}")

        meteo_arrays.append(arr)
        meteo_names.append(name)

    if S != len(expected_sites):
        raise ValueError(f"气象数据站点数 {S} 与配置站点数 {len(expected_sites)} 不一致")

    meteo_data = np.stack(meteo_arrays, axis=-1)
    return meteo_data, meteo_names


def load_water_quality(data_dir, site_names, wq_fields, T):
    """
    读取水质数据与对应掩码
    返回：
        wq_array   : (T, S, F_wq)
        mask_array : (T, S, F_wq)
    """
    if not wq_fields:
        raise ValueError("wq_names 为空，无法加载水质特征。")

    S = len(site_names)
    F_wq = len(wq_fields)
    wq_array = np.full((T, S, F_wq), np.nan, dtype=np.float32)
    mask_array = np.zeros((T, S, F_wq), dtype=np.float32)

    for i, site in enumerate(site_names):
        csv_path = os.path.join(data_dir, f"{site}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"站点文件缺失：{csv_path}")

        df = pd.read_csv(csv_path, parse_dates=['监测时间'])

        for j, field in enumerate(wq_fields):
            if field not in df.columns:
                raise ValueError(f"站点 {site} 的 CSV 中缺少字段: {field}")

            values = df[field].values.astype(np.float32)
            if len(values) != T:
                print(f"警告: {site} 的 {field} 长度 {len(values)} 不等于 T={T}，将截断/补 NaN")
                if len(values) > T:
                    values = values[:T]
                else:
                    values = np.concatenate([values, np.full(T - len(values), np.nan, dtype=np.float32)])

            wq_array[:, i, j] = values
            mask_array[:, i, j] = (~np.isnan(values)).astype(np.float32)

    return wq_array, mask_array


def load_static_features(static_csv_path, site_names, site_col, ignore_cols=None, selected_features=None):
    """
    读取子流域静态特征，并按 site_names 顺序对齐。

    参数：
        static_csv_path: 静态特征 CSV 路径
        site_names: 站点顺序
        site_col: 站点匹配列名，如“流域名称”
        ignore_cols: 不参与特征拼接的列
        selected_features: 指定参与拼接的静态特征列；None 表示自动选择

    返回：
        static_data: (S, F_static)
        static_feature_names: list[str]
    """
    if static_csv_path is None:
        raise ValueError("已启用 static 特征，但 static_feature_path 未设置。")
    if not os.path.exists(static_csv_path):
        raise FileNotFoundError(f"静态特征文件不存在: {static_csv_path}")

    ignore_cols = ignore_cols or []
    df = pd.read_csv(static_csv_path)

    if site_col not in df.columns:
        raise ValueError(f"静态特征表中找不到站点匹配列: {site_col}")

    # 去空格，避免名称匹配失败
    df[site_col] = df[site_col].astype(str).str.strip()
    expected_sites = [str(x).strip() for x in site_names]

    if df[site_col].duplicated().any():
        dup_sites = df.loc[df[site_col].duplicated(), site_col].tolist()
        raise ValueError(f"静态特征表中存在重复站点名: {dup_sites}")

    all_candidate_cols = [c for c in df.columns if c not in ignore_cols]
    if site_col in all_candidate_cols:
        all_candidate_cols.remove(site_col)

    if selected_features is None:
        static_feature_names = all_candidate_cols
    else:
        missing_cols = [c for c in selected_features if c not in df.columns]
        if missing_cols:
            raise ValueError(f"static_feature_names 中这些列在 CSV 中不存在: {missing_cols}")
        static_feature_names = selected_features

    if not static_feature_names:
        raise ValueError("静态特征列为空，请检查 static_feature_names 或 static_ignore_cols 配置。")

    missing_sites = [s for s in expected_sites if s not in set(df[site_col])]
    if missing_sites:
        raise ValueError(f"静态特征表中缺少这些站点: {missing_sites}")

    df = df.set_index(site_col)
    df_selected = df.loc[expected_sites, static_feature_names].copy()

    # 全部转为数值
    for col in static_feature_names:
        df_selected[col] = pd.to_numeric(df_selected[col], errors='coerce')

    if df_selected.isna().any().any():
        bad_cols = df_selected.columns[df_selected.isna().any()].tolist()
        bad_rows = df_selected[df_selected.isna().any(axis=1)].index.tolist()
        raise ValueError(
            f"静态特征存在无法转成数值或缺失的值。问题列: {bad_cols}，问题站点: {bad_rows}"
        )

    static_data = df_selected.to_numpy(dtype=np.float32)  # (S, F_static)
    return static_data, static_feature_names


def fill_missing(data, feature_names, fill_strategies, mask_suffix='_mask'):
    """
    对数值特征（不包括掩码）进行缺失值填充。
    只允许 'zero' 或 'forward'。
    """
    filled = data.copy()

    for i, name in enumerate(feature_names):
        if name.endswith(mask_suffix):
            continue

        strategy = fill_strategies.get(name, 'forward')
        col = filled[..., i]

        if strategy == 'zero':
            col = np.nan_to_num(col, nan=0.0)
        elif strategy == 'forward':
            # 对每个站点单独前向填充
            for s in range(col.shape[1]):
                series = col[:, s]
                if np.isnan(series).any():
                    series = pd.Series(series).ffill().to_numpy()
                    if np.isnan(series).any():
                        series = np.where(np.isnan(series), 0.0, series)
                col[:, s] = series
        else:
            raise ValueError(f"未知填充策略: {strategy}，只允许 'zero' 或 'forward'")

        filled[..., i] = col

    return filled


def log_transform(data, feature_names, log_features, offset=0.01):
    """对指定特征做 log10(x + offset) 变换"""
    transformed = data.copy()
    for name in log_features:
        if name not in feature_names:
            print(f"警告：特征 {name} 不在当前拼接结果中，跳过对数变换")
            continue
        idx = feature_names.index(name)
        transformed[..., idx] = np.log10(transformed[..., idx] + offset)
    return transformed


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    site_names = site_names_ordered
    if not site_names:
        raise ValueError("sites_names 为空，请在 config 中提供站点顺序。")

    S = len(site_names)
    print(f"将处理 {S} 个站点，顺序为: {site_names}")
    print("特征大类开关（仅用于本预处理脚本）:")
    print(f"  meteorology: {INCLUDE_METEO}")
    print(f"  time: {INCLUDE_TIME}")
    print(f"  static: {INCLUDE_STATIC}")
    print(f"  water_quality: {INCLUDE_WQ}")
    print(f"  mask: {INCLUDE_MASK}")

    # 1. 为了确定 T，优先加载气象；若未启用气象，则通过水质长度确定 T
    T = None
    feature_blocks = []
    feature_names = []

    # ---------- 气象 ----------
    if INCLUDE_METEO:
        print("加载气象数据...")
        meteo_data, meteo_names = load_meteorology(meteo_config, METEOROLOGY_DIR, site_names)
        T, S_met, _ = meteo_data.shape
        if S_met != S:
            raise ValueError(f"气象数据站点数 {S_met} 与配置站点数 {S} 不一致")

        feature_blocks.append(meteo_data)
        feature_names.extend(meteo_names)
        print(f"气象数据形状: {meteo_data.shape}, 特征: {meteo_names}")

    # ---------- 时间 ----------
    if INCLUDE_TIME:
        if T is None:
            # 若没有气象，则尝试通过第一个水质文件长度确定 T
            if not wq_fields:
                raise ValueError("未启用气象且 wq_names 为空，无法确定时间长度 T。")
            sample_site = site_names[0]
            sample_csv = os.path.join(WATER_QUALITY_DIR, f"{sample_site}.csv")
            if not os.path.exists(sample_csv):
                raise FileNotFoundError(f"无法确定 T，缺少示例水质文件: {sample_csv}")
            sample_df = pd.read_csv(sample_csv)
            T = len(sample_df)

        print("生成时间特征...")
        time_features, time_feature_names = generate_time_features(START_DATE, T, freq=FREQ)
        time_features_expanded = np.repeat(time_features[:, np.newaxis, :], S, axis=1)

        feature_blocks.append(time_features_expanded)
        feature_names.extend(time_feature_names)
        print(f"时间特征形状: {time_features_expanded.shape}, 特征: {time_feature_names}")

    # ---------- 静态 ----------
    if INCLUDE_STATIC:
        if T is None:
            # 若既没气象也没时间，这里也需要先确定 T
            if not wq_fields:
                raise ValueError("未启用气象/时间且 wq_names 为空，无法确定时间长度 T。")
            sample_site = site_names[0]
            sample_csv = os.path.join(WATER_QUALITY_DIR, f"{sample_site}.csv")
            if not os.path.exists(sample_csv):
                raise FileNotFoundError(f"无法确定 T，缺少示例水质文件: {sample_csv}")
            sample_df = pd.read_csv(sample_csv)
            T = len(sample_df)

        print("加载静态子流域特征...")
        static_site_data, static_feature_names = load_static_features(
            static_csv_path=STATIC_FEATURE_PATH,
            site_names=site_names,
            site_col=STATIC_SITE_COL,
            ignore_cols=STATIC_IGNORE_COLS,
            selected_features=STATIC_FEATURE_NAMES,
        )
        # (S, F_static) -> (T, S, F_static)
        static_data = np.repeat(static_site_data[np.newaxis, :, :], T, axis=0)

        feature_blocks.append(static_data)
        feature_names.extend(static_feature_names)
        print(f"静态特征形状: {static_data.shape}, 特征: {static_feature_names}")

    # ---------- 水质 / 掩码 ----------
    need_wq_loading = INCLUDE_WQ or INCLUDE_MASK
    if need_wq_loading:
        if T is None:
            sample_site = site_names[0]
            sample_csv = os.path.join(WATER_QUALITY_DIR, f"{sample_site}.csv")
            if not os.path.exists(sample_csv):
                raise FileNotFoundError(f"无法确定 T，缺少示例水质文件: {sample_csv}")
            sample_df = pd.read_csv(sample_csv)
            T = len(sample_df)

        print("加载水质数据并生成掩码...")
        wq_array, mask_array = load_water_quality(WATER_QUALITY_DIR, site_names, wq_fields, T)

        if INCLUDE_WQ:
            feature_blocks.append(wq_array)
            feature_names.extend(wq_fields)
            print(f"水质特征形状: {wq_array.shape}, 特征: {wq_fields}")

        if INCLUDE_MASK:
            mask_feature_names = [f"{f}{MASK_SUFFIX}" for f in wq_fields]
            feature_blocks.append(mask_array)
            feature_names.extend(mask_feature_names)
            print(f"掩码特征形状: {mask_array.shape}, 特征: {mask_feature_names}")

    if not feature_blocks:
        raise ValueError("没有任何特征被选中，无法构造数据集。")

    # 固定顺序：气象 -> 时间 -> 静态 -> 水质 -> 掩码
    all_data = np.concatenate(feature_blocks, axis=-1)
    print(f"拼接后数据形状: {all_data.shape}")
    print(f"特征顺序: {feature_names}")

    print("进行缺失值填充...")
    all_data_filled = fill_missing(
        data=all_data,
        feature_names=feature_names,
        fill_strategies=fill_strategies,
        mask_suffix=MASK_SUFFIX,
    )

    print("进行对数变换...")
    all_data_transformed = log_transform(
        data=all_data_filled,
        feature_names=feature_names,
        log_features=log_features,
        offset=LOG_OFFSET,
    )

    print("保存结果...")
    np.save(os.path.join(OUTPUT_DIR, 'preprocessed_data.npy'), all_data_transformed)
    with open(os.path.join(OUTPUT_DIR, 'feature_names.pkl'), 'wb') as f:
        pickle.dump(feature_names, f)
# 保存 metadata.npz
    times = pd.date_range(start=START_DATE, periods=T, freq=FREQ)

    np.savez(
    os.path.join(OUTPUT_DIR, 'metadata.npz'),
    times=np.array(times, dtype='datetime64[s]'),
    stations=np.array(site_names, dtype=object),
    feature_names=np.array(feature_names, dtype=object)
    )
    
    print(f"预处理完成，结果保存在 {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
