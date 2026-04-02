"""
GCN-LSTM / GAT-LSTM 数据预处理流水线
输入：
    - 气象数据：一个 3D .npy 文件，形状 (T, S, F_met)
    - 水质数据：每个站点一个 CSV，文件名格式为 站点名.csv
    - 子流域静态特征：一个 CSV，按站点名匹配
输出：
    - preprocessed_data.npy : 形状 (T, S, F) 的原始尺度数组
    - feature_names.pkl     : 特征名称列表
    - metadata.npz          : 时间、站点、特征名

说明：
    1. 特征大类开关只用于“本预处理脚本里决定拼哪些特征”，不服务于训练代码。
    2. 最终拼接顺序固定为：气象 -> 时间 -> 静态 -> 水质 -> 掩码
    3. 水质按气象 metadata.npz 里的时间轴自动对齐，不再按长度硬截断。
    4. 时间特征也直接基于气象 metadata.npz 的时间轴生成。
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

cfg = config_all.get('QXmultidata_preprocessing', {})

# ---------- 路径 ----------
MET_NPY_PATH = cfg.get('input_qx_npy')
MET_FEATURE_NAMES_PATH = cfg.get('input_qx_feature_names_path')
MET_META_PATH = cfg.get('input_qx_metadata')
WATER_QUALITY_DIR = cfg.get('input_wq_path')
OUTPUT_DIR = cfg.get('output_path')
STATIC_FEATURE_PATH = cfg.get('static_feature_path')

# ---------- 站点 ----------
site_names_ordered = cfg.get('sites_names', [])

# ---------- 气象 / 水质 ----------
wq_fields = cfg.get('wq_names', [])
fill_strategies = cfg.get('completion method', {})
log_features = cfg.get('wq_log', [])
LOG_OFFSET = cfg.get('LOG_OFFSET', 0.01)
MASK_SUFFIX = cfg.get('MASK_SUFFIX', '_mask')

# ---------- 时间 ----------
START_DATE = cfg.get('start_time')
FREQ = cfg.get('time_frequency', '4h')

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
def generate_time_features(time_index):
    """
    直接基于给定时间轴生成时间特征
    返回：
        time_features: (T, F_time)
        time_feature_names: list[str]
    """
    dates = pd.DatetimeIndex(time_index)

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


def load_meteorology(npy_path, feature_names_path, meta_path, expected_sites):
    """
    加载一个 3D 气象文件
    返回：
        meteo_data: (T, S, F_met)
        meteo_names: list[str]
        meta_times: DatetimeIndex
    """
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"气象文件不存在: {npy_path}")
    if not os.path.exists(feature_names_path):
        raise FileNotFoundError(f"气象特征名文件不存在: {feature_names_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"气象 metadata 文件不存在: {meta_path}")

    meteo_data = np.load(npy_path).astype(np.float32)
    if meteo_data.ndim != 3:
        raise ValueError(f"气象文件应为 (T, S, F)，实际为 {meteo_data.shape}")

    with open(feature_names_path, 'rb') as f:
        meteo_names = pickle.load(f)

    meta = np.load(meta_path, allow_pickle=True)
    if 'stations' not in meta:
        raise KeyError("气象 metadata 中缺少 stations")
    if 'times' not in meta:
        raise KeyError("气象 metadata 中缺少 times")

    meta_sites = [str(x).strip() for x in meta['stations'].tolist()]
    meta_times = pd.to_datetime(meta['times'])

    expected_sites = [str(x).strip() for x in expected_sites]

    if meta_sites != expected_sites:
        raise ValueError(
            "气象 metadata 中的站点顺序与配置不一致。\n"
            f"config: {expected_sites}\n"
            f"meta:   {meta_sites}"
        )

    T, S, F = meteo_data.shape
    if S != len(expected_sites):
        raise ValueError(f"气象站点数 {S} 与配置站点数 {len(expected_sites)} 不一致")

    if len(meteo_names) != F:
        raise ValueError(f"feature_names 长度 {len(meteo_names)} 与气象特征数 {F} 不一致")

    expected_times = pd.date_range(start=START_DATE, periods=meteo_data.shape[0], freq=FREQ)
    if len(meta_times) != len(expected_times) or not np.all(meta_times == expected_times):
        raise ValueError("气象 metadata 中的时间轴与当前 config 的 start_time / time_frequency 不一致")

    return meteo_data, meteo_names, meta_times


def load_water_quality(data_dir, site_names, wq_fields, target_time_index):
    """
    读取水质数据并按 target_time_index 对齐
    返回：
        wq_array   : (T, S, F_wq)
        mask_array : (T, S, F_wq)
    """
    if not wq_fields:
        raise ValueError("wq_names 为空，无法加载水质特征。")

    target_time_index = pd.DatetimeIndex(target_time_index)
    T = len(target_time_index)
    S = len(site_names)
    F_wq = len(wq_fields)

    wq_array = np.full((T, S, F_wq), np.nan, dtype=np.float32)
    mask_array = np.zeros((T, S, F_wq), dtype=np.float32)

    for i, site in enumerate(site_names):
        csv_path = os.path.join(data_dir, f"{site}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"站点文件缺失：{csv_path}")

        df = pd.read_csv(csv_path, parse_dates=['监测时间'])
        if '监测时间' not in df.columns:
            raise ValueError(f"站点 {site} 的 CSV 中缺少列: 监测时间")

        df = df.sort_values('监测时间').copy()
        df['监测时间'] = pd.to_datetime(df['监测时间'])

        for j, field in enumerate(wq_fields):
            if field not in df.columns:
                raise ValueError(f"站点 {site} 的 CSV 中缺少字段: {field}")

            series = pd.Series(
                pd.to_numeric(df[field], errors='coerce').astype(np.float32).values,
                index=df['监测时间']
            )

            # 如果同一时间有重复记录，保留最后一条
            series = series.groupby(level=0).last()

            aligned = series.reindex(target_time_index)

            wq_array[:, i, j] = aligned.values.astype(np.float32)
            mask_array[:, i, j] = (~np.isnan(aligned.values)).astype(np.float32)

            orig_valid = int(np.sum(~np.isnan(series.values)))
            aligned_valid = int(np.sum(~np.isnan(aligned.values)))
            if len(series) != T:
                print(
                    f"站点 {site} 的 {field}: 原长度 {len(series)}，"
                    f"按气象时间轴对齐后长度 {T}，有效值 {orig_valid}->{aligned_valid}"
                )

    return wq_array, mask_array


def load_static_features(static_csv_path, site_names, site_col, ignore_cols=None, selected_features=None):
    """
    读取子流域静态特征，并按 site_names 顺序对齐。
    """
    if static_csv_path is None:
        raise ValueError("已启用 static 特征，但 static_feature_path 未设置。")
    if not os.path.exists(static_csv_path):
        raise FileNotFoundError(f"静态特征文件不存在: {static_csv_path}")

    ignore_cols = ignore_cols or []
    df = pd.read_csv(static_csv_path)

    if site_col not in df.columns:
        raise ValueError(f"静态特征表中找不到站点匹配列: {site_col}")

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

    for col in static_feature_names:
        df_selected[col] = pd.to_numeric(df_selected[col], errors='coerce')

    if df_selected.isna().any().any():
        bad_cols = df_selected.columns[df_selected.isna().any()].tolist()
        bad_rows = df_selected[df_selected.isna().any(axis=1)].index.tolist()
        raise ValueError(
            f"静态特征存在无法转成数值或缺失的值。问题列: {bad_cols}，问题站点: {bad_rows}"
        )

    static_data = df_selected.to_numpy(dtype=np.float32)
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

    T = None
    meteo_time_index = None
    feature_blocks = []
    feature_names = []

    # ---------- 气象 ----------
    if INCLUDE_METEO:
        print("加载气象数据...")
        meteo_data, meteo_names, meteo_time_index = load_meteorology(
            MET_NPY_PATH,
            MET_FEATURE_NAMES_PATH,
            MET_META_PATH,
            site_names
        )
        T, S_met, _ = meteo_data.shape
        if S_met != S:
            raise ValueError(f"气象数据站点数 {S_met} 与配置站点数 {S} 不一致")

        feature_blocks.append(meteo_data)
        feature_names.extend(meteo_names)
        print(f"气象数据形状: {meteo_data.shape}, 特征: {meteo_names}")

    # ---------- 时间 ----------
    if INCLUDE_TIME:
        if meteo_time_index is not None:
            time_index_for_features = meteo_time_index
            T = len(time_index_for_features)
        else:
            if T is None:
                if not wq_fields:
                    raise ValueError("未启用气象且 wq_names 为空，无法确定时间长度 T。")
                sample_site = site_names[0]
                sample_csv = os.path.join(WATER_QUALITY_DIR, f"{sample_site}.csv")
                if not os.path.exists(sample_csv):
                    raise FileNotFoundError(f"无法确定 T，缺少示例水质文件: {sample_csv}")
                sample_df = pd.read_csv(sample_csv)
                T = len(sample_df)

            time_index_for_features = pd.date_range(start=START_DATE, periods=T, freq=FREQ)

        print("生成时间特征...")
        time_features, time_feature_names = generate_time_features(time_index_for_features)
        time_features_expanded = np.repeat(time_features[:, np.newaxis, :], S, axis=1)

        feature_blocks.append(time_features_expanded)
        feature_names.extend(time_feature_names)
        print(f"时间特征形状: {time_features_expanded.shape}, 特征: {time_feature_names}")

    # ---------- 静态 ----------
    if INCLUDE_STATIC:
        if T is None:
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
        static_data = np.repeat(static_site_data[np.newaxis, :, :], T, axis=0)

        feature_blocks.append(static_data)
        feature_names.extend(static_feature_names)
        print(f"静态特征形状: {static_data.shape}, 特征: {static_feature_names}")

    # ---------- 水质 / 掩码 ----------
    need_wq_loading = INCLUDE_WQ or INCLUDE_MASK
    if need_wq_loading:
        if meteo_time_index is None:
            raise ValueError("当前脚本要求水质按气象 metadata 时间轴对齐，因此必须启用 meteorology 并提供 input_qx_metadata。")

        print("加载水质数据并按气象时间轴对齐生成掩码...")
        wq_array, mask_array = load_water_quality(
            WATER_QUALITY_DIR,
            site_names,
            wq_fields,
            meteo_time_index
        )

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

    if meteo_time_index is not None:
        times = meteo_time_index
    else:
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
