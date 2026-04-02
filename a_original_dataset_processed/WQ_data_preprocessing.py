#处理水质数据集：1.删除早于开始时间和晚于结束时间的数据；2.补全缺失的时间点（每天00,04,08,12,16,20）；3.删除指定列；4.处理异常值和滑动窗口处理。
import os
import pandas as pd
import numpy as np
import yaml

# ========== 用户配置区 ==========
CONFIG_PATH = "/home/fanyunkai/FYK_GCNLSTM/configs.yaml"  # 可根据实际情况调整路径

with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config_all = yaml.safe_load(f)

# 提取水质预处理部分的配置
preprocess_cfg = config_all['WQ_data_preprocessing']

GLOBAL_START_DATE = preprocess_cfg['start_time']
GLOBAL_END_DATE   = preprocess_cfg['end_time']
INPUT_FOLDER      = preprocess_cfg['input_path']
OUTPUT_FOLDER     = preprocess_cfg['output_path']
# ===============================

# 第一步：删除早于全局开始时间的数据
def filter_by_date(df, start_date=GLOBAL_START_DATE):
    df['监测时间'] = pd.to_datetime(df['监测时间'], format='%Y-%m-%d %H:%M:%S')
    df = df[df['监测时间'] >= pd.to_datetime(start_date)]
    return df

# 补全缺失的时间点（每天00,04,08,12,16,20），基于全局时间范围
def add_missing_timepoints(df, start_date=GLOBAL_START_DATE, end_date=GLOBAL_END_DATE):
    """
    根据全局开始和结束时间生成完整的时间序列（每4小时），
    与原始数据进行左连接，缺失的时刻对应指标列为NaN。
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    # 生成完整的时间序列，间隔4小时
    full_time = pd.date_range(start=start, end=end, freq='4H')
    full_df = pd.DataFrame({'监测时间': full_time})
    # 左连接，保留所有时刻
    merged = pd.merge(full_df, df, on='监测时间', how='left')
    return merged

# 第二步：删除指定列
def drop_columns(df):
    columns_to_drop = ['水质类别', 'pH类别', '溶解氧类别', '高锰酸盐类别', '氨氮类别', '叶绿素', '藻密度', '站点情况']
    # 只删除存在的列，避免报错
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(columns=columns_to_drop)
    return df

# 第三步：处理异常值和滑动窗口处理
def clean_data(df, column_names, window_size=100, threshold_high=4, threshold_low=0.25, min_valid_neighbors=50):
    stats = {
        "orig_nan": 0,
        "non_positive": 0,
        "high_anomaly": 0,
        "low_anomaly": 0,
        "normal": 0,
    }
    
    for col in column_names:
        # 如果列不存在，跳过
        if col not in df.columns:
            continue
        # 强制转换为数值类型，无法转换的填为 NaN
        s_num = pd.to_numeric(df[col], errors='coerce')
        
        # 1) 记录原始的 NaN 值
        orig_nan_mask = s_num.isna()
        stats["orig_nan"] += orig_nan_mask.sum()

        # 2) 记录非正值 (<=0)
        non_pos_mask = (~orig_nan_mask) & (s_num <= 0)
        stats["non_positive"] += non_pos_mask.sum()

        cleaned = s_num.copy()
        cleaned[non_pos_mask] = np.nan  # 将非正值标记为 NaN

        # 3/4) 滑动窗口清理异常高/低值
        values = cleaned.to_numpy(dtype=float)
        n = len(values)
        new_values = values.copy()

        high_cnt = 0
        low_cnt = 0

        for idx in range(n):
            cur = values[idx]
            if np.isnan(cur):
                continue

            # 设置窗口的范围
            start_idx = max(0, idx - window_size)
            end_idx = min(n, idx + window_size + 1)

            window_vals = values[start_idx:end_idx]
            valid = window_vals[~np.isnan(window_vals)]

            if valid.size < min_valid_neighbors:
                continue

            mean_val = valid.mean()

            if cur >= mean_val * threshold_high:
                new_values[idx] = np.nan
                high_cnt += 1
            elif cur <= mean_val * threshold_low:
                new_values[idx] = np.nan
                low_cnt += 1

        df[col] = new_values
        stats["high_anomaly"] += high_cnt
        stats["low_anomaly"] += low_cnt
        stats["normal"] += (pd.Series(new_values).notna().sum())

    return df, stats

# 处理所有CSV文件
def process_files(input_folder, output_folder):
    # 获取文件夹中所有CSV文件
    files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 汇总处理信息
    summary_txt = []
    skipped_files = []   # 记录跳过的文件
    
    for file in files:
        file_path = os.path.join(input_folder, file)
        print(f"开始处理文件: {file}")
        df = pd.read_csv(file_path)
        
        # 第一步：过滤监测时间（只保留 >= 全局开始时间）
        df = filter_by_date(df)
        
        # 检查过滤后是否为空
        if df.empty:
            msg = f"文件 {file} 在删除早于{GLOBAL_START_DATE}的数据后为空，跳过处理。"
            print(msg)
            skipped_files.append(file)
            summary_txt.append(f"{file} 跳过：过滤后无数据。\n")
            continue
        
        # 再过滤掉晚于全局结束时间的数据（确保不超出范围）
        df = df[df['监测时间'] <= pd.to_datetime(GLOBAL_END_DATE)]
        if df.empty:
            msg = f"文件 {file} 在过滤超出结束时间后为空，跳过处理。"
            print(msg)
            skipped_files.append(file)
            summary_txt.append(f"{file} 跳过：超出结束时间后无数据。\n")
            continue
        
        # 补全缺失的时间点（基于全局时间范围）
        df = add_missing_timepoints(df)
        
        # 第二步：删除指定列
        df = drop_columns(df)
        
        # 第三步：清理数据，处理异常值
        column_names = ['水温', 'pH', '溶解氧', '高锰酸盐指数', '氨氮', '总磷', '总氮', '电导率', '浊度']
        cleaned_df, stats = clean_data(df, column_names)
        
        # 保存处理后的数据
        output_file_path = os.path.join(output_folder, file)
        cleaned_df.to_csv(output_file_path, index=False, encoding="utf-8-sig")
        
        # 记录处理情况
        summary_txt.append(f"{file} 处理统计：\n")
        for key, value in stats.items():
            summary_txt.append(f"{key}: {value} 个\n")
        summary_txt.append("\n")

        # 输出成功处理文件的信息
        print(f"已成功处理文件: {file}")
    
    # 汇总跳过的文件
    if skipped_files:
        summary_txt.append("\n跳过的文件列表（过滤后无数据）：\n")
        for f in skipped_files:
            summary_txt.append(f"{f}\n")
    
    # 将处理统计信息保存到txt文件
    with open(os.path.join(output_folder, "summary.txt"), "w", encoding="utf-8-sig") as f:
        f.writelines(summary_txt)

# 执行数据处理（使用配置区的路径）
process_files(INPUT_FOLDER, OUTPUT_FOLDER)