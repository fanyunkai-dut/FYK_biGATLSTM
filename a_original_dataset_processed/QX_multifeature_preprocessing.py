#跟QX_precidata_preprocessing和QX_otherdata_preprocessing二选一，这一版本直接构造所有气象且多特征。
import os
import re
import pickle
import numpy as np
import pandas as pd
import netCDF4 as nc
from scipy.spatial import cKDTree
from datetime import datetime, timedelta
import yaml

# ==================== 基本配置 ====================
CONFIG_PATH = "/home/fanyunkai/FYK_GCNLSTM/configs.yaml"

# 水质时间轴：4小时一步
FREQ_HOURS = 4

# 72h / 4h = 18步
API_STEPS = 18
API_DECAY = 0.9
HIST_HOURS_FOR_API = (API_STEPS - 1) * FREQ_HOURS   # 68h

DAY24_STEPS = 24 // FREQ_HOURS   # 6
DAY48_STEPS = 48 // FREQ_HOURS   # 12
DAY72_STEPS = 72 // FREQ_HOURS   # 18

FEATURE_NAMES = [
    "P4", "P8", "P12", "P24", "P48", "P72", "Imax24", "API72",
    "T_now", "T24_mean",
    "RH_now", "RH24_mean",
    "SWD_now", "SWD24_mean",
    "LWD24_mean",
    "Wind24_mean",
    "Pres_now"
]

# ==================== 读取配置 ====================
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config_all = yaml.safe_load(f)

preprocess_cfg = config_all["QX_multifeature_preprocessing"]

START_BEIJING = pd.to_datetime(preprocess_cfg["start_time"]).to_pydatetime()
END_BEIJING   = pd.to_datetime(preprocess_cfg["end_time"]).to_pydatetime()

excel_file = preprocess_cfg["lon_lat_path"]
output_file = preprocess_cfg["output"]

VAR_CFG = preprocess_cfg["variables"]

REQUIRED_KEYS = ["precip", "temp", "rh", "swd", "lwd", "wind", "pres"]
for key in REQUIRED_KEYS:
    if key not in VAR_CFG:
        raise KeyError(f"配置中缺少变量: {key}")


# ==================== 辅助函数 ====================
def parse_filename(filename):
    pattern = r"changjiang_(\d{7})\.(\d{2})\.nc"
    match = re.search(pattern, filename)
    if not match:
        return None

    date_part, hour = match.groups()
    year = int(date_part[:4])
    doy = int(date_part[4:])
    hour = int(hour)

    date = datetime(year, 1, 1) + timedelta(days=doy - 1)
    return datetime(year, date.month, date.day, hour)


def get_all_nc_files(folder):
    files = []
    for f in os.listdir(folder):
        if f.endswith(".nc"):
            dt = parse_filename(f)
            if dt is not None:
                files.append((dt, os.path.join(folder, f)))
    files.sort(key=lambda x: x[0])
    return files


def read_excel_stations(excel_path):
    """
    返回:
        {sheet_name: ndarray(n_points, 2)}  # (lon, lat)
    """
    xl = pd.ExcelFile(excel_path)
    stations = {}
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        lons = df["经度"].values
        lats = df["纬度"].values
        stations[sheet] = np.column_stack([lons, lats])
    return stations


def build_grid_kdtree(nc_file):
    ds = nc.Dataset(nc_file)
    lat_var = ds.variables["lat"][:]
    lon_var = ds.variables["lon"][:]

    if lat_var.ndim == 1 and lon_var.ndim == 1:
        lon_grid, lat_grid = np.meshgrid(lon_var, lat_var)
        lat_flat = lat_grid.flatten()
        lon_flat = lon_grid.flatten()
        grid_shape = (len(lat_var), len(lon_var))
    elif lat_var.ndim == 2 and lon_var.ndim == 2:
        lat_flat = lat_var.flatten()
        lon_flat = lon_var.flatten()
        grid_shape = lat_var.shape
    else:
        ds.close()
        raise ValueError("无法识别经纬度维度")

    grid_points = np.column_stack([lon_flat, lat_flat])  # (lon, lat)
    tree = cKDTree(grid_points)
    ds.close()
    return tree, grid_shape, len(grid_points)


def find_nearest_indices(tree, station_points):
    _, indices = tree.query(station_points, k=1)
    return indices


def read_variable_data(file_path, var_name):
    ds = nc.Dataset(file_path)
    if var_name not in ds.variables:
        ds.close()
        raise KeyError(f"{file_path} 中不存在变量: {var_name}")

    arr = ds.variables[var_name][:]
    if np.ma.isMaskedArray(arr):
        arr = arr.filled(np.nan)

    arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim == 3:
        arr = arr[0, :, :]
    elif arr.ndim == 2:
        pass
    else:
        ds.close()
        raise ValueError(f"{file_path} 中变量 {var_name} 维度异常: {arr.ndim}")

    ds.close()
    return arr.flatten()


def utc_to_beijing(utc_dt):
    return utc_dt + timedelta(hours=8)


def generate_time_series(start_dt, end_dt, freq_hours):
    times = []
    current = start_dt
    while current <= end_dt:
        times.append(current)
        current += timedelta(hours=freq_hours)
    return times


def calc_api72(p_hist_last_18, decay=0.9):
    """
    p_hist_last_18: 长度18，顺序为 [t-68h, ..., t-4h, t]
    """
    reversed_hist = p_hist_last_18[::-1]   # [t, t-4h, ..., t-68h]
    weights = decay ** np.arange(API_STEPS, dtype=np.float32)
    return float(np.sum(reversed_hist * weights))


def verify_required_utc_times(time_to_file, required_utc_times, var_key):
    missing = [t for t in required_utc_times if t not in time_to_file]
    if missing:
        show_n = min(10, len(missing))
        raise RuntimeError(
            f"变量 {var_key} 缺少 {len(missing)} 个UTC文件。\n"
            f"前 {show_n} 个缺失时刻: {missing[:show_n]}"
        )


def interpolate_to_target_times(source_times_beijing, source_values_2d, target_times_beijing):
    """
    source_values_2d: (T_source, S)
    返回: (T_target, S)
    """
    x_src = np.array([pd.Timestamp(t).timestamp() for t in source_times_beijing], dtype=np.float64)
    x_tgt = np.array([pd.Timestamp(t).timestamp() for t in target_times_beijing], dtype=np.float64)

    if x_tgt[0] < x_src[0] or x_tgt[-1] > x_src[-1]:
        raise ValueError(
            f"目标时间超出源时间范围。\n"
            f"source: {source_times_beijing[0]} ~ {source_times_beijing[-1]}\n"
            f"target: {target_times_beijing[0]} ~ {target_times_beijing[-1]}"
        )

    T_tgt = len(target_times_beijing)
    S = source_values_2d.shape[1]
    out = np.zeros((T_tgt, S), dtype=np.float32)

    for s in range(S):
        out[:, s] = np.interp(
            x_tgt,
            x_src,
            source_values_2d[:, s].astype(np.float64)
        ).astype(np.float32)

    return out


# ==================== 主程序 ====================
def main():
    # 1. 读取站点
    stations = read_excel_stations(excel_file)
    station_names = list(stations.keys())
    n_stations = len(station_names)
    print(f"共读取 {n_stations} 个站点: {station_names}")

    # 2. 构建网格树
    precip_files = get_all_nc_files(VAR_CFG["precip"]["nc_path"])
    if not precip_files:
        raise RuntimeError("未找到任何降雨NC文件")

    first_file = precip_files[0][1]
    tree, grid_shape, n_grid = build_grid_kdtree(first_file)
    print(f"网格形状: {grid_shape}, 总网格点数: {n_grid}")

    # 3. 预计算每个站点/子流域的网格索引
    station_indices = []
    for name in station_names:
        points = stations[name]
        indices = find_nearest_indices(tree, points)
        station_indices.append(indices)
        print(f"站点 {name}: {len(points)} 个气象点，索引范围 {indices.min()} ~ {indices.max()}")

    # 4. 目标输出时间（北京时间，4小时频次）
    output_times_beijing = generate_time_series(START_BEIJING, END_BEIJING, FREQ_HOURS)

    # 为构造 P72 / API72，需要往前追 68 小时
    hist_start_beijing = START_BEIJING - timedelta(hours=HIST_HOURS_FOR_API)
    target_times_for_features = generate_time_series(hist_start_beijing, END_BEIJING, FREQ_HOURS)

    print(f"历史起点（北京时间）: {hist_start_beijing}")
    print(f"输出起点（北京时间）: {START_BEIJING}")
    print(f"输出终点（北京时间）: {END_BEIJING}")
    print(f"特征时间轴长度: {len(target_times_for_features)}")
    print(f"最终输出样本数: {len(output_times_beijing)}")

    # 5. 为插值准备原始源时刻（UTC 3h 文件 -> 北京时间 3h 序列）
    query_start_utc = hist_start_beijing - timedelta(hours=8)
    query_end_utc = END_BEIJING - timedelta(hours=8)

    # 向下/向上补到 3h UTC 整点
    source_start_utc = query_start_utc.replace(minute=0, second=0, microsecond=0)
    source_start_utc = source_start_utc - timedelta(hours=source_start_utc.hour % 3)

    source_end_utc = query_end_utc.replace(minute=0, second=0, microsecond=0)
    if source_end_utc.hour % 3 != 0:
        source_end_utc = source_end_utc + timedelta(hours=(3 - source_end_utc.hour % 3))

    required_source_utc_times = []
    current = source_start_utc
    while current <= source_end_utc:
        required_source_utc_times.append(current)
        current += timedelta(hours=3)

    source_times_beijing = [utc_to_beijing(t) for t in required_source_utc_times]

    print(f"原始源时刻（UTC）: {source_start_utc} ~ {source_end_utc}")
    print(f"原始源时刻数: {len(required_source_utc_times)}")

    # 6. 为每个变量建立 UTC->file_path 映射，并检查覆盖
    var_time_to_file = {}
    for var_key in REQUIRED_KEYS:
        folder = VAR_CFG[var_key]["nc_path"]
        file_list = get_all_nc_files(folder)
        if not file_list:
            raise RuntimeError(f"{var_key} 文件夹中未找到NC文件: {folder}")

        time_to_file = {dt: path for dt, path in file_list}
        verify_required_utc_times(time_to_file, required_source_utc_times, var_key)
        var_time_to_file[var_key] = time_to_file

        all_utc_sorted = sorted(time_to_file.keys())
        print(
            f"{var_key:>6s} | 文件数={len(file_list)} | UTC覆盖: "
            f"{all_utc_sorted[0]} ~ {all_utc_sorted[-1]}"
        )

    # 7. 先在原始3h源时刻上生成站点聚合序列
    # precip 用 sum；其余变量用 mean
    source_series_dict = {
        key: np.zeros((len(required_source_utc_times), n_stations), dtype=np.float32)
        for key in REQUIRED_KEYS
    }

    print("\n开始读取原始NC并计算站点聚合气象序列（原始3h时刻）...")
    for t_idx, utc_dt in enumerate(required_source_utc_times):
        if (t_idx + 1) % 100 == 0 or t_idx == 0 or t_idx == len(required_source_utc_times) - 1:
            print(f"  处理源时刻 {t_idx + 1}/{len(required_source_utc_times)}: UTC {utc_dt}")

        flat_data = {}
        for var_key in REQUIRED_KEYS:
            file_path = var_time_to_file[var_key][utc_dt]
            var_name = VAR_CFG[var_key]["var_name"]
            flat_data[var_key] = read_variable_data(file_path, var_name)

        for s_idx, indices in enumerate(station_indices):
            # 降雨：sum（沿用旧脚本）
            source_series_dict["precip"][t_idx, s_idx] = np.nansum(flat_data["precip"][indices])

            # 其他变量：mean（沿用旧脚本）
            for var_key in ["temp", "rh", "swd", "lwd", "wind", "pres"]:
                source_series_dict[var_key][t_idx, s_idx] = np.nanmean(flat_data[var_key][indices])

    # 8. 重采样到北京时间4h目标时间轴
    print("\n开始将原始3h序列线性插值到北京时间4h目标时间轴...")
    target_series_dict = {}
    for var_key in REQUIRED_KEYS:
        print(f"  插值变量: {var_key}")
        target_series_dict[var_key] = interpolate_to_target_times(
            source_times_beijing=source_times_beijing,
            source_values_2d=source_series_dict[var_key],
            target_times_beijing=target_times_for_features
        )

    # 9. 构造丰富特征（按4h时间轴）
    n_times = len(output_times_beijing)
    n_features = len(FEATURE_NAMES)
    result = np.zeros((n_times, n_stations, n_features), dtype=np.float32)

    print("\n开始构造17个水质气象特征（4h版）...")
    output_offset = API_STEPS - 1   # 17

    for out_i in range(n_times):
        cur_idx = out_i + output_offset

        if (out_i + 1) % 200 == 0 or out_i == 0 or out_i == n_times - 1:
            print(f"  构造输出时刻 {out_i + 1}/{n_times}: {output_times_beijing[out_i]}")

        for s_idx in range(n_stations):
            # ---------- 降雨 ----------
            p_hist_72 = target_series_dict["precip"][cur_idx - 17: cur_idx + 1, s_idx]   # 18个4h点
            p_hist_24 = p_hist_72[-DAY24_STEPS:]   # 6
            p_hist_48 = p_hist_72[-DAY48_STEPS:]   # 12
            p_hist_12 = p_hist_72[-3:]             # 12h = 3步
            p_hist_8  = p_hist_72[-2:]             # 8h  = 2步

            if len(p_hist_72) != API_STEPS:
                raise ValueError(f"p_hist_72 长度异常: {len(p_hist_72)} at idx={out_i}")

            P4 = p_hist_72[-1]
            P8 = np.sum(p_hist_8)
            P12 = np.sum(p_hist_12)
            P24 = np.sum(p_hist_24)
            P48 = np.sum(p_hist_48)
            P72 = np.sum(p_hist_72)
            Imax24 = np.max(p_hist_24)
            API72 = calc_api72(p_hist_72, decay=API_DECAY)

            # ---------- 温度 ----------
            t_hist_24 = target_series_dict["temp"][cur_idx - 5: cur_idx + 1, s_idx]   # 6个4h点
            T_now = t_hist_24[-1]
            T24_mean = np.mean(t_hist_24)

            # ---------- 相对湿度 ----------
            rh_hist_24 = target_series_dict["rh"][cur_idx - 5: cur_idx + 1, s_idx]
            RH_now = rh_hist_24[-1]
            RH24_mean = np.mean(rh_hist_24)

            # ---------- 短波辐射 ----------
            swd_hist_24 = target_series_dict["swd"][cur_idx - 5: cur_idx + 1, s_idx]
            SWD_now = swd_hist_24[-1]
            SWD24_mean = np.mean(swd_hist_24)

            # ---------- 长波辐射 ----------
            lwd_hist_24 = target_series_dict["lwd"][cur_idx - 5: cur_idx + 1, s_idx]
            LWD24_mean = np.mean(lwd_hist_24)

            # ---------- 风速 ----------
            wind_hist_24 = target_series_dict["wind"][cur_idx - 5: cur_idx + 1, s_idx]
            Wind24_mean = np.mean(wind_hist_24)

            # ---------- 压强 ----------
            Pres_now = target_series_dict["pres"][cur_idx, s_idx]

            result[out_i, s_idx, :] = np.array([
                P4, P8, P12, P24, P48, P72, Imax24, API72,
                T_now, T24_mean,
                RH_now, RH24_mean,
                SWD_now, SWD24_mean,
                LWD24_mean,
                Wind24_mean,
                Pres_now
            ], dtype=np.float32)

    # 10. 保存
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.save(output_file, result)

    meta_file = output_file.replace(".npy", "_metadata.npz")
    np.savez(
        meta_file,
        times=np.array(output_times_beijing, dtype="datetime64[s]"),
        stations=np.array(station_names, dtype=object),
        feature_names=np.array(FEATURE_NAMES, dtype=object)
    )

    feature_file = output_file.replace(".npy", "_feature_names.pkl")
    with open(feature_file, "wb") as f:
        pickle.dump(FEATURE_NAMES, f)

    print("\n处理完成。")
    print(f"输出结果: {output_file}")
    print(f"元数据:   {meta_file}")
    print(f"特征名:   {feature_file}")
    print(f"输出数组形状: {result.shape}")
    print(f"特征数: {len(FEATURE_NAMES)}")
    print(f"特征顺序: {FEATURE_NAMES}")


if __name__ == "__main__":
    main()