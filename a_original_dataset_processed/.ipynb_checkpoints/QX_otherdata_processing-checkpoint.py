#气象数据处理脚本（除降水外其他）：1.转换UTC时间到北京时间 2.一天8测变成一天6测，没有的时间点做线性插值 3.根据excel表格里的经纬度去做某子流域内所有气象网格的平均
import os
import re
import numpy as np
import pandas as pd
import netCDF4 as nc
from scipy.spatial import cKDTree
from datetime import datetime, timedelta, time

# ==================== 请修改以下超参数 ====================
nc_folder = '/home/fanyunkai/FYK_GCNLSTM/dataset/QX_original_dataset/changjiang_lwd'   # NC文件所在文件夹
excel_file = '/home/fanyunkai/FYK_GCNLSTM/xiangjiang/xiangjiang.xlsx'                     # 站点Excel文件
output_file = '/home/fanyunkai/FYK_GCNLSTM/xiangjiang/xiangjiang_processed_lwd_data.npy'  # 输出文件

# 生成文件的北京时间范围
START_BEIJING = datetime(2020, 11, 9, 0, 0, 0)      # 起始日期（包含当天0点）
END_BEIJING = datetime(2025, 9, 30, 23, 59, 59)    # 结束日期（包含当天20点）

# NC文件中的温度变量名（请根据实际数据修改）
VAR_NAME = 'downward_longwave_radiation'        # 例如 't2m', 'temperature', 'air_temperature' 等
# ==========================================================

def parse_filename(filename):
    """解析文件名：changjiang_YYYYDOY.HH.nc (UTC时间)"""
    pattern = r'changjiang_(\d{7})\.(\d{2})\.nc'
    match = re.search(pattern, filename)
    if not match:
        return None
    date_part, hour = match.groups()
    year = int(date_part[:4])
    doy = int(date_part[4:])
    hour = int(hour)
    date = datetime(year, 1, 1) + timedelta(days=doy-1)
    return datetime(year, date.month, date.day, hour)

def get_all_nc_files(folder):
    files = []
    for f in os.listdir(folder):
        if f.endswith('.nc'):
            dt = parse_filename(f)
            if dt:
                files.append((dt, os.path.join(folder, f)))
    files.sort(key=lambda x: x[0])
    return files

def read_excel_stations(excel_path):
    xl = pd.ExcelFile(excel_path)
    stations = {}
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        lons = df['经度'].values
        lats = df['纬度'].values
        stations[sheet] = np.column_stack([lats, lons])
    return stations

def build_grid_kdtree(nc_file):
    ds = nc.Dataset(nc_file)
    lat_var = ds.variables['lat'][:]
    lon_var = ds.variables['lon'][:]
    
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
        raise ValueError("无法识别的经纬度维度")
    
    grid_points = np.column_stack([lat_flat, lon_flat])
    tree = cKDTree(grid_points)
    ds.close()
    return tree, grid_shape, len(grid_points)

def find_nearest_indices(tree, station_points):
    distances, indices = tree.query(station_points, k=1)
    return indices

def read_var_data(file_path, var_name, fill_value=-9999.0):
    """
    从NC文件读取指定变量，返回二维数组(lat, lon)的展平一维数组
    将填充值替换为0（可根据需要修改）
    """
    ds = nc.Dataset(file_path)
    data = ds.variables[var_name][0, :, :]  # (lat, lon)
    if hasattr(ds.variables[var_name], '_FillValue'):
        fill = ds.variables[var_name]._FillValue
    else:
        fill = fill_value
    data = np.where(data == fill, 0.0, data)  # 缺失值替换为0（可根据需要改为 np.nan）
    ds.close()
    return data.flatten()

def main():
    # 1. 读取站点
    stations = read_excel_stations(excel_file)
    station_names = list(stations.keys())
    n_stations = len(station_names)
    print(f"共读取 {n_stations} 个站点")

    # 2. 获取NC文件（UTC时间）
    nc_files = get_all_nc_files(nc_folder)
    if not nc_files:
        print("未找到任何NC文件")
        return
    print(f"共找到 {len(nc_files)} 个NC文件")

    # 构建UTC时间到文件路径的映射
    time_to_file = {dt: path for dt, path in nc_files}
    source_times = list(time_to_file.keys())
    first_utc = min(source_times)
    last_utc = max(source_times)

    # 3. 构建KDTree并预计算索引
    first_file = nc_files[0][1]
    tree, grid_shape, n_grid = build_grid_kdtree(first_file)
    print(f"网格形状: {grid_shape}, 总网格点数: {n_grid}")

    station_indices = []
    for name in station_names:
        points = stations[name]
        indices = find_nearest_indices(tree, points)
        station_indices.append(indices)
        print(f"站点 {name}: {len(points)} 个气象点")

    # 4. 生成所有输出时刻（北京时间），按指定范围
    output_times_beijing = []
    current = START_BEIJING
    while current.date() <= END_BEIJING.date():
        for hour in [0, 4, 8, 12, 16, 20]:
            dt = datetime(current.year, current.month, current.day, hour)
            # 只添加在结束日期及之前的时刻
            if dt <= END_BEIJING:
                output_times_beijing.append(dt)
        current += timedelta(days=1)
    print(f"待处理输出时刻数（北京时间）: {len(output_times_beijing)}")

    # 5. 初始化结果数组
    result = np.zeros((len(output_times_beijing), n_stations))

    # 6. 逐北京时间时刻处理
    for i, beijing_dt in enumerate(output_times_beijing):
        hour_beijing = beijing_dt.hour
        date_beijing = beijing_dt.date()
        print(f"处理北京时间 {beijing_dt}")

        if hour_beijing == 0:
            # 北京0点 = 前一日23（UTC 15）与当日2（UTC 前一日18）插值
            utc1 = datetime.combine(date_beijing - timedelta(days=1), time(15, 0))
            utc2 = datetime.combine(date_beijing - timedelta(days=1), time(18, 0))
            w1, w2 = 2/3, 1/3
            if utc1 not in time_to_file or utc2 not in time_to_file:
                print(f"  警告：缺失文件 {utc1} 或 {utc2}，跳过")
                continue
            d1 = read_var_data(time_to_file[utc1], VAR_NAME)
            d2 = read_var_data(time_to_file[utc2], VAR_NAME)
            for s_idx, indices in enumerate(station_indices):
                # 先对每个网格点插值，再对站点内网格点取平均
                interp = w1 * d1[indices] + w2 * d2[indices]
                result[i, s_idx] = np.mean(interp)

        elif hour_beijing == 4:
            # 北京4点 = 当日2（UTC 前一日18）与5（UTC 前一日21）插值
            utc1 = datetime.combine(date_beijing - timedelta(days=1), time(18, 0))
            utc2 = datetime.combine(date_beijing - timedelta(days=1), time(21, 0))
            w1, w2 = 1/3, 2/3
            if utc1 not in time_to_file or utc2 not in time_to_file:
                print(f"  警告：缺失文件 {utc1} 或 {utc2}，跳过")
                continue
            d1 = read_var_data(time_to_file[utc1], VAR_NAME)
            d2 = read_var_data(time_to_file[utc2], VAR_NAME)
            for s_idx, indices in enumerate(station_indices):
                interp = w1 * d1[indices] + w2 * d2[indices]
                result[i, s_idx] = np.mean(interp)

        elif hour_beijing == 8:
            # 北京8点 = UTC当日00，直接读取
            utc = datetime.combine(date_beijing, time(0, 0))
            if utc not in time_to_file:
                print(f"  警告：缺失文件 {utc}，跳过")
                continue
            data_flat = read_var_data(time_to_file[utc], VAR_NAME)
            for s_idx, indices in enumerate(station_indices):
                result[i, s_idx] = np.mean(data_flat[indices])

        elif hour_beijing == 12:
            # 北京12点 = 当日11（UTC 03）与14（UTC 06）插值
            utc1 = datetime.combine(date_beijing, time(3, 0))
            utc2 = datetime.combine(date_beijing, time(6, 0))
            w1, w2 = 2/3, 1/3
            if utc1 not in time_to_file or utc2 not in time_to_file:
                print(f"  警告：缺失文件 {utc1} 或 {utc2}，跳过")
                continue
            d1 = read_var_data(time_to_file[utc1], VAR_NAME)
            d2 = read_var_data(time_to_file[utc2], VAR_NAME)
            for s_idx, indices in enumerate(station_indices):
                interp = w1 * d1[indices] + w2 * d2[indices]
                result[i, s_idx] = np.mean(interp)

        elif hour_beijing == 16:
            # 北京16点 = 当日14（UTC 06）与17（UTC 09）插值
            utc1 = datetime.combine(date_beijing, time(6, 0))
            utc2 = datetime.combine(date_beijing, time(9, 0))
            w1, w2 = 1/3, 2/3
            if utc1 not in time_to_file or utc2 not in time_to_file:
                print(f"  警告：缺失文件 {utc1} 或 {utc2}，跳过")
                continue
            d1 = read_var_data(time_to_file[utc1], VAR_NAME)
            d2 = read_var_data(time_to_file[utc2], VAR_NAME)
            for s_idx, indices in enumerate(station_indices):
                interp = w1 * d1[indices] + w2 * d2[indices]
                result[i, s_idx] = np.mean(interp)

        elif hour_beijing == 20:
            # 北京20点 = UTC当日12，直接读取
            utc = datetime.combine(date_beijing, time(12, 0))
            if utc not in time_to_file:
                print(f"  警告：缺失文件 {utc}，跳过")
                continue
            data_flat = read_var_data(time_to_file[utc], VAR_NAME)
            for s_idx, indices in enumerate(station_indices):
                result[i, s_idx] = np.mean(data_flat[indices])

        else:
            continue

    # 7. 保存
    np.save(output_file, result)
    meta_file = output_file.replace('.npy', '_metadata.npz')
    np.savez(meta_file, times=output_times_beijing, stations=station_names)
    print(f"结果已保存至 {output_file}")
    print(f"元数据保存至 {meta_file}")

if __name__ == "__main__":
    main()