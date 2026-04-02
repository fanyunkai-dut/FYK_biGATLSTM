import numpy as np

# 加载数据
data = np.load('/home/fanyunkai/FYK_GCNLSTM/xiangjiang11_multiQXfeature/preprocessed_data.npy', allow_pickle=True)  # 请替换为实际文件路径

# 检查维度
if data.ndim == 3:
    print(f"三维数据，形状: {data.shape}")
    # 查看第15个站点（索引14）的前5个时间步的所有特征
    station_idx = 0
    time_steps = 3
    if station_idx < data.shape[1]:
        subset = data[:time_steps, station_idx, :]
        print(f"第{station_idx+1}个站点的前{time_steps}个时间步的所有特征：")
        print(subset)
    else:
        print(f"站点索引超出范围，有效范围0-{data.shape[1]-1}")
else:
    print("数据不是三维数组，请检查。")