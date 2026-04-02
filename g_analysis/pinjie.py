import os
import pickle
import numpy as np
import pandas as pd

# ==================== 路径配置 ====================
# 原水质输入（没有 runoff 的 33 维输入）
WQ_NPY_PATH = "/home/fanyunkai/FYK_GCNLSTM/xiangjiang11_multiQXfeature/preprocessed_data.npy"
WQ_META_PATH = "/home/fanyunkai/FYK_GCNLSTM/xiangjiang11_multiQXfeature/metadata.npz"
WQ_FEATURE_NAMES_PATH = "/home/fanyunkai/FYK_GCNLSTM/xiangjiang11_multiQXfeature/feature_names.pkl"

# 预测得到的 runoff
RUNOFF_PRED_PATH = "/home/fanyunkai/FYK_GCNLSTM_FLOW/xiangjiang/runoff_inference_result/output/pred_runoff_log.npy"
RUNOFF_META_PATH = "/home/fanyunkai/FYK_GCNLSTM_FLOW/xiangjiang/runoff_inference_result/output/metadata.npz"

# 输出
OUTPUT_DIR = "/home/fanyunkai/FYK_GCNLSTM/xiangjiang11_multiQXfeature/runoff_inference"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_NPY_PATH = os.path.join(OUTPUT_DIR, "preprocessed_data_with_runoff.npy")
OUTPUT_FEATURE_NAMES_PATH = os.path.join(OUTPUT_DIR, "feature_names_with_runoff.pkl")
OUTPUT_META_PATH = os.path.join(OUTPUT_DIR, "metadata_with_runoff.npz")
OUTPUT_RUNOFF_ALIGNED_PATH = os.path.join(OUTPUT_DIR, "runoff_aligned.npy")


# ==================== 读取原水质数据 ====================
print("加载原水质输入数据...")
wq_data = np.load(WQ_NPY_PATH).astype(np.float32)   # (T1, N, 33)
with open(WQ_FEATURE_NAMES_PATH, "rb") as f:
    wq_feature_names = pickle.load(f)

wq_meta = np.load(WQ_META_PATH, allow_pickle=True)
wq_times = pd.to_datetime(wq_meta["times"])
wq_stations = [str(x) for x in wq_meta["stations"].tolist()]

T1, N1, F1 = wq_data.shape
print(f"原水质数据形状: {wq_data.shape}")
print(f"原特征数: {len(wq_feature_names)}")


# ==================== 读取 runoff 预测 ====================
print("\n加载 runoff 预测数据...")
runoff_pred = np.load(RUNOFF_PRED_PATH).astype(np.float32)   # (T2, N, 1)

runoff_meta = np.load(RUNOFF_META_PATH, allow_pickle=True)
runoff_times = pd.to_datetime(runoff_meta["times"])
runoff_stations = [str(x) for x in runoff_meta["stations"].tolist()]

T2, N2, F2 = runoff_pred.shape
print(f"runoff 预测数据形状: {runoff_pred.shape}")

if F2 != 1:
    raise ValueError(f"runoff 预测最后一维应为 1，实际为 {F2}")

if N1 != N2:
    raise ValueError(f"站点数不一致：水质={N1}, runoff={N2}")

if wq_stations != runoff_stations:
    raise ValueError(
        "水质数据与 runoff 预测的站点顺序不一致。\n"
        f"WQ stations: {wq_stations}\n"
        f"Runoff stations: {runoff_stations}"
    )

# 防止重复添加
if "runoff" in wq_feature_names:
    raise ValueError("原 feature_names 中已经包含 runoff，不能重复拼接。")


# ==================== 按时间对齐 runoff 到水质时间轴 ====================
print("\n按时间戳对齐 runoff 到水质时间轴...")

# 初始化全 NaN
runoff_aligned = np.full((len(wq_times), N1, 1), np.nan, dtype=np.float32)

# 建立时间索引映射
wq_time_to_idx = {t: i for i, t in enumerate(wq_times)}
matched = 0

for i, t in enumerate(runoff_times):
    if t in wq_time_to_idx:
        j = wq_time_to_idx[t]
        runoff_aligned[j, :, :] = runoff_pred[i, :, :]
        matched += 1

print(f"成功匹配的时间点数: {matched}/{len(runoff_times)}")
print(f"对齐后 runoff 形状: {runoff_aligned.shape}")

# ==================== 只对前面缺失部分做后向填充 ====================
print("\n对前部缺失时间点做后向填充...")

for s in range(N1):
    series = pd.Series(runoff_aligned[:, s, 0], index=wq_times)

    # 找到第一个非 NaN 位置
    first_valid_idx = series.first_valid_index()

    if first_valid_idx is None:
        raise ValueError(f"站点 {wq_stations[s]} 的 runoff 预测全是 NaN，无法后向填充。")

    first_valid_pos = series.index.get_loc(first_valid_idx)

    # 仅填前面的缺失，不动后面
    series.iloc[:first_valid_pos] = series.iloc[first_valid_pos]

    runoff_aligned[:, s, 0] = series.to_numpy(dtype=np.float32)

# 检查前部是否填好了
head_nan_count = np.isnan(runoff_aligned[:30]).sum()
print(f"前30个时间步中剩余 NaN 数量: {head_nan_count}")

# 如果你想把中间极少数 NaN 也补掉，可以打开下面这段
# for s in range(N1):
#     series = pd.Series(runoff_aligned[:, s, 0], index=wq_times)
#     series = series.bfill()
#     runoff_aligned[:, s, 0] = series.to_numpy(dtype=np.float32)


# ==================== 拼接成 (10722, 28, 34) ====================
print("\n拼接 runoff 到原水质输入...")

new_data = np.concatenate([wq_data, runoff_aligned], axis=-1).astype(np.float32)
new_feature_names = list(wq_feature_names) + ["runoff"]

print(f"拼接后数据形状: {new_data.shape}")
print(f"拼接后特征数: {len(new_feature_names)}")
print(f"最后一列特征名: {new_feature_names[-1]}")

# ==================== 保存 ====================
np.save(OUTPUT_NPY_PATH, new_data)
np.save(OUTPUT_RUNOFF_ALIGNED_PATH, runoff_aligned)

with open(OUTPUT_FEATURE_NAMES_PATH, "wb") as f:
    pickle.dump(new_feature_names, f)

np.savez(
    OUTPUT_META_PATH,
    times=np.array(wq_times, dtype="datetime64[s]"),
    stations=np.array(wq_stations, dtype=object),
    feature_names=np.array(new_feature_names, dtype=object)
)

print("\n完成。")
print(f"输出数据: {OUTPUT_NPY_PATH}")
print(f"对齐后的 runoff: {OUTPUT_RUNOFF_ALIGNED_PATH}")
print(f"特征名: {OUTPUT_FEATURE_NAMES_PATH}")
print(f"元数据: {OUTPUT_META_PATH}")