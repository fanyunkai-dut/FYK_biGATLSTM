import numpy as np

# 加载 .npy 文件
data = np.load('/home/fanyunkai/FYK_GCNLSTM/xiangjiang5_sub_gat/QX/xiangjiang_processed_precipitation_data.npy', allow_pickle=True)

# 打印数据的形状，查看它的维度
print("数据的形状：")
print(data.shape)

# 如果是二维数据，打印前100行和前50列（与原代码一致）
if data.ndim == 2:
    print("前50行数据：")
    print(data[:50, :])
    print("前50列数据：")
    print(data[:, :50])
# 如果是三维数据，打印前两个样本的对应切片
elif data.ndim == 3:
    print("三维数据，打印前两个样本的前5行和前5列：")
    np.set_printoptions(threshold=np.inf)
    # 样本数至少为2，避免索引越界
    n_samples = max(1, data.shape[1])
    for i in range(n_samples):
        print(f"样本 {i}:")
        # 前50行（第二维）所有列（第三维）
        print(data[:10, i, :])
        # 所有行（第二维）前50列（第三维）
else:
    print("加载的数据不是二维或三维数组，请检查文件内容。")