import pickle

# PKL文件路径
path = '/home/fanyunkai/FYK_GCNLSTM/xiangjiang11_multiQXfeature/feature_names.pkl'

# 以二进制读取模式打开PKL文件
with open(path, 'rb') as file:
# 使用pickle模块加载文件内容
 data = pickle.load(file)

# 打印加载的数据
print(data)