"""
评估 GAT-LSTM 模型在不同缺失模式下的性能
使用方法：python evaluate_missing_patterns.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import yaml
import pickle

# ==================== 配置区 ====================
CONFIG_PATH = "/home/fanyunkai/FYK_GCNLSTM/configs.yaml"
MODEL_PATH = "/home/fanyunkai/FYK_GCNLSTM/xiangjiang6/results/best_model.pth"  # 最佳模型路径
OUTPUT_DIR = "/home/fanyunkai/FYK_GCNLSTM/xiangjiang6/results"  # 结果保存目录
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载配置
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config_all = yaml.safe_load(f)

# 模型训练配置
cfg = config_all.get('GATLSTM_training', {})
# 数据预处理配置（用于获取站点数等）
preprocess_cfg = config_all.get('data_preprocessing', {})

# 模型超参数（必须与训练时一致）
SEQ_LEN = cfg.get('SEQ_LEN', 6)
BATCH_SIZE = cfg.get('BATCH_SIZE', 32)
GAT_HIDDEN = cfg.get('GCN_HIDDEN', 16)
LSTM_HIDDEN = cfg.get('LSTM_HIDDEN', 64)
OUTPUT_DIM = cfg.get('OUTPUT_DIM', 1)
NUM_HEADS = cfg.get('NUM_HEADS', 8)
feature_names = cfg.get('feature_names', [])
INPUT_DIM = len(feature_names)  # 特征数（22）
TARGET_NAMES = cfg.get('target_names', ['总氮'])
TARGET_IDXS = [feature_names.index(name) for name in TARGET_NAMES]

# 站点数（从预处理配置中获取）
sites_names = preprocess_cfg.get('sites_names', [])
N = len(sites_names)  # 站点数（28）
if N == 0:
    raise ValueError("未找到站点列表，请检查配置中的 sites_names")

# 水质指标列表（用于掩码提取）
wq_names = preprocess_cfg.get('wq_names', ['总氮', '总磷', '水温', 'pH', '溶解氧'])
# 目标变量在 wq_names 中的索引（用于掩码）
target_wq_indices = [wq_names.index(name) for name in TARGET_NAMES]

# 测试数据目录（与训练输出目录相同，即预处理输出目录）
TEST_DATA_DIR = preprocess_cfg.get('output_path', '/home/fanyunkai/FYK_GCNLSTM/xiangjiang5_sub_gat/mask_test')

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 模型定义（必须与训练时一致） ====================
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, adj, num_heads=1, concat=True, dropout=0.2, alpha=0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        self.register_buffer("adj", adj)

        self.W = nn.ModuleList([
            nn.Linear(in_features, out_features, bias=False) for _ in range(num_heads)
        ])
        self.a = nn.ParameterList([
            nn.Parameter(torch.empty(2 * out_features, 1)) for _ in range(num_heads)
        ])
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.2)

        for w in self.W:
            nn.init.xavier_uniform_(w.weight)
        for a in self.a:
            nn.init.xavier_uniform_(a)

    def forward(self, x):
        batch, N, _ = x.size()
        adj_mask = self.adj.unsqueeze(0).expand(batch, -1, -1)

        head_outputs = []
        for h in range(self.num_heads):
            Wh = self.W[h](x)
            a = self.a[h]

            Wh_i = Wh.unsqueeze(2).expand(-1, -1, N, -1)
            Wh_j = Wh.unsqueeze(1).expand(-1, N, -1, -1)
            a_input = torch.cat([Wh_i, Wh_j], dim=-1)
            e = self.leakyrelu(torch.matmul(a_input, a).squeeze(-1))
            e = e.masked_fill(adj_mask == 0, float('-inf'))
            attention = torch.softmax(e, dim=-1)
            attention = self.dropout(attention)
            h_prime = torch.matmul(attention, Wh)
            head_outputs.append(h_prime)

        if self.concat:
            output = torch.cat(head_outputs, dim=-1)
        else:
            output = torch.mean(torch.stack(head_outputs, dim=0), dim=0)
        return output


class GAT_LSTM(nn.Module):
    def __init__(self, node_num, input_dim, gat_dim, lstm_hidden, output_dim, adj, num_heads=1):
        super().__init__()
        self.gat = GraphAttentionLayer(input_dim, gat_dim, adj, num_heads=num_heads, concat=True)
        lstm_input_dim = gat_dim * num_heads
        self.lstm = nn.LSTM(lstm_input_dim, lstm_hidden, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, output_dim)

    def forward(self, x):
        batch, seq_len, node_num, _ = x.shape
        x_gat = []
        for t in range(seq_len):
            xt = x[:, t, :, :]
            ht = self.gat(xt)
            x_gat.append(ht)
        x_gat = torch.stack(x_gat, dim=1)
        x_gat = x_gat.permute(0, 2, 1, 3).contiguous()
        x_gat = x_gat.view(batch * node_num, seq_len, -1)
        lstm_out, _ = self.lstm(x_gat)
        last_out = lstm_out[:, -1, :]
        pred = self.fc(last_out)
        pred = pred.view(batch, node_num, -1)
        return pred

# ==================== 辅助函数 ====================
def build_adj_tensor(adj_path, N):
    """加载邻接矩阵并转换为张量"""
    file_ext = os.path.splitext(adj_path)[1].lower()
    if file_ext == '.npy':
        adj = np.load(adj_path)
    elif file_ext == '.csv':
        adj = np.loadtxt(adj_path, delimiter=',')
        if adj.shape[0] == N and adj.shape[1] == N + 1:
            adj = adj[:, 1:]
        elif adj.shape[0] == N + 1 and adj.shape[1] == N + 1:
            adj = adj[1:, 1:]
    else:
        raise ValueError(f"不支持的文件格式: {file_ext}")
    adj_binary = (adj != 0).astype(np.float32)
    np.fill_diagonal(adj_binary, 1.0)
    return torch.tensor(adj_binary, dtype=torch.float32)

def build_sliding_window(data, seq_len, target_idxs):
    """构建滑动窗口样本
    data: (T, N, F)
    返回: X (T-seq_len, seq_len, N, F), y (T-seq_len, N, len(target_idxs))
    """
    T, N, F = data.shape
    samples_X = []
    samples_y = []
    for t in range(seq_len, T):
        X_window = data[t-seq_len:t]          # (seq_len, N, F)
        y_t = data[t][:, target_idxs]         # (N, len(target_idxs))
        samples_X.append(X_window)
        samples_y.append(y_t)
    X = np.stack(samples_X, axis=0)
    y = np.stack(samples_y, axis=0)
    return X, y

def evaluate_model(model, test_loader, device, original_mask):
    """评估模型在测试集上的性能
    test_loader: 返回 (X, y, mask) 的 DataLoader（mask 是原始掩码，用于筛选真实观测）
    original_mask: 原始掩码数组 (n_samples, N, n_targets)，用于全局筛选（也可由 loader 提供）
    返回: R2, RMSE, MAE
    """
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_X, batch_y, batch_mask in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            batch_mask = batch_mask.to(device)
            pred = model(batch_X)
            # 只保留原始观测位置（mask=1）
            valid = (batch_mask > 0.5)
            if valid.sum() == 0:
                continue
            pred_valid = pred[valid].cpu().numpy()
            target_valid = batch_y[valid].cpu().numpy()
            all_preds.append(pred_valid)
            all_targets.append(target_valid)
    if not all_preds:
        return np.nan, np.nan, np.nan
    pred_concat = np.concatenate(all_preds)
    target_concat = np.concatenate(all_targets)
    r2 = r2_score(target_concat, pred_concat)
    rmse = np.sqrt(mean_squared_error(target_concat, pred_concat))
    mae = mean_absolute_error(target_concat, pred_concat)
    return r2, rmse, mae

def main():
    print(f"使用设备: {DEVICE}")
    # 加载邻接矩阵
    adj_path = cfg.get('ADJ_PATH')
    adj_tensor = build_adj_tensor(adj_path, N)
    adj_tensor = adj_tensor.to(DEVICE)

    # 初始化模型并加载权重
    model = GAT_LSTM(
        node_num=N,
        input_dim=INPUT_DIM,
        gat_dim=GAT_HIDDEN,
        lstm_hidden=LSTM_HIDDEN,
        output_dim=OUTPUT_DIM,
        adj=adj_tensor,
        num_heads=NUM_HEADS
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print("模型加载成功")

    # 循环处理四种缺失模式
    results = {}
    for mode in [1, 2, 3, 4]:
        print(f"\n===== 处理缺失模式 {mode} =====")
        # 加载测试数据
        test_data_path = os.path.join(TEST_DATA_DIR, f"test_data_mode{mode}.npy")
        original_mask_path = os.path.join(TEST_DATA_DIR, f"test_original_mask_mode{mode}.npy")
        if not os.path.exists(test_data_path):
            print(f"跳过模式 {mode}：文件 {test_data_path} 不存在")
            continue
        data = np.load(test_data_path)                 # (T_test, N, F)
        original_mask = np.load(original_mask_path)    # (T_test, N, n_targets)
        # 检查数据形状
        T_test, N_data, F_data = data.shape
        assert N_data == N, f"站点数不匹配: {N_data} vs {N}"
        # 构建滑动窗口样本
        X_test, y_test = build_sliding_window(data, SEQ_LEN, TARGET_IDXS)
        # 构建对应的原始掩码（需要与滑动窗口对齐）
        n_samples = X_test.shape[0]
        mask_test = np.zeros((n_samples, N, OUTPUT_DIM), dtype=np.float32)
        for idx, t in enumerate(range(SEQ_LEN, T_test)):
            # 使用目标变量在 wq_names 中的索引提取掩码列
            mask_test[idx] = original_mask[t][:, target_wq_indices]  # (N, OUTPUT_DIM)
        # 转换为张量
        X_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_tensor = torch.tensor(y_test, dtype=torch.float32)
        mask_tensor = torch.tensor(mask_test, dtype=torch.float32)

        # 创建 DataLoader
        dataset = TensorDataset(X_tensor, y_tensor, mask_tensor)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=torch.cuda.is_available())
        # 评估
        r2, rmse, mae = evaluate_model(model, loader, DEVICE, mask_tensor)
        results[mode] = (r2, rmse, mae)
        print(f"模式 {mode}: R² = {r2:.4f}, RMSE = {rmse:.4f}, MAE = {mae:.4f}")

    # 保存结果到文件
    result_file = os.path.join(OUTPUT_DIR, "missing_pattern_evaluation.txt")
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("Missing Pattern Evaluation Results\n")
        f.write("==================================\n")
        for mode, (r2, rmse, mae) in results.items():
            f.write(f"Mode {mode}:\n")
            f.write(f"  R²   : {r2:.6f}\n")
            f.write(f"  RMSE : {rmse:.6f}\n")
            f.write(f"  MAE  : {mae:.6f}\n")
            f.write("\n")
    print(f"\n结果已保存至 {result_file}")

if __name__ == "__main__":
    main()