#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
分站点评估绘图脚本（支持中文、交互式HTML）
自动处理 train, val, test 三个数据集，分别生成图表。
输入：{train|val|test}_pred.npy, {train|val|test}_target.npy, {train|val|test}_mask.npy (位于 OUTPUT_DIR)
输出：在 OUTPUT_DIR/plots/{train|val|test}/ 下生成每个参数的散点图和时间序列HTML
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import yaml

# 设置中文字体，避免警告
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 尝试导入plotly（可选）
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("警告: plotly 未安装，无法生成交互式HTML时间序列。请运行: pip install plotly")

# ==================== 从 configs.yaml 读取配置 ====================
CONFIG_PATH = "configs.yaml"  # 可根据需要改为绝对路径

with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config_all = yaml.safe_load(f)

cfg = config_all.get('stations_val', {})

OUTPUT_DIR = cfg.get('OUTPUT_DIR', '/home/fanyunkai/FYK_GCNLSTM/xiangjiang2/results')
PARAM_NAMES = cfg.get('PARAM_NAMES', ['总氮', '总磷', 'pH', '水温','溶解氧'])
STATION_NAMES = cfg.get('STATION_NAMES', None)       # 若为 null 则使用默认站点名称
MAKE_PLOTLY_TS = cfg.get('MAKE_PLOTLY_TS', True)
DOWNSAMPLE = cfg.get('DOWNSAMPLE', 1)
SHOW_ONLY_KNOWN = cfg.get('SHOW_ONLY_KNOWN', True)
VAL_STYLE = cfg.get('VAL_STYLE', "rug")              # 可选 'rug' 或 'band'
# ==================================================

def _safe_xy(gt, pred, mask):
    """提取被 mask 标记且有限值的 (x,y) 点"""
    mask = mask.astype(bool)                     # 确保 mask 为布尔型
    m = mask & np.isfinite(gt) & np.isfinite(pred)
    x = gt[m].astype(np.float64).ravel()
    y = pred[m].astype(np.float64).ravel()
    return x, y

def _metrics(x, y):
    """计算 RMSE, MAE, R2 等指标"""
    if x.size == 0:
        return {"n": 0, "rmse": np.nan, "mae": np.nan, "r2": np.nan}
    err = y - x
    rmse = np.sqrt(np.mean(err**2))
    mae = np.mean(np.abs(err))
    ss_res = np.sum(err**2)
    ss_tot = np.sum((x - np.mean(x))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"n": int(x.size), "rmse": float(rmse), "mae": float(mae), "r2": float(r2)}

def _scatter_xy(x, y, title, out_path):
    """绘制散点图并保存为 PNG"""
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=8, alpha=0.4)
    if x.size > 0 and y.size > 0:
        lo = np.nanmin([np.min(x), np.min(y)])
        hi = np.nanmax([np.max(x), np.max(y)])
        plt.plot([lo, hi], [lo, hi], linewidth=1, color='gray', linestyle='--')  # y=x 线
        plt.xlim(lo, hi)
        plt.ylim(lo, hi)
    plt.xlabel("实测值")
    plt.ylabel("预测值")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    # 确保目录存在
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"  已保存散点图: {out_path}")

def plot_timeseries_plotly(gt_row, pred_row, mask_row, title, out_path_html,
                           show_only_known=True, val_style="rug", downsample=1):
    """
    交互式 Plotly 时间序列（与别人代码类似，但只用了一个 mask）
    - 黑点：实测
    - 红点：预测
    - mask 位置：用 rug 或 band 标记（这里 mask 视作已知点位置）
    """
    if not HAS_PLOTLY:
        print("  警告: plotly未安装，跳过HTML生成")
        return

    gt_row = np.asarray(gt_row)
    pred_row = np.asarray(pred_row)
    mask_row = np.asarray(mask_row).astype(bool)

    T = gt_row.shape[0]
    t = np.arange(T)

    if show_only_known:
        known = mask_row & np.isfinite(gt_row) & np.isfinite(pred_row)
    else:
        known = np.isfinite(gt_row) & np.isfinite(pred_row)

    t_k = t[known]
    gt_k = gt_row[known]
    pred_k = pred_row[known]

    if downsample > 1 and t_k.size > 0:
        idx = np.arange(0, t_k.size, downsample)
        t_k = t_k[idx]
        gt_k = gt_k[idx]
        pred_k = pred_k[idx]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=t_k, y=gt_k,
        mode="markers",
        name="实测",
        marker=dict(size=5, color="black"),
        hovertemplate="时间=%{x}<br>实测=%{y}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=t_k, y=pred_k,
        mode="markers",
        name="预测",
        marker=dict(size=5, color="red", opacity=0.6),
        hovertemplate="时间=%{x}<br>预测=%{y}<extra></extra>"
    ))

    # 标记所有 mask 为 True 的位置（即已知点位置）
    known_times = np.where(mask_row)[0]

    shapes = []
    if val_style == "band":
        for tt in known_times:
            shapes.append(dict(
                type="rect",
                xref="x", yref="paper",
                x0=tt - 0.5, x1=tt + 0.5,
                y0=0, y1=1,
                fillcolor="rgba(0,0,0,0.06)",
                line=dict(width=0)
            ))
    else:  # "rug"
        for tt in known_times:
            shapes.append(dict(
                type="line",
                xref="x", yref="paper",
                x0=tt, x1=tt,
                y0=1.0, y1=0.96,
                line=dict(color="rgba(0,0,0,0.35)", width=1)
            ))

    fig.update_layout(
        title=title,
        xaxis_title="时间索引",
        yaxis_title="值",
        hovermode="x unified",
        template="plotly_white",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        shapes=shapes
    )

    os.makedirs(os.path.dirname(out_path_html), exist_ok=True)
    fig.write_html(out_path_html)
    print(f"  已保存HTML: {out_path_html}")

def evaluate_parameter(gt_2d, pred_2d, mask_2d, param_name, out_dir, station_names):
    """
    对一个参数（二维数据）进行绘图
    gt_2d, pred_2d, mask_2d: 形状均为 (站点数, 时间步数)
    """
    num_sites, T = gt_2d.shape
    if station_names is None:
        station_names = [f"站点{i}" for i in range(num_sites)]
    else:
        assert len(station_names) == num_sites, "站点名称数量必须与站点数一致"

    # 创建输出子目录
    param_out_dir = os.path.join(out_dir, param_name)
    os.makedirs(param_out_dir, exist_ok=True)
    print(f"\n处理参数 [{param_name}]，输出目录: {param_out_dir}")

    # ---------- 全局散点图 ----------
    x_all, y_all = _safe_xy(gt_2d, pred_2d, mask_2d)
    m_all = _metrics(x_all, y_all)
    global_scatter_path = os.path.join(param_out_dir, "global_scatter.png")
    _scatter_xy(
        x_all, y_all,
        title=f"{param_name} 全局散点 (n={m_all['n']}, RMSE={m_all['rmse']:.4f}, MAE={m_all['mae']:.4f}, R2={m_all['r2']:.4f})",
        out_path=global_scatter_path
    )

    # ---------- 分站点散点图 ----------
    site_scatter_dir = os.path.join(param_out_dir, "per_site_scatter")
    os.makedirs(site_scatter_dir, exist_ok=True)

    for i in range(num_sites):
        name = station_names[i]
        x_i, y_i = _safe_xy(gt_2d[i], pred_2d[i], mask_2d[i])
        if x_i.size == 0:
            print(f"  站点 {name} 无有效数据，跳过散点图")
            continue
        m_i = _metrics(x_i, y_i)
        scatter_path = os.path.join(site_scatter_dir, f"{name}_scatter.png")
        _scatter_xy(
            x_i, y_i,
            title=f"{name} {param_name} 散点 (n={m_i['n']}, RMSE={m_i['rmse']:.4f}, MAE={m_i['mae']:.4f}, R2={m_i['r2']:.4f})",
            out_path=scatter_path
        )

    # ---------- 分站点时间序列 ----------
    if MAKE_PLOTLY_TS and HAS_PLOTLY:
        ts_dir = os.path.join(param_out_dir, "per_site_timeseries_html")
        os.makedirs(ts_dir, exist_ok=True)

        for i in range(num_sites):
            name = station_names[i]
            # 检查该站点是否有任何有效点（mask中有True）
            if not np.any(mask_2d[i]):
                print(f"  站点 {name} 无有效观测，跳过时间序列")
                continue
            html_path = os.path.join(ts_dir, f"{name}_timeseries.html")
            plot_timeseries_plotly(
                gt_row=gt_2d[i],
                pred_row=pred_2d[i],
                mask_row=mask_2d[i],
                title=f"{name} {param_name} 时间序列 (黑点=实测, 红点=预测)",
                out_path_html=html_path,
                show_only_known=SHOW_ONLY_KNOWN,
                val_style=VAL_STYLE,
                downsample=DOWNSAMPLE
            )
    elif MAKE_PLOTLY_TS and not HAS_PLOTLY:
        print("  提示: 已启用MAKE_PLOTLY_TS但plotly未安装，跳过HTML生成")

    print(f"[{param_name}] 处理完成，全局指标: n={m_all['n']}, RMSE={m_all['rmse']:.4f}, MAE={m_all['mae']:.4f}, R2={m_all['r2']:.4f}")
    return m_all  # 返回该参数全局指标，便于汇总

def evaluate_dataset(dataset_name, station_names):
    """对指定数据集（train/val/test）进行完整评估"""
    pred_path = os.path.join(OUTPUT_DIR, f'{dataset_name}_pred.npy')
    target_path = os.path.join(OUTPUT_DIR, f'{dataset_name}_target.npy')
    mask_path = os.path.join(OUTPUT_DIR, f'{dataset_name}_mask.npy')

    # 检查文件是否存在
    for p in [pred_path, target_path, mask_path]:
        if not os.path.exists(p):
            print(f"警告: 文件 {p} 不存在，跳过 {dataset_name} 数据集")
            return None

    # 加载数据
    pred = np.load(pred_path)      # (样本数, N, 2)
    target = np.load(target_path)
    mask = np.load(mask_path)

    print(f"\n{'='*50}")
    print(f"开始处理数据集: {dataset_name}")
    print(f"{'='*50}")
    print(f"  pred shape: {pred.shape}")
    print(f"  target shape: {target.shape}")
    print(f"  mask shape: {mask.shape}")

    # 转置为 (N, 样本数, 2) 便于按站点索引
    pred = np.transpose(pred, (1, 0, 2))   # (N, 样本数, 2)
    target = np.transpose(target, (1, 0, 2))
    mask = np.transpose(mask, (1, 0, 2))

    num_sites = pred.shape[0]
    num_timesteps = pred.shape[1]
    print(f"  转置后: 站点数={num_sites}, 时间步数={num_timesteps}")

    # 准备输出目录
    out_dir = os.path.join(OUTPUT_DIR, 'plots', dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    # 存储每个参数的全局指标
    param_metrics = {}

    # 对每个参数分别处理
    for param_idx, param_name in enumerate(PARAM_NAMES):
        # 取出该参数的所有数据 (N, 样本数)
        gt_param = target[:, :, param_idx]   # (N, 样本数)
        pred_param = pred[:, :, param_idx]   # (N, 样本数)
        mask_param = mask[:, :, param_idx]   # (N, 样本数)

        # 调用评估绘图函数
        m = evaluate_parameter(
            gt_2d=gt_param,
            pred_2d=pred_param,
            mask_2d=mask_param,
            param_name=param_name,
            out_dir=out_dir,
            station_names=station_names
        )
        param_metrics[param_name] = m

    # 输出该数据集的汇总信息
    print(f"\n{dataset_name} 数据集处理完成，各参数全局指标汇总:")
    for pname, m in param_metrics.items():
        print(f"  {pname}: n={m['n']}, RMSE={m['rmse']:.4f}, MAE={m['mae']:.4f}, R2={m['r2']:.4f}")
    return param_metrics

def main():
    # 检查输出目录是否存在
    if not os.path.exists(OUTPUT_DIR):
        raise FileNotFoundError(f"输出目录不存在: {OUTPUT_DIR}")

    # 准备站点名称
    # 需要先加载一个数据集来确定站点数，但为了方便，我们使用 test 数据集来获取站点数（如果不存在，则从其他数据集获取）
    # 先尝试加载 test 数据确定站点数，否则用 train 或 val
    num_sites = None
    for dset in ['test', 'train', 'val']:
        pred_path = os.path.join(OUTPUT_DIR, f'{dset}_pred.npy')
        if os.path.exists(pred_path):
            pred = np.load(pred_path)
            num_sites = pred.shape[1]  # (样本数, N, 2)
            break
    if num_sites is None:
        raise FileNotFoundError("未找到任何预测文件（train/val/test_pred.npy），无法确定站点数")

    if STATION_NAMES is None:
        station_names = [f"站点{i}" for i in range(num_sites)]
    else:
        station_names = STATION_NAMES
        if len(station_names) != num_sites:
            print(f"警告: 配置中站点名称数量({len(station_names)})与实际站点数({num_sites})不一致，将使用默认名称")
            station_names = [f"站点{i}" for i in range(num_sites)]

    # 依次处理三个数据集
    for dataset_name in ['train', 'val', 'test']:
        evaluate_dataset(dataset_name, station_names)

    print("\n所有数据集处理完成！请检查以下目录中的文件:")
    for dset in ['train', 'val', 'test']:
        print(f"  {os.path.join(OUTPUT_DIR, 'plots', dset)}")

if __name__ == "__main__":
    main()