import pandas as pd
import yaml
import os

# ==================== 加载配置 ====================
config_file = "/home/fanyunkai/FYK_GCNLSTM/configs.yaml"  # 根据实际路径调整
if not os.path.exists(config_file):
    raise FileNotFoundError(f"配置文件 {config_file} 不存在")

with open(config_file, "r", encoding="utf-8") as f:
    config_raw = yaml.safe_load(f)

# 提取嵌套的配置（假设顶层键为 QX_subwatershed_sites）
top_key = "QX_subwatershed_sites"
if top_key not in config_raw:
    raise KeyError(f"配置文件中缺少顶层键 '{top_key}'，请检查 YAML 结构")

config = config_raw[top_key]

# 从配置中获取参数
input_csv = config.get("input_csv")
output_xlsx = config.get("output_xlsx")
csv_encoding = config.get("csv_encoding", "utf-8")
sheet_mapping = config.get("sheet_mapping", {})
order = config.get("order", [])

# 确保 sheet_mapping 的键为整数
sheet_mapping = {int(k): v for k, v in sheet_mapping.items()}
# ==================================================

# 检查必要参数
if not input_csv:
    raise ValueError("配置中缺少 input_csv 路径")
if not output_xlsx:
    raise ValueError("配置中缺少 output_xlsx 路径")

# 读取CSV文件
df = pd.read_csv(input_csv, encoding=csv_encoding)

# 检查必要的列是否存在
required_cols = ['GRIDCODE', '经度', '纬度']
if not all(col in df.columns for col in required_cols):
    raise ValueError(f"CSV文件必须包含列：{required_cols}")

# 将GRIDCODE转换为整数，确保与映射匹配
df['GRIDCODE'] = df['GRIDCODE'].astype(int)

# 创建Excel写入器
with pd.ExcelWriter(output_xlsx, engine='openpyxl') as writer:
    for code in order:
        if code not in sheet_mapping:
            print(f"警告：GRIDCODE {code} 未在 sheet_mapping 中定义，跳过该工作表")
            continue

        sheet_name = sheet_mapping[code]
        subset = df[df['GRIDCODE'] == code][['经度', '纬度']]

        if subset.empty:
            print(f"提示：GRIDCODE {code} 无对应数据，将创建空工作表")
            subset = pd.DataFrame(columns=['经度', '纬度'])

        subset.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"处理完成！输出文件：{output_xlsx}")