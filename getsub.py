#得到commonsense中长的文本
import os
import pandas as pd

# 原始数据路径
input_path = r"G:\我的研\情感计算\道德\ethics\dataset\commonsense\train.csv"

# 输出目录和文件名
output_dir = r"G:\我的研\情感计算\道德\ethics\dataset\filter\commonsense"
output_path = os.path.join(output_dir, "train_filter.csv")

# 如果输出目录不存在，就创建
os.makedirs(output_dir, exist_ok=True)

# 读取 CSV
df = pd.read_csv(input_path)

# 筛选 is_short 为 FALSE 的数据
filtered_df = df[df["is_short"] == False]

# 保存结果
filtered_df.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"筛选完成，共保存 {len(filtered_df)} 条数据到：")
print(output_path)
