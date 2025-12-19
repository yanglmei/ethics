#将commonsense的两个视角下的结果合并到一个文件中
import json
import pandas as pd

# =========================
# 文件路径
# =========================
first_scores_path = r"G:\我的研\情感计算\道德\ethics\results\commonsense\train_first_perspective_50_scores.jsonl"
third_scores_path = r"G:\我的研\情感计算\道德\ethics\results\commonsense\train_third_perspective_50_scores.jsonl"
csv_path = r"G:\我的研\情感计算\道德\ethics\dataset\filter\commonsense\train_filter_50_third_person.csv"
output_jsonl_path = r"G:\我的研\情感计算\道德\ethics\results\commonsense\result_50_gemini-pro.jsonl"

# =========================
# 1. 读取 jsonl 文件
# =========================
def read_jsonl(file_path):
    data = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            data[item["index"]] = float(item["score"])
    return data

first_scores = read_jsonl(first_scores_path)
third_scores = read_jsonl(third_scores_path)

# =========================
# 2. 读取 CSV，仅取前 50 条 label
# =========================
df = pd.read_csv(csv_path)
df_50 = df.iloc[:50]          # 👈 明确只用前 50 条
labels = df_50["label"].tolist()

# =========================
# 3. 合并数据（按 index 对齐）
# =========================
combined = []
for idx in range(len(labels)):
    combined.append({
        "index": idx,
        "first_perspective_score": first_scores.get(idx),
        "third_perspective_score": third_scores.get(idx),
        "label": labels[idx]
    })

# =========================
# 4. 保存为 jsonl
# =========================
with open(output_jsonl_path, "w", encoding="utf-8") as f:
    for record in combined:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print("✅ 合并完成，仅使用前 50 条数据")
print("📄 输出文件：", output_jsonl_path)
