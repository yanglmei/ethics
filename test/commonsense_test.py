#选择前五十条，将input改成第三人称描述
import os
import pandas as pd
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

# 原始数据路径
input_path = r"G:\我的研\情感计算\道德\ethics\dataset\filter\commonsense\train_filter.csv"
df = pd.read_csv(input_path)

# 只取前 100 条
df_100 = df.iloc[:100]

texts = df_100["input"].tolist()
print("示例原始文本：", texts[0])

from agent.perspective_rewrite_agent import PerspectiveRewriteAgent
agent = PerspectiveRewriteAgent(
    name="PerspectiveRewriteAgent",
    model="pro-deepseek-r1",  # 或你自己的 "pro-deepseek-v3"
    api_key="sk-brvjbprrs2ihfjv5",
    api_base="https://cloud.infini-ai.com/maas/v1"   # 或者你自己的base
)

# 调用 agent 处理文本
results = agent.process_texts(
    texts=texts,
)

# 构建 index -> rewritten_text 的映射
index_to_text = {r["index"]: r["rewritten"] for r in results}

# 只保留前 100 条数据，并替换 input 为改写后的文本
df_new = df_100.copy()
df_new["input"] = df_new.index.map(lambda i: index_to_text.get(i, df_100.loc[i, "input"]))

# 保存新的 CSV
output_csv = r"G:\我的研\情感计算\道德\ethics\dataset\filter\commonsense\train_filter_100_third_person.csv"
df_new.to_csv(output_csv, index=False, encoding="utf-8-sig")

print("✅ 第三人称改写完成，文件已保存：")
print(output_csv)
