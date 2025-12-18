#测第一人称和第三人称视角下的commonsense
import os
import pandas as pd
import sys
import json

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
# 原始数据路径
#input_path = r"G:\我的研\情感计算\道德\ethics\dataset\filter\commonsense\train_filter.csv"
input_path = r"G:\我的研\情感计算\道德\ethics\dataset\filter\commonsense\train_filter_50_third_person.csv"
df = pd.read_csv(input_path)

# 只取前 50 条
df_50 = df.iloc[:50]

texts = df_50["input"].tolist()

print(texts[0])

from agent.third_perspective_agent import MoralAgent
# agent = MoralAgent(
#     name="MoralAgent",
#     model="pro-deepseek-r1",  # 或你自己的 "pro-deepseek-v3"
#     api_key="sk-brvjbprrs2ihfjv5",
#     api_base="https://cloud.infini-ai.com/maas/v1"   # 或者你自己的base
# )
agent = MoralAgent(
    name="MoralAgent",
    model="gpt-4o",  # 或你自己的 "pro-deepseek-v3"
    api_key="sk-AkWKr0v706oQniwQC8Bf507f6c154556B8F836F8764e2360",
    api_base="https://api.mixrai.com/v1/"   # 或者你自己的base
)

results = agent.process_texts(
    texts=texts,

)

index_to_text = {
    r["index"]: r["score"]
    for r in results
}

# 4. 保存为 jsonl
# ===============================
output_jsonl = r"G:\我的研\情感计算\道德\ethics\results\commonsense\train_third_perspective_50_scores.jsonl"

os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

with open(output_jsonl, "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("✅ Moral score 已保存为 jsonl 文件：")
print(output_jsonl)