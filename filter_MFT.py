import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
import re
import os

# ================= 配置区域 =================
# 1. 文件路径 (保持你原有的路径配置，使用了 r字符串 防止转义)
INPUT_FILE = r'G:\我的研\情感计算\道德\ethics\dataset\filter\commonsense\train_filter_noedited.csv'       
OUTPUT_FILE = r'G:\我的研\情感计算\道德\ethics\dataset\filter\commonsense\data_with_qwen_mft.csv' 
COLUMN_NAME = 'input'         # 需要分析的那一列的表头名

# 2. 模型本地路径 (关键修改点)
# 指向包含 config.json 和 .safetensors 文件的那个文件夹
MODEL_PATH = r"G:\我的研\情感计算\道德\ethics\models\qwen\Qwen2___5-3B-Instruct"

# ===========================================

def load_model():
    """加载本地 Qwen 模型"""
    print(f"正在加载本地模型，路径: {MODEL_PATH}")
    
    # 检查路径是否存在，防止路径写错
    if not os.path.exists(MODEL_PATH):
        print(f"错误：找不到路径 {MODEL_PATH}")
        print("请检查文件夹路径是否正确，或者文件夹名是否包含空格等特殊字符。")
        exit()

    try:
        # 加载 Tokenizer
        # 因为是本地路径，不需要 cache_dir，它直接读文件夹
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH, 
            trust_remote_code=True
        )
        
        # 加载 模型
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map="auto",          # 自动根据显存分配设备
            torch_dtype=torch.float16,  # 3B模型建议使用 float16
            trust_remote_code=True
        )
        
        print("模型加载成功！")
        return tokenizer, model
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("常见原因：1.路径错误 2.显存不足 3.缺少transformers库依赖")
        exit()

def build_prompt(text):
    """构建符合 MFQ2 定义的提示词"""
    prompt = f"""
You are a moral reasoning expert specializing in Moral Foundations Theory (MFQ2).

You will be given a short commonsense moral scenario or moral judgment text.
The text may describe a situation, an action, or a moral evaluation (e.g., from r/AmItheAsshole).

Your task is to analyze the moral values reflected in the text below.

### Moral Foundations (MFQ2):
1. Care / Harm – concern for suffering, compassion, or causing harm
2. Fairness / Cheating – justice, rights, equality, or unfair advantage
3. Loyalty / Betrayal – group loyalty, betrayal, or allegiance
4. Authority / Subversion – respect for rules, roles, hierarchy, or authority
5. Purity / Degradation – cleanliness, sanctity, taboo, or moral disgust
6. Proportionality – whether punishment, reward, or response is appropriate to the action

### Text to analyze:
"{text}"

### Instructions:
- Assign a score from 1 to 5 for each moral foundation:
  1 = Not relevant at all  
  5 = Extremely relevant or strongly reflected
- Judge based only on the content of the text.
- Do NOT judge whether the action is overall right or wrong.
- Do NOT introduce new facts or assumptions.

### Output format:
Return **only** a valid JSON object, with no explanations and no extra text:

{{
  "Care": X,
  "Fairness": X,
  "Loyalty": X,
  "Authority": X,
  "Purity": X,
  "Proportionality": X
}}
"""
    return prompt

def parse_response(response_text):
    # 简单的清洗，去除可能存在的 markdown 标记
    clean_text = re.sub(r'```json\s*', '', response_text)
    clean_text = re.sub(r'```', '', clean_text).strip()

    try:
        data = json.loads(clean_text)
        # 使用 .get 默认为0，防止模型偶尔漏掉某个键
        return {
            "Care": data.get("Care", 0),
            "Fairness": data.get("Fairness", 0),
            "Loyalty": data.get("Loyalty", 0),
            "Authority": data.get("Authority", 0),
            "Purity": data.get("Purity", 0),
            "Proportionality": data.get("Proportionality", 0),
        }
    except Exception:
        print("JSON parse error. Raw output:\n", response_text)
        # 解析失败返回全0
        return {
            "Care": 0,
            "Fairness": 0,
            "Loyalty": 0,
            "Authority": 0,
            "Purity": 0,
            "Proportionality": 0,
        }

def main():
    print(f"Reading input file: {INPUT_FILE}")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print("错误：找不到输入文件，请检查 INPUT_FILE 路径。")
        return

    # 确保那一列是字符串类型
    df[COLUMN_NAME] = df[COLUMN_NAME].astype(str)

    tokenizer, model = load_model()

    # 初始化存储结果的字典
    scores = {
        "Care": [],
        "Fairness": [],
        "Loyalty": [],
        "Authority": [],
        "Purity": [],
        "Proportionality": []
    }

    print("Start inference...")

    # 遍历每一行文本
    for text in tqdm(df[COLUMN_NAME]):
        # 文本太短直接跳过，填0
        if len(text.strip()) < 2:
            for k in scores:
                scores[k].append(0)
            continue

        # 构建对话消息
        messages = [
            {"role": "system", "content": "You are a helpful assistant that outputs only JSON."},
            {"role": "user", "content": build_prompt(text)}
        ]

        # 转换为模型输入格式
        text_input = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = tokenizer([text_input], return_tensors="pt").to(model.device)

        # 推理生成
        with torch.no_grad():
            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=256,
                temperature=0.1, # 保证结果稳定
                do_sample=False
            )

        # 截取新生成的token
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        result = parse_response(response)

        # 存入列表
        for k in scores:
            scores[k].append(result[k])

    # 将结果列写入 DataFrame
    for k in scores:
        df[f"Score_{k}"] = scores[k]

    # （可选）自动生成主导道德维度：选出分数最高的那一列作为 Label
    # idxmax 会返回列名 (例如 "Score_Care")，我们去掉 "Score_" 前缀
    df["Primary_Moral_Foundation"] = df[
        [f"Score_{k}" for k in scores]
    ].idxmax(axis=1).str.replace("Score_", "")

    # 保存
    df.to_csv(OUTPUT_FILE, index=False)
    print("Done! Saved to:", OUTPUT_FILE)

if __name__ == "__main__":
    main()