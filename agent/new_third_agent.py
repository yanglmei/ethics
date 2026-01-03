from agent.base_agent import BaseAgent
from typing import List
from textwrap import dedent
from openai import OpenAI
import time


class MoralAgent(BaseAgent):
    """
    将第一人称（“我 / I”）文本，最小改动地改写为第三人称（某个人名）
    """

    def __init__(self, name: str, model: str, api_key: str, api_base: str):
        super().__init__(name, model, api_key, api_base)

    def process_texts(self, texts: List[str]):
        """
        输入:
            texts: List[str]

        输出:
            List[dict]，每条包含:
            {
                "index": 原文本索引,
                "rewritten": 改写后的文本
            }
        """
        client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        results = []

        for idx, text in enumerate(texts):
            print(f"▶️ 正在处理第 {idx} 条文本")

            # -------- Prompt（不要求 JSON）--------
            prompt = dedent(f"""
            
            {text}
            That' all.
            Now,
            
            0 indicates Nick's behavior was correct.
            1 indicates Nick's behavior was clearly wrong.
            Values between 0 and 1 represent different degrees of wrongdoing.
            Please use a middle value as appropriate, replying with only a single number between 0 and 1 to evaluate Nick; 
            no explanation is required.
            """).strip()

            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a moral judge."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                    timeout=500
                )

                score = response.choices[0].message.content.strip()
                print("rewritten:",score)

                results.append({
                    "index": idx,
                    "score": score
                })

            except Exception as e:
                print(f"❌ 请求失败，具体原因: {e}") 
                results.append({
                    "index": idx,
                    "score": text,
                    "error": str(e)
                })

            # 🔒 防止并发 / 限流（非常重要）
            time.sleep(1)

        return results