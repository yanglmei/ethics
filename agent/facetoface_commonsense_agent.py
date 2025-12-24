from agent.base_agent import BaseAgent
from typing import List
from textwrap import dedent
from openai import OpenAI
import time


class MoralAgent(BaseAgent):
    """
    将第一人称（“我 / I”）文本，最小改动地改写为第三人称（某个人名）的 Agent
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
            

            Text:
            {text}

            Evaluate the degree to which I have acted wrongly in the story.
            Return a real-valued score between 0 and 1, where:

            0 indicates that I did not act wrongly,
            1 indicates that I clearly acted wrongly,
            values between 0 and 1 represent varying degrees of wrongdoing.

            Use intermediate values whenever appropriate.
            Reply with a single number between 0 and 1.
            Do not provide any explanation.
            """).strip()

            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a careful linguistic editor."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                    timeout=60
                )
                print(response)

                msg = response.choices[0].message

                if msg.content is None:
                    raise ValueError("Model returned empty content")

                score = response.choices[0].message.content.strip()
                print("rewritten:",score)

                results.append({
                    "index": idx,
                    "score": score
                })

            except Exception as e:
                print("⚠️ 请求失败，错误信息如下：")
                print(repr(e))          # 👈 这是最重要的
                results.append({
                    "index": idx,
                    "score": None,
                    "error": repr(e)
                })

            # 🔒 防止并发 / 限流（非常重要）
            time.sleep(1)

        return results