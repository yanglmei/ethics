from agent.base_agent import BaseAgent
from typing import List
from textwrap import dedent
from openai import OpenAI
import time


class MoralAgent(BaseAgent):

    def __init__(self, name: str, model: str, api_key: str, api_base: str):
        super().__init__(name, model, api_key, api_base)

    def process_texts(self, texts: List[str],excuses: List[str]):
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

        for idx, (text,excuse) in enumerate(zip(texts,excuses)):
            print(f"▶️ 正在处理第 {idx} 条文本")

            prompt = dedent(f"""
            Text:{text}
            excuses:{excuse}

            Evaluate how reasonable or unreasonable the statement or behavior described in the scenario would be perceived by most people.

            Return a real-valued score between 0 and 1, where:
            
            1 indicates that most people would consider it reasonable,
            
            0 indicates that most people would consider it unreasonable.
            
            Values between 0 and 1 represent varying degrees of perceived unreasonableness.
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
                #print(response)

                msg = response.choices[0].message

                if msg.content is None:
                    raise ValueError("Model returned empty content")

                score = response.choices[0].message.content.strip()
                print("score:",score)

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
