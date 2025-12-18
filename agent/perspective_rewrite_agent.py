from agent.base_agent import BaseAgent
from typing import List
from textwrap import dedent
from openai import OpenAI
import time


class PerspectiveRewriteAgent(BaseAgent):
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
            Rewrite the following text from a first-person perspective into a third-person perspective, making minimal changes.

            Instructions:

            Replace “I / me / my / mine / myself / 我 / 我的” with a person name (e.g., use “Nick” if the speaker is male, or “Judy” if the speaker is female).

            Adjust the grammar of the entire text as needed (e.g., my → his / her).

            Preserve the original meaning, tone, and sentence structure as much as possible.

            Do NOT add or remove any information.

            Do NOT change the emotional content.

            IMPORTANT:

            Return only the rewritten text.

            Do NOT include any explanations, notes, or JSON.

            Do NOT add any extra content.

            Text:
            {text}
            """).strip()

            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a careful linguistic editor."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                    timeout=500
                )

                rewritten = response.choices[0].message.content.strip()
                print("rewritten:",rewritten)

                results.append({
                    "index": idx,
                    "rewritten": rewritten
                })

            except Exception as e:
                print("⚠️ 请求失败，保留原文本")
                results.append({
                    "index": idx,
                    "rewritten": text,
                    "error": str(e)
                })

            time.sleep(1)

        return results
