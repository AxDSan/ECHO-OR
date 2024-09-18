from openai import AsyncOpenAI
from src.utils.config import Config

class LLMService:
    def __init__(self):
        self.client = AsyncOpenAI(
            base_url=Config.API_URL,
            api_key=Config.OPENROUTER_API_KEY,
        )

    async def generate_text(self, prompt, max_tokens, temperature, top_p, repetition_penalty):
        try:
            completion = await self.client.chat.completions.create(
                extra_headers={"X-Title": Config.YOUR_APP_NAME},
                model=Config.LLM_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=repetition_penalty,
                n=1,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating completion: {e}")
            return ""
