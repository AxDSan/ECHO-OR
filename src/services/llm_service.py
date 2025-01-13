import aiohttp
import asyncio
import logging
import json
from typing import Optional
from src.utils.config import Config

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {Config.API_KEY}",
            "HTTP-Referer": "https://github.com/0x90/echo-or",
            "X-Title": Config.YOUR_APP_NAME,
            "Content-Type": "application/json"
        }

    async def generate_text(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None, 
        temperature: float = 0.7, 
        top_p: float = 0.9, 
        repetition_penalty: float = 1.0, 
        timeout: int = 30
    ) -> str:
        try:
            payload = {
                "model": "meta-llama/llama-3.3-70b-instruct",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant that provides clear, step-by-step reasoning."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens if max_tokens else 1000
            }

            logger.debug(f"Sending request to OpenRouter with payload: {json.dumps(payload, indent=2)}")

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=timeout
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"API returned status {response.status}: {error_text}")
                        return ""

                    response_json = await response.json()
                    logger.debug(f"Received response: {json.dumps(response_json, indent=2)}")

                    if "error" in response_json:
                        logger.error(f"Error from LLM: {response_json['error'].get('message')}")
                        return ""

                    if not response_json.get("choices"):
                        logger.error("No choices in response")
                        logger.error(f"Full response: {json.dumps(response_json, indent=2)}")
                        return ""

                    content = response_json["choices"][0].get("message", {}).get("content", "")
                    if not content:
                        logger.error("No content in response")
                        return ""

                    return content.strip()

        except asyncio.TimeoutError:
            logger.error(f"Request timed out after {timeout} seconds")
            return ""
        except Exception as e:
            logger.error(f"Error generating completion: {str(e)}")
            logger.exception("Full traceback:")
            return ""
