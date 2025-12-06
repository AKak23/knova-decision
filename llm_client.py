"""LLM client wrapper for Qwen via LangChain ChatModel."""

import os
from typing import List, Optional

from langchain_core.messages import BaseMessage

try:
    from langchain_openai import ChatOpenAI
except ImportError as exc:
    # 兼容旧版 LangChain 包路径
    try:
        from langchain.chat_models import ChatOpenAI  # type: ignore
    except Exception as inner_exc:  # pragma: no cover - import fallback
        raise ImportError(
            "ChatOpenAI is required from langchain-openai or langchain.chat_models"
        ) from inner_exc


API_KEY = "sk-XXXXX"  # 可直接填写真实 key，注意不要提交到仓库


class LLMClient:
    """Simple wrapper to centralize ChatModel creation and invocation."""

    def __init__(
        self,
        model_name: str = "qwen-plus",
        temperature: float = 0.4,
        base_url: Optional[str] = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key_env: str = "QWEN_API_KEY",
        api_key: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.api_key = api_key or API_KEY
        self._chat: Optional[ChatOpenAI] = None

    def _build_client(self) -> ChatOpenAI:
        api_key = self.api_key or os.environ.get(self.api_key_env)
        # 兼容直接把真实 key 填到 api_key_env 的情况
        if not api_key and self.api_key_env.startswith("sk-"):
            api_key = self.api_key_env
        if api_key:
            # 去除无意间粘贴的智能引号和空白
            api_key = (
                api_key.replace("“", '"')
                .replace("”", '"')
                .replace("'", "")
                .replace('"', "")
                .strip()
            )
        if not api_key:
            raise EnvironmentError(
                f"Missing API key. Please export {self.api_key_env} or pass api_key to LLMClient."
            )
        return ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            api_key=api_key,
            base_url=self.base_url,
        )

    def chat(self, messages: List[BaseMessage]) -> str:
        """Invoke the chat model and return the text content."""
        if self._chat is None:
            self._chat = self._build_client()
        response = self._chat.invoke(messages)
        return response.content
