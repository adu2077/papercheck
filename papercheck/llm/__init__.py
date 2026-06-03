"""可插拔 LLM 接入层（厂商中立）：配 PAPERCHECK_LLM_CMD 用任意 LLM，或自动探测已装 CLI。"""
from papercheck.llm.provider import (
    LLMProvider, CLIProvider, CommandProvider, CallableProvider, NullProvider, get_provider,
)

__all__ = [
    "LLMProvider", "CLIProvider", "CommandProvider", "CallableProvider",
    "NullProvider", "get_provider",
]
