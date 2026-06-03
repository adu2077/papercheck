"""可插拔 LLM 接入——**厂商中立**，不绑定任何特定厂商。

判官/通览是可选功能。选择后端的优先级：
1. 环境变量 `PAPERCHECK_LLM_CMD`：任意命令模板（占位符 `{prompt}` `{images}`）。
   **最通用**，适配任何 LLM CLI（自建/本地/云端皆可）。
2. 自动探测：PATH 上已安装的受支持 agent CLI（见 `_PRESETS`，可自行扩展）。
3. 都没有：`NullProvider`，调用时报清晰错误，`--judge` 优雅降级（不崩）。

统一接口 `LLMProvider.complete(prompt, images) -> str`。视觉由各后端自行处理
（把图片绝对路径写进提示词，让会读图的 CLI 自取；或用 `{images}` 占位符）。
"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from typing import Callable, Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    def complete(self, prompt: str, images: list[str] | None = None) -> str:
        """给提示词(+可选图片路径)，返回 LLM 文本输出。"""
        ...


def _with_images(prompt: str, images: list[str]) -> str:
    if not images:
        return prompt
    listing = "\n".join(f"- {os.path.abspath(p)}" for p in images)
    return f"{prompt}\n\n请先查看以下证据图片（按路径读取）再作判断：\n{listing}"


class CLIProvider:
    """调用本地某个 LLM CLI。argv_builder(prompt, images) -> 完整 argv 列表（不过 shell）。"""

    def __init__(self, argv_builder: Callable[[str, list[str]], list[str]],
                 name: str = "cli", timeout: int = 120):
        self.argv_builder = argv_builder
        self.name = name
        self.timeout = timeout

    def complete(self, prompt: str, images: list[str] | None = None) -> str:
        argv = self.argv_builder(prompt, images or [])
        try:
            r = subprocess.run(argv, capture_output=True, text=True, timeout=self.timeout)
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            raise RuntimeError(f"调用 LLM CLI（{self.name}）失败：{e}") from e
        if r.returncode != 0:
            raise RuntimeError(f"LLM CLI（{self.name}）退出码 {r.returncode}：{r.stderr[:300]}")
        return r.stdout.strip()


class CommandProvider:
    """通用 CLI 后端：用户给命令模板，配自己的 LLM（任意厂商）。

    模板支持占位符 {prompt} 和 {images}（空格分隔的图片路径，已做 shell 转义）。
    例：  PAPERCHECK_LLM_CMD='llm -m gpt-4o -a {images} {prompt}'
          PAPERCHECK_LLM_CMD='ollama run llava {prompt}'
    """

    def __init__(self, template: str, timeout: int = 180):
        self.template = template
        self.timeout = timeout

    def complete(self, prompt: str, images: list[str] | None = None) -> str:
        imgs = " ".join(shlex.quote(os.path.abspath(p)) for p in (images or []))
        # 用 replace 而非 str.format：模板里可能含字面花括号（如 JSON），format 会误判为占位符
        cmd_str = (self.template
                   .replace("{prompt}", shlex.quote(prompt))
                   .replace("{images}", imgs))
        r = subprocess.run(cmd_str, shell=True, capture_output=True, text=True, timeout=self.timeout)
        if r.returncode != 0:
            raise RuntimeError(f"自定义 LLM 命令退出码 {r.returncode}：{r.stderr[:300]}")
        return r.stdout.strip()


class CallableProvider:
    """把任意 Python 可调用对象包成 provider（便于嵌入式集成与测试）。"""

    def __init__(self, fn: Callable[[str, list[str] | None], str]):
        self.fn = fn

    def complete(self, prompt: str, images: list[str] | None = None) -> str:
        return self.fn(prompt, images)


class NullProvider:
    """未配置任何 LLM 时的占位：调用即报清晰错误，由上层 try/except 优雅降级。"""

    def complete(self, prompt: str, images: list[str] | None = None) -> str:
        raise RuntimeError(
            "未配置 LLM 后端。请设环境变量 PAPERCHECK_LLM_CMD（任意 LLM CLI 命令模板），"
            "或安装一个受支持的本地 agent CLI（如 claude / codex）。"
        )


# ---- 受支持的本地 agent CLI 预设（探测顺序；用户可按需增改，比如加 openclaw）----
# 每个 builder: (prompt, images, model) -> argv 列表。这些只是"PATH 上有就用"的便捷适配，
# 不代表对任何厂商的偏好；最通用的方式始终是 PAPERCHECK_LLM_CMD。

def _claude_argv(prompt, images, model):
    argv = ["claude", "-p", _with_images(prompt, images), "--output-format", "text"]
    if model:
        argv += ["--model", model]
    if images:
        argv += ["--allowedTools", "Read"]
    return argv


def _codex_argv(prompt, images, model):
    return ["codex", "exec", "-m", model or "gpt-5.5", "--ephemeral", _with_images(prompt, images)]


_PRESETS = [("claude", _claude_argv), ("codex", _codex_argv)]


def get_provider() -> LLMProvider:
    """按优先级解析后端：环境命令模板 > 自动探测已装 CLI > Null（优雅降级）。

    PAPERCHECK_LLM_CMD：自定义命令模板（最通用）。
    PAPERCHECK_LLM_MODEL：传给被探测到的 CLI 的模型名（可选）。
    """
    cmd = os.environ.get("PAPERCHECK_LLM_CMD")
    if cmd:
        return CommandProvider(cmd)
    model = os.environ.get("PAPERCHECK_LLM_MODEL")
    for name, builder in _PRESETS:
        if shutil.which(name):
            return CLIProvider(lambda p, i, _b=builder: _b(p, i, model), name=name)
    return NullProvider()
