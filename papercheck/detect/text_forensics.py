"""文本/元数据取证：论文工厂与 AI 代写的文本特征。

- ChatGPT 指纹：正文里残留的 AI 助手话术（"As an AI language model…"），
  说明整段未经编辑直接粘贴自 LLM。
- Tortured phrases：用洗稿/同义替换工具躲查重生造的怪词
  （"bosom peril"=breast cancer），论文工厂典型痕迹。

纯正则/词典匹配，离线即可。措辞守"疑似/待核查"。
（撤稿引用检查需联网查撤稿库，留作后续。）
"""
from __future__ import annotations

import re

from papercheck.findings import Finding, Category


# AI 助手残留话术（小写匹配）
_CHATGPT_PATTERNS = [
    "as an ai language model", "as a language model", "as an ai,",
    "i cannot provide", "i'm sorry, but i cannot", "i am sorry, but i cannot",
    "as of my last knowledge update", "my last knowledge update",
    "i don't have access to real-time", "i do not have access to real-time",
    "regenerate response", "knowledge cutoff",
    "certainly! here is", "certainly, here is", "i hope this helps",
    "as an artificial intelligence",
]

# 已知 tortured phrase → 正常术语（来自文献）
_TORTURED = {
    "bosom peril": "breast cancer",
    "bosom malignancy": "breast cancer",
    "kidney disappointment": "kidney failure",
    "lactose bigotry": "lactose intolerance",
    "fake neural organization": "artificial neural network",
    "fake neural organizations": "artificial neural networks",
    "counterfeit consciousness": "artificial intelligence",
    "counterfeit neural organization": "artificial neural network",
    "irregular timberland": "random forest",
    "arbitrary woodland": "random forest",
    "credulous bayes": "naive bayes",
    "gullible bayes": "naive bayes",
    "haze figuring": "cloud computing",
    "enormous information": "big data",
    "gigantic information": "big data",
    "leftover organization": "residual network",
    "signal to commotion": "signal to noise",
    "mean square mistake": "mean square error",
    "bolster vector machine": "support vector machine",
}


def chatgpt_fingerprint_check(pages) -> list[Finding]:
    findings = []
    for pg in pages:
        low = pg.text.lower()
        hits = [p for p in _CHATGPT_PATTERNS if p in low]
        if hits:
            findings.append(Finding(
                detector="chatgpt_fingerprint", category=Category.TEXT, severity=0.82,
                title="正文残留 AI 助手话术（疑似未编辑的 LLM 生成）",
                description=(
                    f"第 {pg.page} 页正文出现 AI 助手固定话术：{hits}。"
                    "正式论文里不应出现这类措辞，强烈提示该段直接粘贴自 LLM 输出，待核查。"
                ),
                evidence={"page": pg.page, "patterns": hits},
                locations=[f"p.{pg.page}"],
            ))
    return findings


def tortured_phrase_check(pages, min_hits: int = 1) -> list[Finding]:
    findings = []
    all_hits = {}
    for pg in pages:
        low = re.sub(r"\s+", " ", pg.text.lower())   # 归一空白：PDF 换行会把短语断开导致漏检
        for bad, good in _TORTURED.items():
            if re.search(r"\b" + re.escape(bad) + r"\b", low):
                all_hits.setdefault(bad, (good, pg.page))
    if len(all_hits) >= min_hits:
        items = [f"「{b}」(应为 {g})" for b, (g, _) in all_hits.items()]
        sev = min(0.85, 0.45 + 0.12 * len(all_hits))
        findings.append(Finding(
            detector="tortured_phrases", category=Category.TEXT, severity=sev,
            title=f"检出 {len(all_hits)} 个 tortured phrase（疑似洗稿/论文工厂）",
            description=(
                "正文出现同义替换工具生造的怪词：" + "；".join(items) +
                "。这类词是为躲避查重把正常术语机器改写所致，是论文工厂的典型痕迹，待核查。"
            ),
            evidence={"phrases": {b: g for b, (g, _) in all_hits.items()}},
            locations=[f"p.{p}" for _, (_, p) in all_hits.items()],
        ))
    return findings


def run_text_checks(pages) -> list[Finding]:
    findings = []
    findings += chatgpt_fingerprint_check(pages)
    findings += tortured_phrase_check(pages)
    findings.sort(key=lambda f: f.severity, reverse=True)
    return findings
