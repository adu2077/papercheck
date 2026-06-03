"""LLM 判官：对检测器产出的候选做语义裁决。

程序检测器擅长"廉价、精确、可规模化"地找候选，但分不清"造假重复"和
"良性相似"（共用坐标轴、相似 loading control）。这一层把证据图交给 LLM，
让它给出 真可疑/良性/不确定 + 理由 + 置信度，据此调整严重度——把噪声候选
裁成可信短名单。

LLM 后端可插拔（见 papercheck.llm.provider）。措辞仍守"疑似/待核查"。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from papercheck.findings import Finding, Category
from papercheck.llm.provider import LLMProvider, get_provider


_VERDICTS = {"suspicious", "benign", "uncertain"}

# 不同类别问 LLM 的侧重点
_CATEGORY_QUESTION = {
    Category.IMAGE_DUP: (
        "这两张图/面板是否构成**有问题的图像复用**（同一图像被当作不同实验结果，"
        "或经裁剪/旋转/翻转后重复使用）？还是仅仅是**良性相似**（如共用的坐标轴、"
        "排版、不同数据但外观相近的对照）？"
    ),
    Category.IMAGE_MANIP: (
        "这张图内是否存在**可疑的复制粘贴篡改**（克隆细胞/条带/区域）？还是自然的"
        "重复纹理或正常结构？"
    ),
    Category.CROSS_PAPER: (
        "这两张分属不同论文的图是否**疑似同源盗用**？还是只是同类实验常见的相似？"
    ),
    Category.CHART: (
        "这张图表的条形/误差棒与其代表的数值是否存在**误导性不一致**？"
    ),
    Category.STATS: (
        "结合证据，这条统计异常更像是**真实的数据问题/编造**，还是可由舍入、"
        "正常变异或写法差异解释的**良性**情况？"
    ),
}

_IMG_KEYS = ["image_a", "image_b", "image", "match_viz", "copymove_viz"]


def _gather_images(finding: Finding) -> list[str]:
    out = []
    for k in _IMG_KEYS:
        p = finding.evidence.get(k)
        if p and p not in out and Path(p).exists():
            out.append(p)
    return out


def _build_prompt(finding: Finding) -> str:
    q = _CATEGORY_QUESTION.get(finding.category, "这条线索是否构成真实的学术不端疑点？")
    return (
        "你是学术图像/数据取证助手。下面是程序自动检测出的一条**疑似**线索，"
        "请你结合证据冷静判断，不要轻易下'造假'定论，但要区分真问题与良性相似。\n\n"
        f"【类别】{finding.category.value}\n"
        f"【程序给的描述】{finding.description}\n\n"
        f"【你的判断任务】{q}\n\n"
        "只输出一个 JSON 对象，不要任何多余文字：\n"
        '{"verdict": "suspicious|benign|uncertain", "confidence": 0.0-1.0, '
        '"reasoning": "一句话中文理由"}'
    )


def parse_verdict(text: str) -> dict:
    """容错解析 LLM 输出的 JSON（中文引号、代码围栏、多余文字都尽量兜住）。"""
    if not text:
        return {"verdict": "uncertain", "confidence": 0.0, "reasoning": "LLM 无输出"}
    t = text.strip()
    t = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", t).strip()  # 去代码围栏
    # 取第一个 {...} 块
    s, e = t.find("{"), t.rfind("}")
    if s != -1 and e != -1 and e > s:
        t = t[s:e + 1]
    for attempt in (t, t.replace("“", '"').replace("”", '"').replace("，", ",")):
        try:
            d = json.loads(attempt)
            v = str(d.get("verdict", "uncertain")).lower().strip()
            if v not in _VERDICTS:
                v = "uncertain"
            try:
                c = max(0.0, min(1.0, float(d.get("confidence", 0.0))))
            except (TypeError, ValueError):
                c = 0.0
            return {"verdict": v, "confidence": c,
                    "reasoning": str(d.get("reasoning", "")).strip()[:300]}
        except json.JSONDecodeError:
            continue
    # 实在解析不了，从文本里猜个倾向
    low = text.lower()
    v = "benign" if ("benign" in low or "良性" in text) else (
        "suspicious" if ("suspicious" in low or "可疑" in text) else "uncertain")
    return {"verdict": v, "confidence": 0.3, "reasoning": "JSON 解析失败，按文本推断"}


def judge_finding(finding: Finding, provider: LLMProvider) -> dict:
    prompt = _build_prompt(finding)
    images = _gather_images(finding)
    try:
        raw = provider.complete(prompt, images)
    except Exception as e:  # LLM 不可用不应让整条流水线崩
        return {"verdict": "uncertain", "confidence": 0.0, "reasoning": f"LLM 调用失败：{e}"}
    return parse_verdict(raw)


def apply_verdict(finding: Finding, verdict: dict) -> Finding:
    """把裁决写入证据，并据此调整严重度。

    - benign 且置信高 → 大幅降权（基本判为良性）
    - suspicious 且置信高 → 适度加权
    - uncertain / 低置信 → 基本不动
    """
    finding.evidence["llm_verdict"] = verdict
    v, c = verdict["verdict"], verdict["confidence"]
    if v == "benign":
        finding.severity = round(finding.severity * (1.0 - 0.7 * c), 4)
    elif v == "suspicious":
        finding.severity = round(min(0.99, finding.severity + 0.15 * c), 4)
    if verdict.get("reasoning"):
        finding.description += f"\n\n🧠 LLM 判官（{v}, 置信 {c:.0%}）：{verdict['reasoning']}"
    return finding


_HOLISTIC_PROMPT = (
    "你是学术图像取证专家。请整体审视这张科研图（可能是西部印迹、显微图、FACS、"
    "凝胶等），找出任何**人为操纵迹象**：拼接缝、复制粘贴/克隆的区域、背景灰度断层、"
    "不自然的擦除痕迹、异常一致或重复的条带、与正常实验不符的特征。\n"
    "不要凭空臆断；没有明显问题就如实说没有。只输出 JSON：\n"
    '{"has_concern": true/false, "severity": 0.0-1.0, "issues": ["简短问题点"], '
    '"reasoning": "一句话中文说明"}'
)


def parse_holistic(text: str) -> dict:
    """解析整图通览的 JSON（容错）。"""
    base = {"has_concern": False, "severity": 0.0, "issues": [], "reasoning": ""}
    if not text:
        return base
    t = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text.strip()).strip()
    s, e = t.find("{"), t.rfind("}")
    if s != -1 and e != -1 and e > s:
        t = t[s:e + 1]
    for attempt in (t, t.replace("“", '"').replace("”", '"').replace("，", ",")):
        try:
            d = json.loads(attempt)
            return {
                "has_concern": bool(d.get("has_concern", False)),
                "severity": max(0.0, min(1.0, float(d.get("severity", 0.0) or 0.0))),
                "issues": [str(x)[:120] for x in (d.get("issues") or [])][:6],
                "reasoning": str(d.get("reasoning", "")).strip()[:300],
            }
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return base


def holistic_review(images, provider: LLMProvider | None = None,
                    log=lambda *a: None) -> list[Finding]:
    """把每张图整张交给 LLM 做开放式取证，找检测器没编进去的破绽。

    每张图一次 LLM 调用，较贵，默认在流水线里关闭。
    """
    provider = provider or get_provider()
    findings: list[Finding] = []
    for im in images:
        try:
            raw = provider.complete(_HOLISTIC_PROMPT, [im.path])
        except Exception as e:
            log(f"  整图通览失败 {im.label()}: {e}")
            continue
        d = parse_holistic(raw)
        if d["has_concern"] and d["severity"] >= 0.3:
            findings.append(Finding(
                detector="llm_holistic", category=Category.IMAGE_MANIP,
                severity=min(0.9, d["severity"]),
                title="LLM 整图通览发现可疑迹象",
                description="LLM 开放式审查：" + d["reasoning"]
                            + ("；问题点：" + "、".join(d["issues"]) if d["issues"] else ""),
                evidence={"image": im.path, "issues": d["issues"], "reasoning": d["reasoning"]},
                locations=[im.label()],
            ))
        log(f"  通览 {im.label()}: concern={d['has_concern']} sev={d['severity']:.2f}")
    return findings


def judge_findings(findings: list[Finding], provider: LLMProvider | None = None,
                   only_categories: set | None = None, log=lambda *a: None) -> list[Finding]:
    """逐条裁决并调整严重度，返回按新严重度排序的列表。"""
    provider = provider or get_provider()
    for i, f in enumerate(findings):
        if only_categories and f.category not in only_categories:
            continue
        verdict = judge_finding(f, provider)
        apply_verdict(f, verdict)
        log(f"  判官[{i+1}/{len(findings)}] {f.category.value}: "
            f"{verdict['verdict']}({verdict['confidence']:.0%}) → sev={f.severity:.2f}")
    findings.sort(key=lambda f: f.severity, reverse=True)
    return findings
