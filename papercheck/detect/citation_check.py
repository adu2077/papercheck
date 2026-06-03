"""引用撤稿核查：论文是否引用了已被撤稿的文献（论文工厂/不严谨的痕迹）。

从正文抽 DOI，查 OpenAlex 的 is_retracted 字段。OpenAlex 免费、无需 key、
已整合 Retraction Watch 数据。网络不可用时静默跳过，不连累整条流水线。
"""
from __future__ import annotations

import json
import re
import urllib.request
import urllib.parse

from papercheck.findings import Finding, Category

# DOI 模式（保守，去掉尾部标点）
_DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", re.IGNORECASE)
_MAILTO = "papercheck@example.org"


def extract_dois(pages, max_dois: int = 80) -> list[str]:
    seen = []
    for pg in pages:
        for m in _DOI_RE.findall(pg.text):
            doi = m.rstrip(".,;)").lower()
            if doi not in seen:
                seen.append(doi)
            if len(seen) >= max_dois:
                return seen
    return seen


def openalex_retracted(doi: str, timeout: int = 10) -> bool | None:
    """查 OpenAlex 该 DOI 是否撤稿。返回 True/False；网络/解析失败返回 None。"""
    url = ("https://api.openalex.org/works/https://doi.org/"
           + urllib.parse.quote(doi) + "?mailto=" + _MAILTO
           + "&select=id,is_retracted")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": f"papercheck (mailto:{_MAILTO})"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        return bool(data.get("is_retracted", False))
    except Exception:
        return None


def check_retracted_citations(pages, status_fn=None, max_dois: int = 80,
                              log=lambda *a: None) -> list[Finding]:
    """抽 DOI 并逐个查撤稿状态。status_fn 可注入（便于测试）；默认走 OpenAlex。"""
    status_fn = status_fn or openalex_retracted
    dois = extract_dois(pages, max_dois)
    if not dois:
        return []
    retracted, checked = [], 0
    for doi in dois:
        st = status_fn(doi)
        if st is None:
            continue  # 查不到，跳过
        checked += 1
        if st:
            retracted.append(doi)
    log(f"  引用核查：抽到 {len(dois)} 个 DOI，成功核查 {checked} 个，撤稿 {len(retracted)} 个")
    if not retracted:
        return []
    return [Finding(
        detector="retracted_citation", category=Category.TEXT,
        severity=min(0.8, 0.5 + 0.1 * len(retracted)),
        title=f"引用了 {len(retracted)} 篇已撤稿文献",
        description=(
            "正文/参考文献中引用了以下已被撤稿的论文：" + "、".join(retracted[:10])
            + "。引用撤稿文献提示文献把关不严，也是论文工厂的常见特征，待核查相关论断是否受影响。"
        ),
        evidence={"retracted_dois": retracted, "n_dois_checked": checked},
        locations=[],
    )]
