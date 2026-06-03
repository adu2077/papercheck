"""抽取 PDF 正文文本，并用正则提取"论文里报告的统计检验结果"。

statcheck 思路：论文正文里写的 t(df)=…, p=… 是一个**可被重算**的三元组。
把它们抠出来，下游 stats_anomaly 用 df+统计量重算 p，和作者写的 p 比对，
不一致就说明"报告的统计结果内部矛盾"（统计造假/笔误的强信号）。

⚠️ 正则必须用真实输出验证（见全局规范）。这里的模式覆盖 statcheck R 包的
主流形态，单测在 tests/ 里用已知字符串钉死。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fitz


@dataclass
class PageText:
    page: int
    text: str


@dataclass
class ReportedStat:
    """一条从正文里抠出来的可重算统计结果。"""
    kind: str                 # "t" | "F" | "r" | "z" | "chi2"
    df1: float | None
    df2: float | None         # 仅 F 检验有第二个自由度
    statistic: float
    p_comparator: str         # "=" | "<" | ">"
    p_reported: float
    page: int
    raw: str

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


# 一个数字（含可选符号、小数、前导小数点 .05）
_NUM = r"[-+]?\d*\.?\d+"

# t(df) = stat, p (=|<|>) pval
_T_RE = re.compile(
    rf"\bt\s*\(\s*({_NUM})\s*\)\s*=\s*({_NUM})\s*,\s*p\s*([=<>])\s*({_NUM})",
    re.IGNORECASE,
)
# F(df1, df2) = stat, p ...
_F_RE = re.compile(
    rf"\bF\s*\(\s*({_NUM})\s*,\s*({_NUM})\s*\)\s*=\s*({_NUM})\s*,\s*p\s*([=<>])\s*({_NUM})",
    re.IGNORECASE,
)
# r(df) = stat, p ...
_R_RE = re.compile(
    rf"\br\s*\(\s*({_NUM})\s*\)\s*=\s*({_NUM})\s*,\s*p\s*([=<>])\s*({_NUM})",
    re.IGNORECASE,
)
# z = stat, p ...
_Z_RE = re.compile(
    rf"\bz\s*=\s*({_NUM})\s*,\s*p\s*([=<>])\s*({_NUM})",
    re.IGNORECASE,
)
# χ2(df) = stat, p ...  /  chi2(df)...  /  X2(df)...
_CHI_RE = re.compile(
    rf"(?:χ2|χ²|chi2|chi-square|X2)\s*\(\s*({_NUM})\s*(?:,\s*N\s*=\s*{_NUM}\s*)?\)\s*=\s*({_NUM})\s*,\s*p\s*([=<>])\s*({_NUM})",
    re.IGNORECASE,
)

# mean ± sd / M = x, SD = y  —— 供 GRIM 等用（n 往往不在同一句，下游尽力配对）
# \bM / \bSD 词边界：避免 "BM = 3"、"RMSD = .." 之类被误当 M=/SD=
MEAN_SD_RE = re.compile(
    rf"(?:\bM\s*=\s*({_NUM})\s*,?\s*\bSD\s*=\s*({_NUM})|({_NUM})\s*[±]\s*({_NUM}))",
    re.IGNORECASE,
)

# 样本量 n=.. / N=..
_N_RE = re.compile(r"\b[nN]\s*=\s*(\d{1,7})\b")
# 整数/计数/量表语境关键词（GRIM/GRIMMER 仅对这类数据成立）
_INT_CTX = re.compile(
    r"\b(scale|score[ds]?|likert|point[s]?|item[s]?|questionnaire|count[s]?|rating[s]?|"
    r"survey|number of|times|频次|计分|量表|问卷|评分)\b", re.IGNORECASE)
# 量表上下界："scale of 1 to 7" / "ranged from 0 to 100" / "1–7"
_SCALE_RE = re.compile(
    r"(?:scale|rang\w*)\s+(?:of\s+|from\s+)?(\d+)\s*(?:to|through|[-–—])\s*(\d+)", re.IGNORECASE)


@dataclass
class MeanSdN:
    mean: float
    sd: float | None
    n: int
    page: int
    context: str
    scale: tuple | None = None       # (vmin, vmax) 若抽到量表界
    mean_decimals: int | None = None  # 原始字符串小数位（GRIM 必须用它，不能从 float 反推）
    sd_decimals: int | None = None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _str_decimals(s: str) -> int:
    """从数字**字符串**取小数位（"4.00"→2, "4"→0）——比从 float repr 反推可靠。"""
    s = s.strip()
    return len(s.split(".")[1]) if "." in s else 0


def _msd(m) -> tuple[float, float, int, int] | None:
    """从 MEAN_SD_RE 的 match 取 (mean, sd, mean小数位, sd小数位)。"""
    if m.group(1) and m.group(2):
        a, b = m.group(1), m.group(2)
    elif m.group(3) and m.group(4):
        a, b = m.group(3), m.group(4)
    else:
        return None
    return float(a), float(b), _str_decimals(a), _str_decimals(b)


def extract_mean_sd_n(pages) -> list[MeanSdN]:
    """从正文抽 (均值, SD, n) 三元组——**仅在整数/计数/量表语境**里抽，
    避免把连续数据(体重/年龄)误喂给 GRIM/GRIMMER 造成误报。

    要求同一句里同时有：整数语境关键词 + (M±SD 或 M=,SD=) + n=。
    """
    out: list[MeanSdN] = []
    for pg in pages:
        text = re.sub(r"\s+", " ", pg.text)
        for sent in re.split(r"(?<=[.;])\s", text):
            if not _INT_CTX.search(sent):
                continue
            nm = _N_RE.search(sent)
            if not nm:
                continue
            n = int(nm.group(1))
            if n < 2 or n > 100000:
                continue
            sc = _SCALE_RE.search(sent)
            scale = None
            if sc:
                lo, hi = int(sc.group(1)), int(sc.group(2))
                if hi > lo:
                    scale = (lo, hi)
            for m in MEAN_SD_RE.finditer(sent):
                pair = _msd(m)
                if pair:
                    out.append(MeanSdN(pair[0], pair[1], n, pg.page, sent.strip()[:80],
                                       scale, pair[2], pair[3]))
    return out


def extract_pages(pdf_path: str | Path) -> list[PageText]:
    doc = fitz.open(pdf_path)
    try:
        return [PageText(page=i + 1, text=doc[i].get_text("text")) for i in range(doc.page_count)]
    finally:
        doc.close()


def _f(x: str) -> float:
    return float(x)


def extract_reported_stats(pages: list[PageText]) -> list[ReportedStat]:
    """从每页文本里抠出所有可重算统计三元组。"""
    out: list[ReportedStat] = []
    for pg in pages:
        # 把跨行断开的空白归一，减少 PDF 换行造成的漏配
        text = re.sub(r"\s+", " ", pg.text)

        for m in _T_RE.finditer(text):
            out.append(ReportedStat("t", _f(m[1]), None, _f(m[2]), m[3], _f(m[4]), pg.page, m[0]))
        for m in _F_RE.finditer(text):
            out.append(ReportedStat("F", _f(m[1]), _f(m[2]), _f(m[3]), m[4], _f(m[5]), pg.page, m[0]))
        for m in _R_RE.finditer(text):
            out.append(ReportedStat("r", _f(m[1]), None, _f(m[2]), m[3], _f(m[4]), pg.page, m[0]))
        for m in _Z_RE.finditer(text):
            out.append(ReportedStat("z", None, None, _f(m[1]), m[2], _f(m[3]), pg.page, m[0]))
        for m in _CHI_RE.finditer(text):
            out.append(ReportedStat("chi2", _f(m[1]), None, _f(m[2]), m[3], _f(m[4]), pg.page, m[0]))
    return out
