"""用 pdfplumber 抽取 PDF 表格，并把数值单元格拍平，供统计检测使用。

表格里的数字是本福特定律检验、重复数据点检测的主要原料。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pdfplumber

# 形如 12, 12.3, 1,234.5, -0.05, 1.2e3 的数字单元格
_NUM_CELL = re.compile(r"^[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)?(?:\.\d+)?(?:[eE][-+]?\d+)?$")


@dataclass
class ExtractedTable:
    paper_id: str
    page: int
    index: int                 # 该页第几个表
    rows: list[list[str]]
    numbers: list[float] = field(default_factory=list)

    def label(self) -> str:
        return f"p.{self.page} Table#{self.index + 1}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "page": self.page,
            "index": self.index,
            "n_rows": len(self.rows),
            "n_numbers": len(self.numbers),
        }


def _to_number(cell: str | None) -> float | None:
    if cell is None:
        return None
    s = cell.strip()
    if not s or not _NUM_CELL.match(s):
        return None
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def extract_tables(pdf_path: str | Path, paper_id: str | None = None) -> list[ExtractedTable]:
    pdf_path = Path(pdf_path)
    paper_id = paper_id or pdf_path.stem
    out: list[ExtractedTable] = []

    # pdfminer 对真实世界里结构不规范/被截断的 PDF 很挑剔，整体失败也只跳过表格，
    # 不连累 fitz 那条更稳健的抽图/抽文本链路。
    try:
        pdf = pdfplumber.open(pdf_path)
    except Exception:
        return out
    try:
        for pno, page in enumerate(pdf.pages):
            try:
                tables = page.extract_tables()
            except Exception:
                continue
            for ti, raw_rows in enumerate(tables):
                rows = [[(c or "").strip() for c in row] for row in raw_rows]
                nums = [
                    v
                    for row in rows
                    for v in (_to_number(c) for c in row)
                    if v is not None
                ]
                out.append(ExtractedTable(paper_id, pno + 1, ti, rows, nums))
    finally:
        pdf.close()
    return out
