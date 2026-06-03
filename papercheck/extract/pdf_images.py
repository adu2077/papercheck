"""从 PDF 抽取所有栅格图片（含页内位置），供图片查重/篡改检测使用。

用 PyMuPDF：
- page.get_images(full=True) 列出每页引用的图片 xref
- doc.extract_image(xref) 拿原始字节
- page.get_image_rects(xref) 拿在页面上的位置（bbox）

同一 xref 可能在多页被引用（如重复的 logo），按 (xref) 去重保存像素，
但每个出现位置都记一条 placement，方便"同一篇里这张图出现在哪几个图位"。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF


@dataclass
class ExtractedImage:
    paper_id: str
    xref: int
    path: str                 # 保存到磁盘的 PNG 路径
    width: int
    height: int
    sha1: str                 # 原始字节哈希（用于识别像素级完全相同）
    pages: list[int] = field(default_factory=list)        # 出现在哪些页（1-based）
    bboxes: list[tuple] = field(default_factory=list)     # 每次出现的页面 bbox
    colorspace: str = ""

    @property
    def n_pixels(self) -> int:
        return self.width * self.height

    def label(self) -> str:
        pg = self.pages[0] if self.pages else "?"
        return f"p.{pg} img#{self.xref}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "xref": self.xref,
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "sha1": self.sha1,
            "pages": self.pages,
            "colorspace": self.colorspace,
        }


def extract_images(
    pdf_path: str | Path,
    out_dir: str | Path,
    paper_id: str | None = None,
    min_side: int = 64,
    min_pixels: int = 100 * 100,
) -> list[ExtractedImage]:
    """抽取 PDF 中所有"够大"的栅格图片。

    min_side / min_pixels：过滤掉 logo、图标、分隔线这类太小的图，
    它们既不是造假对象，又会在查重里制造海量噪声配对。
    """
    pdf_path = Path(pdf_path)
    paper_id = paper_id or pdf_path.stem
    img_dir = Path(out_dir) / paper_id / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    by_xref: dict[int, ExtractedImage] = {}

    try:
        for pno in range(doc.page_count):
            page = doc[pno]
            for img in page.get_images(full=True):
                xref = img[0]
                if xref in by_xref:
                    # 已抽过这张图，只补一次出现位置
                    rec = by_xref[xref]
                    rec.pages.append(pno + 1)
                    rec.bboxes.extend(_rects(page, xref))
                    continue

                try:
                    info = doc.extract_image(xref)
                except Exception:
                    continue
                raw = info.get("image")
                if not raw:
                    continue
                w, h = int(info.get("width", 0)), int(info.get("height", 0))
                if min(w, h) < min_side or (w * h) < min_pixels:
                    continue

                sha1 = hashlib.sha1(raw).hexdigest()
                # 统一存成 PNG，避免下游为各种格式分支
                out_path = img_dir / f"x{xref}.png"
                _save_png(doc, xref, info, out_path)

                rec = ExtractedImage(
                    paper_id=paper_id,
                    xref=xref,
                    path=str(out_path),
                    width=w,
                    height=h,
                    sha1=sha1,
                    pages=[pno + 1],
                    bboxes=_rects(page, xref),
                    colorspace=str(info.get("colorspace", "")),
                )
                by_xref[xref] = rec
    finally:
        doc.close()

    return list(by_xref.values())


def _rects(page: "fitz.Page", xref: int) -> list[tuple]:
    try:
        return [tuple(round(v, 1) for v in r) for r in page.get_image_rects(xref)]
    except Exception:
        return []


def _save_png(doc: "fitz.Document", xref: int, info: dict, out_path: Path) -> None:
    """把图片以 PNG 写盘。带 alpha/CMYK 的统一转 RGB(A)。"""
    pix = fitz.Pixmap(doc, xref)
    try:
        if pix.colorspace and pix.colorspace.name not in ("DeviceRGB", "DeviceGray"):
            pix = fitz.Pixmap(fitz.csRGB, pix)
        pix.save(out_path)
    finally:
        pix = None
