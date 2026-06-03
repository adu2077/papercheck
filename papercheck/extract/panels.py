"""把多面板科研图切成子面板，并区分"印迹/照片"与"图表/线条画"。

真实论文的图常是多面板拼图（a/b/c…），且印迹与柱状图/折线图混排在同一张
嵌入图里。直接对整图做图像取证会被图表的重复结构（坐标轴、刻度、相同的柱子、
字体标签）打出大量误报。

做法：
1. 递归 X-Y cut：按"贯穿的白边沟壑"把图切成子面板（印迹内部是灰背景、不是纯白，
   不会被误切；柱状图柱子间是纯白，会被切开——但图表面板本就会被跳过，无所谓）。
2. 逐面板分类：图表灰阶数很少（纯色块+大白底），印迹/照片灰阶丰富、连续色调。
3. 只把"印迹/照片"面板交给图像查重/copy-move；图表面板跳过。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Panel:
    """一个子面板，接口与 ExtractedImage 对齐，可直接喂给图像检测器。"""
    paper_id: str
    source_xref: int      # 所属嵌入图的 xref
    panel_id: int         # 在该图内的序号
    path: str
    width: int
    height: int
    sha1: str
    page: int
    kind: str             # "photo"（印迹/照片，查）| "chart"（图表，跳过）
    bbox: tuple           # 在父图中的 (x, y, w, h)
    pages: list = field(default_factory=list)

    @property
    def xref(self) -> int:
        # 给检测器当唯一 id（可视化文件名用），避免跨面板冲突
        return self.source_xref * 1000 + self.panel_id

    def label(self) -> str:
        return f"p.{self.page} img#{self.source_xref}-P{self.panel_id}"


# ---------------- 分类 ----------------

def classify_panel(gray: np.ndarray) -> str:
    """图表/线条画 vs 印迹/照片。

    判据（满足任一即图表）：
    - 灰阶数少（纯色块；实测纯图表 3~10，印迹 19~43）
    - 几乎全白
    - 墨迹覆盖率低：坐标轴/刻度/文字/折线是稀疏墨迹，印迹/照片是稠密连续色调。
      这条专门挡掉被切碎的坐标轴/标签碎片（它们灰阶可能不少，但墨迹很稀）。
    """
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel() / max(1, gray.size)
    n_levels = int((hist > 0.004).sum())
    white_frac = float((gray > 240).mean())
    ink_frac = float((gray < 200).mean())
    if n_levels < 14 or white_frac > 0.9 or ink_frac < 0.12:
        return "chart"
    return "photo"


# ---------------- 递归 X-Y cut 分割 ----------------

def _interior_gaps(profile: np.ndarray, white_thresh: float, min_gap: int):
    """profile: 每行/列的前景占比。返回内部足够宽的"白沟"中点列表。"""
    white = profile < white_thresh
    n = len(white)
    cuts, i = [], 0
    while i < n:
        if white[i]:
            j = i
            while j < n and white[j]:
                j += 1
            if (j - i) >= min_gap and i > 0 and j < n:   # 内部白沟（非边缘）
                cuts.append((i + j) // 2)
            i = j
        else:
            i += 1
    return cuts


def _tight_bbox(fg: np.ndarray):
    """去掉四周白边，返回紧致 bbox (x,y,w,h)；全白返回 None。"""
    ys, xs = np.where(fg)
    if len(xs) == 0:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    return int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)


def _segment(fg: np.ndarray, ox: int, oy: int, depth: int,
             min_gap: int, min_side: int, white_thresh: float, out: list):
    """在前景掩码 fg 上递归切分，把叶子面板的绝对 bbox 收集到 out。"""
    h, w = fg.shape
    if h < min_side or w < min_side:
        if h >= min_side // 2 and w >= min_side // 2:
            out.append((ox, oy, w, h))
        return

    rcuts = _interior_gaps(fg.mean(axis=1), white_thresh, min_gap) if depth < 6 else []
    ccuts = _interior_gaps(fg.mean(axis=0), white_thresh, min_gap) if depth < 6 else []

    # 先切横沟（分上下带），没有再切竖沟（分左右），都没有就是叶子
    if rcuts:
        bounds = [0] + rcuts + [h]
        for a, b in zip(bounds, bounds[1:]):
            if b - a >= min_side // 2:
                _segment(fg[a:b, :], ox, oy + a, depth + 1, min_gap, min_side, white_thresh, out)
    elif ccuts:
        bounds = [0] + ccuts + [w]
        for a, b in zip(bounds, bounds[1:]):
            if b - a >= min_side // 2:
                _segment(fg[:, a:b], ox + a, oy, depth + 1, min_gap, min_side, white_thresh, out)
    else:
        tb = _tight_bbox(fg)
        if tb:
            tx, ty, tw, th = tb
            if tw >= min_side and th >= min_side:
                out.append((ox + tx, oy + ty, tw, th))


def segment_panels(image_path: str, paper_id: str, page: int, source_xref: int,
                   out_dir: str | Path, min_side: int = 80) -> list[Panel]:
    """把一张嵌入图切成若干面板并分类、存盘。切不动则整图作为单一面板。"""
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    color = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if gray is None or color is None:
        return []
    H, W = gray.shape
    fg = (gray < 245)
    min_gap = max(8, int(0.015 * max(H, W)))

    boxes: list = []
    _segment(fg, 0, 0, 0, min_gap, min_side, 0.02, boxes)
    if not boxes:
        boxes = [(0, 0, W, H)]

    pdir = Path(out_dir) / paper_id / "panels"
    pdir.mkdir(parents=True, exist_ok=True)

    panels: list[Panel] = []
    for pid, (x, y, w, h) in enumerate(boxes):
        crop = color[y:y + h, x:x + w]
        gcrop = gray[y:y + h, x:x + w]
        ok, buf = cv2.imencode(".png", crop)
        if not ok:
            continue
        sha1 = hashlib.sha1(buf.tobytes()).hexdigest()
        ppath = pdir / f"x{source_xref}_p{pid}.png"
        cv2.imwrite(str(ppath), crop)
        panels.append(Panel(
            paper_id=paper_id, source_xref=source_xref, panel_id=pid,
            path=str(ppath), width=w, height=h, sha1=sha1, page=page,
            kind=classify_panel(gcrop), bbox=(x, y, w, h), pages=[page],
        ))
    return panels


def panels_from_images(images, out_dir) -> list[Panel]:
    """对一批 ExtractedImage 逐一分割，汇总所有面板。"""
    out: list[Panel] = []
    for im in images:
        page = im.pages[0] if im.pages else 0
        out += segment_panels(im.path, im.paper_id, page, im.xref, out_dir)
    return out
