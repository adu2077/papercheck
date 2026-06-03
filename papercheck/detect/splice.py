"""拼接缝检测：西部印迹/凝胶把不同来源的条带拼接到一起。

拼接造假很常见——把不同实验的泳道剪下来拼成一张"漂亮"的图。拼接处通常有
一条**贯穿整幅**的背景灰度突变（两块来源的曝光/背景不同）。

做法：算每列(每行)的中位数作为"背景轮廓"，找其中孤立而强烈的阶跃 → 候选缝。
中位数对前景条带稳健，主要反映背景，所以阶跃≈背景断层。

说明：这是轻量启发式，能抓明显拼接；细微拼接需更专业方法(如 ImageTwin)或
交给 LLM 整图通览。措辞守"疑似/待核查"。
"""
from __future__ import annotations

import cv2
import numpy as np

from papercheck.findings import Finding, Category
from papercheck.detect._imaging import _load_gray, tone_levels


def _has_long_lines(gray_u8: np.ndarray, n: int = 3) -> bool:
    """是否有 ≥n 条贯穿的笔直长线（坐标轴/网格/边框）——图表特征，印迹没有。"""
    H, W = gray_u8.shape
    edges = cv2.Canny(gray_u8, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                            minLineLength=int(0.5 * min(H, W)), maxLineGap=4)
    return lines is not None and len(lines) >= n


def find_splice_seams(image, min_step: float = 14.0, peak_ratio: float = 6.0,
                      min_levels: int = 15, min_side: int = 150) -> Finding | None:
    """检测单张图里的拼接缝。image 需有 .path / .label()。"""
    g = _load_gray(image.path)
    if g is None:                       # 坏图/读不出 → 跳过，不崩
        return None
    gray = g.astype(np.float32)
    H, W = gray.shape
    # 跳过：① 过小面板（图标/标签碎片）② 灰阶太少（纯色块）③ 有贯穿长直线（坐标轴/网格）。
    # 后两类面板的坐标轴/柱边会被误判成"背景阶跃"拼接缝。
    if min(H, W) < min_side or tone_levels(g) < min_levels or _has_long_lines(g):
        return None
    best = None  # (orientation, position, step, length)

    for axis, name, dim in [(0, "vertical", W), (1, "horizontal", H)]:
        if dim < 40:
            continue
        med = np.median(gray, axis=axis)          # axis=0→每列中位数(竖缝)；axis=1→每行(横缝)
        d = np.abs(np.diff(med))
        lo, hi = int(0.06 * len(d)), int(0.94 * len(d))
        if hi <= lo:
            continue
        idx = lo + int(np.argmax(d[lo:hi]))
        peak = float(d[idx])
        base = float(np.median(d)) + 1e-6
        if peak >= min_step and peak >= peak_ratio * base:
            if best is None or peak > best[2]:
                best = (name, idx + 1, peak, dim)

    if best is None:
        return None
    name, pos, peak, dim = best
    sev = min(0.85, 0.45 + peak / 80.0)
    where = f"第 {pos} 列" if name == "vertical" else f"第 {pos} 行"
    return Finding(
        detector="splice", category=Category.IMAGE_MANIP, severity=sev,
        title=f"疑似拼接缝（{'竖直' if name=='vertical' else '水平'}）",
        description=(
            f"{image.label()} 在{where}处检测到贯穿整幅的背景灰度突变（阶跃≈{peak:.0f} 灰阶），"
            "提示此处可能由两块不同来源的图像拼接而成。西部印迹/凝胶的泳道拼接是常见造假手法，"
            "建议人工核查该缝两侧是否同源。"
        ),
        evidence={"image": image.path, "orientation": name, "position": pos,
                  "step": round(peak, 1)},
        locations=[image.label()],
    )
