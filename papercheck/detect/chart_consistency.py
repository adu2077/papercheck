"""③ 图表自洽性：柱状图条高 vs 报告数值是否吻合。

抓的造假/笔误：柱子的**像素高度比例**和它代表的数值比例对不上
（例如把劣势组的柱子画高、或两根柱子高度一样却标注不同的值）。

思路：
1. 从图里检测出共享同一基线的若干"柱子"，量出各自像素高度。
2. 若提供了这些柱子应代表的报告数值，做过原点的最小二乘标定 h ≈ k·v，
   逐根算残差；某根残差过大 → 该柱"画错高度"，疑似误导性作图。

不依赖坐标轴 OCR（脆弱），只用"高度比例应与数值比例一致"这个强约束。
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from papercheck.findings import Finding, Category


@dataclass
class Bar:
    x: int
    width: int
    height: int   # 像素高度（从基线往上）


def detect_bars(path: str) -> tuple[list[Bar], int]:
    """检测共享基线的柱子，返回 (从左到右排序的 Bar 列表, 基线 y)。"""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return [], 0
    H, W = img.shape

    # 柱子通常比背景深，Otsu 反相二值化让柱子成为前景
    _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # 开运算去掉网格线/坐标轴细线和文字噪点
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(c) for c in cnts]
    # 过滤出"够大的竖直矩形"作为候选柱子
    rects = [r for r in rects
             if r[3] > 0.05 * H and r[2] > 0.012 * W and r[2] * r[3] > 0.002 * H * W]
    if not rects:
        return [], 0

    # 柱子共享同一基线：取最靠下的底边作为基线，只保留底边贴近它的矩形
    bottoms = sorted(y + h for _, y, _, h in rects)
    baseline = bottoms[-1]
    bars = [Bar(x, w, h) for (x, y, w, h) in rects if abs((y + h) - baseline) < 0.06 * H]
    bars.sort(key=lambda b: b.x)
    return bars, baseline


def measure_whiskers(image_path: str):
    """量每根柱顶上方误差棒的长度（中央窄条里向上数连续深色像素）。"""
    bars, baseline = detect_bars(image_path)
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None or not bars:
        return [], bars, baseline
    H, W = gray.shape
    lengths = []
    for bar in bars:
        cx = bar.x + bar.width // 2
        x0, x1 = max(0, cx - 1), min(W, cx + 2)
        strip = gray[:, x0:x1].min(axis=1)        # 中央 3 列里最深的
        y = baseline - bar.height - 2             # 从柱顶略上方起，跳过柱边
        L = 0
        while y > 0 and strip[y] < 110:           # 深色 = 误差棒线
            L += 1
            y -= 1
        lengths.append(L)
    return lengths, bars, baseline


def check_error_bars(image, min_bars: int = 3, max_cv: float = 0.06,
                     min_height_cv: float = 0.1) -> Finding | None:
    """误差棒异常（弱启发式）：各组误差棒长度几乎一致、但柱高差异明显 → 可疑。

    真实误差棒应随各组数据波动；若一刀切地完全一样（且柱高并不一样），
    提示误差棒可能是"画上去凑数"的。注意这是粗略启发，需人工复核。
    """
    lengths, bars, _ = measure_whiskers(image.path)
    if len(bars) < min_bars:
        return None
    L = np.array(lengths, dtype=float)
    heights = np.array([b.height for b in bars], dtype=float)
    if int((L > 4).sum()) < min_bars:
        return None  # 误差棒太短/检不到
    cv_L = float(L.std() / max(1.0, L.mean()))
    cv_H = float(heights.std() / max(1.0, heights.mean()))
    if cv_L >= max_cv or cv_H <= min_height_cv:
        return None
    return Finding(
        detector="error_bars", category=Category.CHART, severity=0.5,
        title="误差棒疑似异常（各组几乎一致但柱高不同）",
        description=(
            f"{image.label()}：检出 {len(bars)} 根柱，误差棒长度近乎完全一致"
            f"（变异系数 {cv_L:.1%}），而柱高差异明显（变异系数 {cv_H:.0%}）。"
            "真实误差棒应随各组数据波动，一刀切相同*可能*是后期画上去的，属辅助线索，待人工复核。"
        ),
        evidence={"image": image.path, "whisker_lengths": [int(x) for x in L],
                  "bar_heights": [int(h) for h in heights],
                  "whisker_cv": round(cv_L, 4), "height_cv": round(cv_H, 4)},
        locations=[image.label()],
    )


def check_bar_consistency(
    image,
    reported_values: list[float],
    tol: float = 0.15,
) -> Finding | None:
    """柱高比例 vs 报告数值比例。image 需有 .path / .label()。

    reported_values 顺序需与图中柱子从左到右一一对应。
    """
    bars, _ = detect_bars(image.path)
    if len(bars) != len(reported_values) or len(bars) < 2:
        return None  # 柱子数对不上，无法可靠比对，交给人工

    heights = np.array([b.height for b in bars], dtype=float)
    values = np.array(reported_values, dtype=float)
    if np.any(values <= 0):
        return None

    # 用各柱"高/值比例的中位数"标定 k：heights ≈ k · values。
    # 中位数对个别造假柱免疫（最小二乘会被异常柱带偏，导致全体误判）。
    k = float(np.median(heights / values))
    if k <= 0:
        return None
    predicted = k * values
    residuals = (heights - predicted) / predicted   # 相对残差

    bad = [(i, residuals[i]) for i in range(len(bars)) if abs(residuals[i]) > tol]
    if not bad:
        return None

    worst = max(bad, key=lambda t: abs(t[1]))
    sev = min(0.85, 0.45 + abs(worst[1]))
    details = "; ".join(
        f"第{i+1}根: 实测高{heights[i]:.0f}px, 按报告值{values[i]:g}应为{predicted[i]:.0f}px"
        f"（偏差{residuals[i]:+.0%}）"
        for i, _ in bad
    )
    return Finding(
        detector="chart_consistency",
        category=Category.CHART,
        severity=sev,
        title="柱状图条高与报告数值不成比例",
        description=(
            f"{image.label()}：检出 {len(bars)} 根柱子，对其高度做过原点标定后，"
            f"有 {len(bad)} 根与报告数值明显不成比例（阈值 {tol:.0%}）。{details}。"
            " 柱高比例本应与数值比例一致，偏差过大疑似误导性作图或标注错误，待核查。"
        ),
        evidence={
            "image": image.path,
            "heights_px": [round(float(h), 1) for h in heights],
            "reported_values": list(values),
            "predicted_px": [round(float(p), 1) for p in predicted],
            "residuals": [round(float(r), 3) for r in residuals],
            "bad_bars": [i + 1 for i, _ in bad],
        },
        locations=[image.label()],
    )
