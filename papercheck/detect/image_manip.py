"""单图内复制粘贴篡改（copy-move forgery）检测。

造假常见手法：在同一张图里把一块区域复制粘贴到别处（如把一个细胞/条带克隆一份）。
原理：提取关键点，让每个点在**同一张图内**找描述子相近的其他点；如果大量
匹配点对共享**同一个平移向量**(dx,dy)，说明存在一整块被平移复制的区域。

为避免把"自然重复纹理"误判，要求：描述子足够相近 + 空间距离足够远 +
共享同一平移向量的点对数量够多。
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from papercheck.findings import Finding, Category
from papercheck.detect._imaging import _load_gray, tone_levels


def find_copy_move(
    image,
    min_cluster: int = 30,
    min_spatial_dist: float = 30.0,
    desc_dist_max: int = 48,
    offset_bin: int = 8,
    min_corr: float = 0.85,
    min_region_levels: int = 15,
    max_periodic_corr: float = 0.6,
    viz_dir: str | Path | None = None,
) -> Finding | None:
    """检测单张图内的复制粘贴区域。image 需有 .path / .label() / .xref。"""
    img = _load_gray(image.path)
    if img is None:                 # 坏图/读不出 → 跳过，不崩
        return None
    orb = cv2.ORB_create(nfeatures=5000)
    kp, des = orb.detectAndCompute(img, None)
    if des is None or len(des) < 2 * min_cluster:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = bf.knnMatch(des, des, k=8)  # 每点找 8 个最近（含自身）

    # 按量化后的平移向量分桶；用规范化方向避免 a→b 和 b→a 双计
    buckets: dict[tuple, list] = defaultdict(list)
    for mlist in knn:
        for m in mlist:
            if m.queryIdx == m.trainIdx:
                continue
            if m.distance > desc_dist_max:
                continue
            p1 = kp[m.queryIdx].pt
            p2 = kp[m.trainIdx].pt
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            if (dx * dx + dy * dy) ** 0.5 < min_spatial_dist:
                continue  # 太近，可能是同一结构的相邻特征
            # 规范化方向（保证 a→b 与 b→a 落同一桶）。
            # 关键：翻转偏移符号时**必须同步交换点对**，否则同一桶里 p1 会混进
            # 源区和目标区两边的点，src/dst 包围盒被撑大，真实的紧凑簇被误杀。
            if (dx, dy) < (-dx, -dy):
                dx, dy = -dx, -dy
                p1, p2 = p2, p1
            key = (round(dx / offset_bin), round(dy / offset_bin))
            buckets[key].append((p1, p2))

    if not buckets:
        return None

    H, W = img.shape[:2]
    area = float(W * H)

    def _bbox_frac(pts: np.ndarray) -> float:
        w = pts[:, 0].max() - pts[:, 0].min()
        h = pts[:, 1].max() - pts[:, 1].min()
        return (w * h) / area

    # 空间紧凑度闸门：真正的 copy-move 是一整块连续区域被搬走，源点/目标点
    # 各自聚成一团；自然纹理重复（如一堆相似细胞）的匹配点散布全图。
    # 关键：**遍历每个偏移簇**逐个过闸，而不是只看最大簇——真正的复制块
    # 往往不是点数最多的簇，只看最大簇会被散乱的纹理簇盖过、漏掉真阳性。
    best = None  # (n, key, pairs, src_frac, dst_frac)
    for key, pairs in buckets.items():
        if len(pairs) < min_cluster:
            continue
        src = np.array([p1 for p1, _ in pairs])
        dst = np.array([p2 for _, p2 in pairs])
        sf, df = _bbox_frac(src), _bbox_frac(dst)
        if sf > 0.30 or df > 0.30:
            continue  # 点散布大半张图 → 纹理重复，跳过
        if best is None or len(pairs) > best[0]:
            best = (len(pairs), key, pairs, sf, df)

    if best is None:
        return None
    n, best_key, best_pairs, src_frac, dst_frac = best
    # 用匹配点对的中位数偏移（精确），而非量化的 best_key×bin——量化的 ±4px 错位
    # 会在细纹理上把像素相关性拉低，误杀真克隆。
    _off = np.array([(p2[0] - p1[0], p2[1] - p1[1]) for p1, p2 in best_pairs])
    dx, dy = float(np.median(_off[:, 0])), float(np.median(_off[:, 1]))

    # 像素验证 + 区域内容验证：真克隆 → 像素高度相关 **且** 源区是连续色调；
    # 落在文字标签/纯色块上的"高相关"灰阶数很少，滤掉。
    corr, region_levels = _region_corr(img, best_pairs, dx, dy)
    if corr < min_corr or region_levels < min_region_levels:
        return None

    # 周期纹理闸门：若整图按同一偏移平移就高度自相似（斜纹/等距条带/规则刻度/
    # 等间距泳道），说明这是周期结构而非"某一块被局部克隆"。真克隆是局部的——
    # 整图平移只有那一块对得上，全局相关性低。
    if _global_shift_corr(img, dx, dy) > max_periodic_corr:
        return None

    # 紧凑度越高、点数越多、像素越吻合 → 越可疑
    compact = 1.0 - min(1.0, (src_frac + dst_frac) / 0.60)
    sev = min(0.93, 0.38 + n / 80.0 + 0.12 * compact + 0.15 * min(1.0, corr))
    viz_path = None
    if viz_dir is not None:
        viz_dir = Path(viz_dir)
        viz_dir.mkdir(parents=True, exist_ok=True)
        viz_path = str(viz_dir / f"copymove_{image.xref}.png")
        _draw_copymove(img, best_pairs, viz_path)

    return Finding(
        detector="image_manip",
        category=Category.IMAGE_MANIP,
        severity=sev,
        title="疑似单图内复制粘贴篡改 (copy-move)",
        description=(
            f"{image.label()} 内有 {n} 组特征点共享同一平移向量 (~{dx:.0f},{dy:.0f}) 像素，"
            f"且源区与目标区按此偏移对齐后像素相关性达 {corr:.2f}，"
            f"提示存在一块被复制粘贴到别处的区域。建议人工核查该图是否有克隆痕迹。"
        ),
        evidence={
            "image": image.path,
            "n_matches": n,
            "offset_px": [round(dx, 1), round(dy, 1)],
            "pixel_corr": round(corr, 3),
            "copymove_viz": viz_path,
        },
        locations=[image.label()],
    )


def _global_shift_corr(img: np.ndarray, dx: float, dy: float) -> float:
    """整图按 (dx,dy) 平移后与自身的相关系数。周期纹理→高；局部克隆→低（只有那一块对得上）。"""
    H, W = img.shape[:2]
    dx, dy = int(round(dx)), int(round(dy))
    x_lo, x_hi = max(0, -dx), min(W, W - dx)
    y_lo, y_hi = max(0, -dy), min(H, H - dy)
    if x_hi - x_lo < 20 or y_hi - y_lo < 20:
        return 0.0
    a = img[y_lo:y_hi, x_lo:x_hi].astype(np.float32)
    b = img[y_lo + dy:y_hi + dy, x_lo + dx:x_hi + dx].astype(np.float32)
    if a.std() < 1e-3 or b.std() < 1e-3:
        return 0.0
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def _region_corr(img: np.ndarray, pairs: list, dx: float, dy: float):
    """返回 (源区与平移后目标区的像素相关系数, 源区灰阶数)。真克隆 corr≈1；
    灰阶数低=匹配落在文字/线条/纯色块（斜纹柱等）上的结构巧合。"""
    H, W = img.shape[:2]
    src = np.array([p1 for p1, _ in pairs])
    x0, y0 = int(src[:, 0].min()), int(src[:, 1].min())
    x1, y1 = int(src[:, 0].max()) + 1, int(src[:, 1].max()) + 1
    dx, dy = int(round(dx)), int(round(dy))
    # 取源区与目标区都落在图内的公共范围
    sx_lo, sx_hi = max(x0, 0, -dx), min(x1, W, W - dx)
    sy_lo, sy_hi = max(y0, 0, -dy), min(y1, H, H - dy)
    if sx_hi - sx_lo < 12 or sy_hi - sy_lo < 12:
        return 0.0, 0
    a = img[sy_lo:sy_hi, sx_lo:sx_hi].astype(np.float32)
    b = img[sy_lo + dy:sy_hi + dy, sx_lo + dx:sx_hi + dx].astype(np.float32)
    levels = tone_levels(a)
    if a.shape != b.shape or a.std() < 1e-3 or b.std() < 1e-3:
        return 0.0, levels
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1]), levels


def _draw_copymove(gray: np.ndarray, pairs: list, out_path: str) -> None:
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for p1, p2 in pairs:
        a = (int(p1[0]), int(p1[1]))
        b = (int(p2[0]), int(p2[1]))
        cv2.line(vis, a, b, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.circle(vis, a, 3, (0, 200, 0), -1)
        cv2.circle(vis, b, 3, (0, 200, 0), -1)
    cv2.imwrite(out_path, vis)
