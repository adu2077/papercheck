"""图片查重：在一组图片里找"互相复用"的配对。

三档证据：
1. sha1 完全相同      → 像素级同一张图被重复插入（最硬）
2. 感知哈希很近        → 整图近似（重压缩/缩放后复用）
3. ORB 几何一致内点多  → 局部/旋转/翻转/裁剪后的复用（phash 抓不到这种）

第 3 档最关键：phash 是全局哈希，**局部复制**（把一张图的一块贴到另一张）
不会让 phash 接近，必须靠 ORB 特征匹配。所以单篇内部对所有图对都跑 ORB。
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from pathlib import Path

from papercheck.findings import Finding, Category
from papercheck.detect._imaging import phash, hamming, match_images, render_match


def _score(n_inliers: int, inlier_ratio: float, phash_dist: int, corr: float) -> float:
    """把匹配证据折算成置信度。

    像素相关性(corr)是主门槛信号——真复用底下像素相同；
    内点数与整图近似(phash 近)再加成。
    """
    base = min(1.0, n_inliers / 50.0)
    s = 0.35 + 0.35 * base + 0.22 * min(1.0, max(0.0, corr))
    if phash_dist <= 8:                         # 整图也近似 → 更可能是同一张
        s += 0.05
    return min(0.97, s)


def _kind(flipped: bool, pdist: int) -> str:
    return "翻转后复用" if flipped else ("整图近似复用" if pdist <= 8 else "局部/变换后复用")


def _pair_finding(a, b, mr, pdist, sev, viz_dir) -> Finding:
    flipped = mr.transform == "flip"
    viz_path = None
    if viz_dir is not None:
        viz_path = str(Path(viz_dir) / f"match_{a.xref}_{b.xref}.png")
        if not render_match(a.path, b.path, viz_path):
            viz_path = None
    return Finding(
        detector="image_dup", category=Category.IMAGE_DUP, severity=sev,
        title=f"疑似图片复用（{_kind(flipped, pdist)}）",
        description=(
            f"{a.label()} 与 {b.label()} 存在 {mr.n_inliers} 个几何一致的特征对应点"
            f"（内点率 {mr.inlier_ratio:.0%}，对齐后像素相关性 {mr.pixel_corr:.2f}，phash 距离 {pdist}）。"
            + ("检测到镜像翻转关系。" if flipped else "")
            + " 几何一致 + 像素高度相关，强烈提示同源，建议人工核查。"
        ),
        evidence={"image_a": a.path, "image_b": b.path, "n_inliers": mr.n_inliers,
                  "n_good": mr.n_good, "inlier_ratio": round(mr.inlier_ratio, 3),
                  "pixel_corr": round(mr.pixel_corr, 3), "phash_dist": pdist,
                  "transform": mr.transform, "match_viz": viz_path},
        locations=[a.label(), b.label()],
    )


def _cluster_finding(edges, by_xref, viz_dir) -> Finding:
    """把一组互相近似的面板收敛成一条"该区域被跨面板复用"的发现。"""
    edges = sorted(edges, key=lambda e: e[4], reverse=True)   # 按严重度
    a, b, mr, pdist, sev = edges[0]                            # 最强配对做代表/可视化
    node_xrefs = sorted({x for e in edges for x in (e[0].xref, e[1].xref)})
    members = [by_xref[x].label() for x in node_xrefs]
    viz_path = None
    if viz_dir is not None:
        viz_path = str(Path(viz_dir) / f"match_{a.xref}_{b.xref}.png")
        if not render_match(a.path, b.path, viz_path):
            viz_path = None
    pairs = [{"a": e[0].label(), "b": e[1].label(), "n_inliers": e[2].n_inliers,
              "pixel_corr": round(e[2].pixel_corr, 3), "phash_dist": e[3],
              "transform": e[2].transform} for e in edges]
    return Finding(
        detector="image_dup", category=Category.IMAGE_DUP, severity=sev,
        title=f"疑似同一区域被跨面板复用（{len(members)} 个面板成一族）",
        description=(
            f"{len(members)} 个面板互相高度相似（{len(edges)} 处几何一致+像素相关的配对）："
            f"{'、'.join(members)}。同一图像区域在多处出现，常见于上样对照(如 β-actin/α-tubulin)"
            f"被复用，也可能是数据图被跨实验重复——需人工核查是否合理。最强配对："
            f"{a.label()}↔{b.label()}（{mr.n_inliers} 内点, 像素相关 {mr.pixel_corr:.2f}）。"
        ),
        evidence={"members": members, "n_panels": len(members), "n_pairs": len(edges),
                  "pairs": pairs, "image_a": a.path, "image_b": b.path,
                  "match_viz": viz_path, "pixel_corr": round(mr.pixel_corr, 3),
                  "transform": mr.transform, "phash_dist": pdist},
        locations=members,
    )


def find_duplicate_pairs(
    images: list,
    min_inliers: int = 22,
    min_corr: float = 0.45,
    min_region_levels: int = 15,
    run_all_orb: bool = True,
    viz_dir: str | Path | None = None,
) -> list[Finding]:
    """对图片两两比对，产出可疑复用 Finding。

    images: ExtractedImage 列表（需有 .path / .sha1 / .label()）。
    run_all_orb: True=对所有图对跑 ORB（单篇内推荐，能抓局部复用）；
                 False=只对 phash 接近的图对跑 ORB（跨大库时省算力，会漏纯局部复用）。
    """
    findings: list[Finding] = []
    if viz_dir:
        viz_dir = Path(viz_dir)
        viz_dir.mkdir(parents=True, exist_ok=True)

    ph = {im.path: phash(im.path) for im in images}
    by_xref = {im.xref: im for im in images}
    edges = []  # (a, b, mr, pdist, sev) —— 通过门槛的 ORB 配对，待聚类

    for a, b in combinations(images, 2):
        pdist = hamming(ph[a.path], ph[b.path])

        # 第 1 档：像素级完全相同 → 直接成 finding（最硬，不参与聚类）
        if a.sha1 == b.sha1:
            findings.append(Finding(
                detector="image_dup", category=Category.IMAGE_DUP, severity=0.96,
                title="同一张图片被重复使用（像素级完全相同）",
                description=f"{a.label()} 与 {b.label()} 字节完全一致（sha1 相同），属同一张图重复插入。",
                evidence={"image_a": a.path, "image_b": b.path,
                          "phash_dist": 0, "match": "identical", "sha1": a.sha1},
                locations=[a.label(), b.label()],
            ))
            continue

        if not run_all_orb and pdist > 16:
            continue

        mr = match_images(a.path, b.path)
        # 三门槛：内点够多 + 像素真相关 + 匹配区域是连续色调
        if mr.n_inliers < min_inliers or mr.pixel_corr < min_corr or mr.region_levels < min_region_levels:
            continue
        edges.append((a, b, mr, pdist, _score(mr.n_inliers, mr.inlier_ratio, pdist, mr.pixel_corr)))

    # 按连通分量聚类：N 个互相近似的面板（如都含 β-actin 上样对照）收敛成一条，
    # 而不是 N(N-1)/2 条两两配对刷屏。
    parent: dict = {}

    def _find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, *_ in edges:
        parent[_find(a.xref)] = _find(b.xref)

    comps: dict = defaultdict(list)
    for e in edges:
        comps[_find(e[0].xref)].append(e)

    for comp_edges in comps.values():
        nodes = {x for e in comp_edges for x in (e[0].xref, e[1].xref)}
        if len(nodes) <= 2:
            a, b, mr, pdist, sev = comp_edges[0]
            findings.append(_pair_finding(a, b, mr, pdist, sev, viz_dir))
        else:
            findings.append(_cluster_finding(comp_edges, by_xref, viz_dir))

    findings.sort(key=lambda f: f.severity, reverse=True)
    return findings
