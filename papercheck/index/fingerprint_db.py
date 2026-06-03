"""图片指纹库（SQLite），支撑跨论文/跨作者批量盗图比对。

入库：每张抽出的图存 phash + sha1 + 路径 + 来源论文。
比对：跨**不同论文**的图两两比，sha1 相同＝铁证，否则 phash 粗筛 + ORB 几何确认
（能抓到裁剪/旋转/翻转后跨论文复用的同一张图——这正是耿同学批量打假的核心）。
"""
from __future__ import annotations

import sqlite3
from itertools import combinations
from pathlib import Path

import imagehash

from papercheck.findings import Finding, Category
from papercheck.detect._imaging import phash, match_images

_SCHEMA = """
CREATE TABLE IF NOT EXISTS images(
  id INTEGER PRIMARY KEY,
  paper_id TEXT NOT NULL,
  xref INTEGER NOT NULL,
  path TEXT NOT NULL,
  width INTEGER, height INTEGER,
  sha1 TEXT, phash TEXT,
  UNIQUE(paper_id, xref)
);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(db_path))
    con.executescript(_SCHEMA)
    return con


def add_images(con: sqlite3.Connection, images) -> int:
    """把一篇论文抽出的图片入库（按 paper_id+xref 去重，重复入库会覆盖）。"""
    n = 0
    for im in images:
        con.execute(
            "INSERT OR REPLACE INTO images(paper_id,xref,path,width,height,sha1,phash)"
            " VALUES(?,?,?,?,?,?,?)",
            (im.paper_id, im.xref, im.path, im.width, im.height, im.sha1, str(phash(im.path))),
        )
        n += 1
    con.commit()
    return n


def paper_count(con: sqlite3.Connection) -> int:
    return con.execute("SELECT COUNT(DISTINCT paper_id) FROM images").fetchone()[0]


def _label(paper_id: str, xref: int) -> str:
    return f"{paper_id}#img{xref}"


def find_cross_paper_dups(
    con: sqlite3.Connection,
    phash_thresh: int = 40,
    min_inliers: int = 22,
    thorough: bool = True,
    max_orb_pairs: int = 8000,
) -> list[Finding]:
    """跨不同论文找复用同一张图的配对。

    thorough=True：对所有跨论文图对都跑 ORB（能抓局部/变换复用），但配对数超过
    max_orb_pairs 时自动退化为 phash 粗筛以控成本（会漏纯局部复用，运行时会提示）。
    """
    rows = con.execute(
        "SELECT paper_id,xref,path,sha1,phash FROM images ORDER BY paper_id,xref"
    ).fetchall()

    cross_pairs = [
        (a, b) for a, b in combinations(rows, 2) if a[0] != b[0]  # 不同论文
    ]
    prefilter = (not thorough) or len(cross_pairs) > max_orb_pairs
    note = ""
    if prefilter and len(cross_pairs) > max_orb_pairs:
        note = f"（跨论文配对 {len(cross_pairs)} 对超过阈值，已用 phash 粗筛，纯局部复用可能漏检）"

    findings: list[Finding] = []
    for a, b in cross_pairs:
        pa, xa, patha, sha_a, pha = a
        pb, xb, pathb, sha_b, phb = b

        if sha_a == sha_b:
            findings.append(Finding(
                detector="cross_paper", category=Category.CROSS_PAPER, severity=0.97,
                title="跨论文使用了像素级完全相同的图片",
                description=(f"《{pa}》的 img#{xa} 与《{pb}》的 img#{xb} 字节完全一致（sha1 相同）。"
                             " 同一张图出现在两篇不同论文中，疑似盗图/自我抄袭，待核查。"),
                evidence={"image_a": patha, "image_b": pathb, "match": "identical",
                          "paper_a": pa, "paper_b": pb, "sha1": sha_a},
                locations=[_label(pa, xa), _label(pb, xb)],
            ))
            continue

        pdist = imagehash.hex_to_hash(pha) - imagehash.hex_to_hash(phb)
        if prefilter and pdist > phash_thresh:
            continue

        mr = match_images(patha, pathb)
        # 与单篇内一致的三门槛：内点 + 像素相关 + 匹配区域连续色调（之前漏了 region_levels）
        if mr.n_inliers < min_inliers or mr.pixel_corr < 0.45 or mr.region_levels < 15:
            continue

        sev = min(0.93, 0.45 + 0.5 * min(1.0, mr.n_inliers / 50.0))
        flipped = mr.transform == "flip"
        findings.append(Finding(
            detector="cross_paper", category=Category.CROSS_PAPER, severity=sev,
            title="跨论文疑似复用同一张图" + ("（翻转后）" if flipped else "（局部/变换后）"),
            description=(
                f"《{pa}》img#{xa} 与《{pb}》img#{xb} 有 {mr.n_inliers} 个几何一致的特征对应点"
                f"（phash 距离 {pdist}{'，含镜像翻转' if flipped else ''}）。"
                f" 两篇不同论文的图高度疑似同源，待核查。{note}"
            ),
            evidence={"image_a": patha, "image_b": pathb, "n_inliers": mr.n_inliers,
                      "phash_dist": int(pdist), "transform": mr.transform,
                      "paper_a": pa, "paper_b": pb},
            locations=[_label(pa, xa), _label(pb, xb)],
        ))

    findings.sort(key=lambda f: f.severity, reverse=True)
    return findings
