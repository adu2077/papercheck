"""图像匹配底座：感知哈希 + ORB 特征 + 几何验证。

被 image_dup（跨图复用）和 image_manip（单图内复制粘贴）共用。

关键点：靠 RANSAC 单应矩阵做几何验证，因此**旋转/翻转/缩放/裁剪**后的
复用也能识别——这正是耿同学/Bik 抓的那类"换个角度再用一次"的图。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image
import imagehash


def _stat_key(path: str) -> tuple:
    """(mtime_ns, size) 做缓存键的一部分——否则同名文件被覆盖后会取到**陈旧缓存**
    （面板文件名 x{xref}_p{pid}.png 在同名论文/复跑时会重名）。"""
    try:
        st = os.stat(path)
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return (0, 0)


# ---------- 感知哈希（粗筛，便宜） ----------

@lru_cache(maxsize=4096)
def _phash_k(path: str, key: tuple) -> imagehash.ImageHash:
    with Image.open(path) as im:
        return imagehash.phash(im.convert("L"), hash_size=16)


def phash(path: str) -> imagehash.ImageHash:
    return _phash_k(path, _stat_key(path))


def hamming(h1: imagehash.ImageHash, h2: imagehash.ImageHash) -> int:
    return h1 - h2


# ---------- ORB 特征（精筛，几何验证） ----------

@lru_cache(maxsize=512)
def _load_gray_k(path: str, key: tuple, max_side: int = 1200):
    """读灰度图，缩到 max_side 内。坏图/读不出返回 None（不让整批崩）。"""
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        try:
            im = np.array(Image.open(path).convert("L"))   # cv2 偶尔读不了某些 PNG，回退 PIL
        except Exception:
            return None
    if im is None or im.size == 0:
        return None
    h, w = im.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        im = cv2.resize(im, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return im


def _load_gray(path: str, max_side: int = 1200):
    return _load_gray_k(path, _stat_key(path), max_side)


@lru_cache(maxsize=512)
def _orb_k(path: str, key: tuple, nfeatures: int, flip: bool):
    img = _load_gray(path)
    if img is None:
        return np.empty((0, 2), np.float32), None
    if flip:
        img = cv2.flip(img, 1)
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp, des = orb.detectAndCompute(img, None)
    pts = np.float32([k.pt for k in kp]) if kp else np.empty((0, 2), np.float32)
    return pts, des


def _orb_features(path: str, nfeatures: int = 2000, flip: bool = False):
    """ORB 关键点坐标 + 描述子。flip=True 时在水平镜像后的图上算
    （ORB 非镜像不变，正/反各试取更优，稳定抓"翻转复用"）。"""
    return _orb_k(path, _stat_key(path), nfeatures, flip)


@dataclass
class MatchResult:
    n_good: int           # Lowe ratio 通过的匹配数
    n_inliers: int        # RANSAC 几何一致的内点数
    inlier_ratio: float   # n_inliers / n_good
    transform: str        # "similar" | "flip" | "none"
    pixel_corr: float = 0.0   # 对齐后匹配区域的像素相关性（真复用才高）
    region_levels: int = 0    # 匹配区域的有效灰阶数（连续色调 vs 文字/线条/纯色块）


def tone_levels(values: np.ndarray) -> int:
    """一组像素值里"有质量的灰阶数"。连续色调(印迹/照片)多(~20+)，文字/线条/纯色块少(<15)。"""
    if values.size == 0:
        return 0
    hist = np.bincount(values.astype(np.uint8).ravel(), minlength=256).astype(np.float32)
    hist /= float(values.size)
    return int((hist > 0.004).sum())


def _pixel_corr(img1: np.ndarray, img2: np.ndarray, H: np.ndarray, dst_inliers: np.ndarray):
    """把 img2 按 H 对齐到 img1，在内点包围盒内算 (像素相关系数, 区域灰阶数)。

    corr 高=像素真复用；region_levels 低=匹配落在文字/线条/纯色块上（结构巧合），
    而非真图像内容——后者专治"匹配点全在文字标签/斜纹柱上"的高相关误报。
    """
    h1, w1 = img1.shape[:2]
    try:
        warped = cv2.warpPerspective(img2, H, (w1, h1))
        cover = cv2.warpPerspective(np.full(img2.shape[:2], 255, np.uint8), H, (w1, h1))
    except cv2.error:
        return 0.0, 0
    xs, ys = dst_inliers[:, 0], dst_inliers[:, 1]
    x0, x1 = max(0, int(xs.min())), min(w1, int(xs.max()) + 1)
    y0, y1 = max(0, int(ys.min())), min(h1, int(ys.max()) + 1)
    if x1 - x0 < 12 or y1 - y0 < 12:
        return 0.0, 0
    a = img1[y0:y1, x0:x1].astype(np.float32)
    b = warped[y0:y1, x0:x1].astype(np.float32)
    m = cover[y0:y1, x0:x1] > 0
    if m.sum() < 64:
        return 0.0, 0
    av, bv = a[m], b[m]
    levels = tone_levels(av)
    if av.std() < 1e-3 or bv.std() < 1e-3:
        return 0.0, levels
    return float(np.corrcoef(av, bv)[0, 1]), levels


def _match_core(pts1, des1, pts2, des2, img1, img2, ratio: float):
    """返回 (n_good, n_inliers, pixel_corr, region_levels)。"""
    if des1 is None or des2 is None or len(des1) < 8 or len(des2) < 8:
        return 0, 0, 0.0, 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = bf.knnMatch(des1, des2, k=2)
    good = [m for pair in knn if len(pair) == 2 for m, n in [pair] if m.distance < ratio * n.distance]
    if len(good) < 8:
        return len(good), 0, 0.0, 0
    # H 把 img2 坐标映射到 img1，便于把 img2 warp 到 img1 上比像素
    p2 = np.float32([pts2[m.trainIdx] for m in good]).reshape(-1, 1, 2)
    p1 = np.float32([pts1[m.queryIdx] for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(p2, p1, cv2.RANSAC, 5.0)
    if H is None or mask is None:
        return len(good), 0, 0.0, 0
    n_in = int(mask.sum())
    dst_inliers = p1.reshape(-1, 2)[mask.ravel() == 1]
    corr, levels = _pixel_corr(img1, img2, H, dst_inliers) if n_in >= 8 else (0.0, 0)
    return len(good), n_in, corr, levels


def match_images(path1: str, path2: str, ratio: float = 0.75) -> MatchResult:
    """两图做 ORB+RANSAC+像素相关性匹配（正向 + 镜像各一次，取更优）。

    真复用要求：几何一致内点多 **且** 对齐后像素相关性高 **且** 匹配区域是连续色调。
    """
    img1 = _load_gray(path1)
    gray2 = _load_gray(path2)
    if img1 is None or gray2 is None:          # 坏图/读不出 → 安全返回，不崩
        return MatchResult(0, 0, 0.0, "none", 0.0, 0)
    pts1, des1 = _orb_features(path1)
    best = (0, 0, 0.0, 0, "none")  # (n_in, n_good, corr, levels, transform)
    for flip in (False, True):
        img2 = cv2.flip(gray2, 1) if flip else gray2
        pts2, des2 = _orb_features(path2, flip=flip)
        ng, ni, corr, levels = _match_core(pts1, des1, pts2, des2, img1, img2, ratio)
        # 以"像素相关才算数的内点数"择优，避免选到像素不符的镜像巧合
        eff = ni if corr >= 0.4 else ni * 0.2
        best_eff = best[0] if best[2] >= 0.4 else best[0] * 0.2
        if eff > best_eff or best[4] == "none":
            best = (ni, ng, corr, levels, "flip" if flip else "similar")
    ni, ng, corr, levels, transform = best
    return MatchResult(ng, ni, ni / max(1, ng), transform if ni else "none", corr, levels)


def render_match(path1: str, path2: str, out_path: str, ratio: float = 0.75) -> bool:
    """把两图的 RANSAC 内点匹配画成连线图存盘（报告里的"铁证图"）。

    只对已确认的可疑配对调用（数量少），所以这里重算 ORB、用真正的 KeyPoint 画线。
    """
    img1 = _load_gray(path1)
    img2 = _load_gray(path2)
    if img1 is None or img2 is None:
        return False
    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(img1, None)

    # 正/反两种朝向都算，画内点多的那个（翻转复用时对镜像图画连线才对得上）
    best = None  # (n_good_via_lowe, kp2, des2, img2)
    for flip in (False, True):
        im2 = cv2.flip(img2, 1) if flip else img2
        kp2, des2 = orb.detectAndCompute(im2, None)
        if des1 is None or des2 is None or len(des1) < 8 or len(des2) < 8:
            continue
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        knn = bf.knnMatch(des1, des2, k=2)
        good = [m for pair in knn if len(pair) == 2 for m, n in [pair] if m.distance < ratio * n.distance]
        if best is None or len(good) > best[0]:
            best = (len(good), kp2, des2, im2, good)
    if best is None or best[0] < 8:
        return False
    _, kp2, des2, img2, good = best

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None or mask is None:
        return False

    inliers = [g for g, keep in zip(good, mask.ravel().tolist()) if keep]
    vis = cv2.drawMatches(
        img1, kp1, img2, kp2, inliers, None,
        matchColor=(0, 200, 0), singlePointColor=(120, 120, 120),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    return bool(cv2.imwrite(out_path, vis))
