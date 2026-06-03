"""② 统计异常检测器：本福特定律 / GRIM·GRIMMER / statcheck / 重复数据点。

理念：论文报告的数字之间存在**内部约束**，造假/笔误会破坏这些约束，
不看懂内容也能用数学抓出来。

- statcheck   ：t/F/r/z/χ² 统计量 + 自由度能反推 p 值，和作者写的 p 比对。
- GRIM        ：整数数据的均值必须是 k/n，granularity 限制了它能取的小数。
- 本福特定律  ：自然产生、跨多个数量级的数据，首位数分布服从 log 律。
- 重复数据点  ：整行/整段数字被复制粘贴，是编造数据的常见痕迹。

所有检查只产出"可疑线索 + 证据 + 置信度"，措辞用"疑似/待核查"。
"""
from __future__ import annotations

import math
import re
from collections import Counter
from itertools import combinations

from scipy import stats as sps

from papercheck.findings import Finding, Category


# ============================ statcheck ============================

def _recompute_p(s) -> float | None:
    """从报告的统计量 + 自由度反推双侧 p 值。s 是 ReportedStat。

    自由度非法（≤0，PDF/OCR 易抽错成负数或 0）会让 scipy 返回 nan——必须拦掉，
    否则下游 `nan < 0.05 == False` 会冒出假的"显著性翻转"。
    """
    try:
        if s.kind == "t":
            if not s.df1 or s.df1 <= 0:
                return None
            p = 2 * sps.t.sf(abs(s.statistic), s.df1)
        elif s.kind == "F":
            if not s.df1 or not s.df2 or s.df1 <= 0 or s.df2 <= 0:
                return None
            p = sps.f.sf(s.statistic, s.df1, s.df2)
        elif s.kind == "r":
            if not s.df1 or s.df1 <= 0:
                return None
            r = max(-0.999999, min(0.999999, s.statistic))
            t = r * math.sqrt(s.df1 / (1 - r * r))
            p = 2 * sps.t.sf(abs(t), s.df1)
        elif s.kind == "z":
            p = 2 * sps.norm.sf(abs(s.statistic))
        elif s.kind == "chi2":
            if not s.df1 or s.df1 <= 0:
                return None
            p = sps.chi2.sf(s.statistic, s.df1)
        else:
            return None
    except Exception:
        return None
    p = float(p)
    if math.isnan(p) or math.isinf(p):
        return None
    return p


def _reported_significant(s, alpha: float = 0.05) -> bool:
    """作者报告的 p 是否在主张'显著'(p<alpha)。

    `<`：p<X，仅当 X≤alpha 才确立显著。
    `>`：p>X，**永远无法确立显著**（p 比某值大，下界不封顶）→ 一律按不显著处理，
         此前 `p_reported < alpha` 把 "p>.001" 误判为显著，是 bug。
    """
    if s.p_comparator == "<":
        return s.p_reported <= alpha
    if s.p_comparator == ">":
        return False
    return s.p_reported < alpha


def statcheck(reported_stats) -> list[Finding]:
    """重算每条统计结果的 p，和报告值比对。重点抓'显著性翻转'这种硬矛盾。"""
    findings: list[Finding] = []
    for s in reported_stats:
        p_comp = _recompute_p(s)
        if p_comp is None:
            continue

        rep_sig = _reported_significant(s)
        comp_sig = p_comp < 0.05

        if rep_sig != comp_sig:
            # 显著性翻转：报告说显著但重算不显著（或反之）——最硬的矛盾
            findings.append(Finding(
                detector="statcheck",
                category=Category.STATS,
                severity=0.85,
                title="报告的 p 值与统计量自相矛盾（显著性翻转）",
                description=(
                    f"原文「{s.raw.strip()}」：由 {s.kind} 统计量与自由度重算得 p≈{p_comp:.3g}，"
                    f"与报告的 p {s.p_comparator} {s.p_reported:g} 在显著性判定上相反"
                    f"（重算{'显著' if comp_sig else '不显著'}，报告{'显著' if rep_sig else '不显著'}）。"
                    " 这类内部矛盾通常意味着统计量、自由度或 p 值有误，待核查。"
                ),
                evidence={"kind": s.kind, "statistic": s.statistic, "df1": s.df1, "df2": s.df2,
                          "p_reported": s.p_reported, "p_comparator": s.p_comparator,
                          "p_recomputed": round(p_comp, 5), "raw": s.raw.strip()},
                locations=[f"p.{s.page}"],
            ))
        elif s.p_comparator == "=" and abs(p_comp - s.p_reported) > 0.02 and (
            max(p_comp, s.p_reported) / max(1e-6, min(p_comp, s.p_reported)) > 2
        ):
            # 同侧但数量级明显对不上 → 较弱的线索
            findings.append(Finding(
                detector="statcheck",
                category=Category.STATS,
                severity=0.45,
                title="报告的 p 值与统计量数值不吻合",
                description=(
                    f"原文「{s.raw.strip()}」：重算 p≈{p_comp:.3g}，与报告 p={s.p_reported:g} "
                    f"差距较大（未翻转显著性，但数值明显不符），待核查是否笔误。"
                ),
                evidence={"kind": s.kind, "statistic": s.statistic,
                          "p_reported": s.p_reported, "p_recomputed": round(p_comp, 5),
                          "raw": s.raw.strip()},
                locations=[f"p.{s.page}"],
            ))
    return findings


# ============================ GRIM ============================

def grim_consistent(mean: float, n: int, n_items: int = 1, decimals: int | None = None) -> bool:
    """GRIM 检验：n 个（每个由 n_items 道整数题求和的）被试，其均值能否取到报告的小数。

    整数数据的可能均值只能是 (整数)/(n*n_items)。把报告均值乘回去四舍五入再除，
    若按报告小数位四舍五入回不到原值 → 该均值不可能，疑似编造/笔误。
    """
    if n <= 0:
        return True  # 无法判定
    if decimals is None:
        s = f"{mean}"
        decimals = len(s.split(".")[1]) if "." in s else 0
    g = n * n_items
    nearest_int = round(mean * g)
    reconstructed = nearest_int / g
    return round(reconstructed, decimals) == round(mean, decimals)


def _decimals(x) -> int:
    s = repr(float(x))
    if "e" in s or "E" in s:
        return 0
    return len(s.split(".")[1]) if "." in s else 0


def grimmer_consistent(mean: float, sd: float, n: int, n_items: int = 1,
                       mean_decimals: int | None = None, sd_decimals: int | None = None) -> bool:
    """GRIMMER 检验：整数数据下，报告的 SD 是否数学上可能。

    N=n·n_items 个整数，其和 X=round(mean·N)；样本方差 SD²=(Y−X²/N)/(N−1) ⇒
    平方和 Y=SD²(N−1)+X²/N 必须是非负整数。n_items=1 时还要求 Y≡X(mod 2)。
    在 SD 的舍入区间内若找不到这样的整数 Y → 该 SD 不可能（疑似编造/笔误）。
    """
    N = n * n_items
    if N <= 1:
        return True
    md = mean_decimals if mean_decimals is not None else _decimals(mean)
    sdd = sd_decimals if sd_decimals is not None else _decimals(sd)
    if not grim_consistent(mean, n, n_items, md):
        return False  # 均值本身就不可能，SD 无从谈起
    X = round(mean * N)
    half = 0.5 * 10 ** (-sdd)
    sd_lo, sd_hi = max(0.0, sd - half), sd + half
    y_lo = sd_lo ** 2 * (N - 1) + X * X / N
    y_hi = sd_hi ** 2 * (N - 1) + X * X / N
    lo, hi = math.ceil(y_lo - 1e-9), math.floor(y_hi + 1e-9)
    for Y in range(lo, hi + 1):
        if Y >= 0 and (n_items > 1 or (Y % 2) == (X % 2)):
            return True
    return False


def grimmer_check(mean_sd_pairs) -> list[Finding]:
    """mean_sd_pairs: 可迭代 (mean, sd, n, n_items, page, label)。"""
    findings: list[Finding] = []
    for item in mean_sd_pairs:
        mean, sd, n, n_items, page, label = item[:6]
        md = item[6] if len(item) > 6 else None      # 原始字符串小数位（可靠）
        sdd = item[7] if len(item) > 7 else None
        if grimmer_consistent(mean, sd, n, n_items, md, sdd):
            continue
        findings.append(Finding(
            detector="grimmer", category=Category.STATS, severity=0.7,
            title="标准差不满足 GRIMMER 一致性（整数数据下不可能）",
            description=(
                f"{label}：报告均值 {mean}、标准差 {sd}、n={n}"
                f"{'、每人 '+str(n_items)+' 题' if n_items>1 else ''} 在整数数据下数学上无法同时成立"
                f"（平方和须为整数且与均值和同奇偶）。疑似统计量编造或笔误，待核查。"
            ),
            evidence={"mean": mean, "sd": sd, "n": n, "n_items": n_items},
            locations=[f"p.{page}" if page else label],
        ))
    return findings


def _ss_bounds(X: int, n: int, a: int, b: int):
    """和固定为 X 的 n 个 [a,b] 整数，其平方和的 (最小, 最大)。

    最小：取值尽量均匀（相差≤1）；最大：尽量堆到边界（平方凸，集中增量使 SS 最大）。
    """
    base, rem = divmod(X, n)
    min_ss = (base + 1) ** 2 * rem + base ** 2 * (n - rem)
    data = [a] * n
    D = X - a * n
    i = 0
    while D > 0 and i < n:
        add = min(b - a, D)
        data[i] += add
        D -= add
        i += 1
    max_ss = sum(x * x for x in data)
    return min_ss, max_ss


def sprite_consistent(mean: float, sd: float, n: int, vmin: int, vmax: int,
                      sd_decimals: int | None = None) -> bool:
    """SPRITE 族：有界整数量表上，报告的 (mean, sd) 是否可能重建出数据。

    n 个 [vmin,vmax] 整数、和=round(mean·n) 时，方差有可证明的上下界；
    报告 SD 的舍入区间与可达方差区间不相交 → 不可能（如均值贴近边界却给大 SD）。
    """
    if mean < vmin or mean > vmax:
        return False
    X = round(mean * n)
    if not (vmin * n <= X <= vmax * n):
        return False
    sdd = sd_decimals if sd_decimals is not None else _decimals(sd)
    min_ss, max_ss = _ss_bounds(X, n, vmin, vmax)
    min_var = (min_ss - X * X / n) / (n - 1)
    max_var = (max_ss - X * X / n) / (n - 1)
    half = 0.5 * 10 ** (-sdd)
    var_lo = max(0.0, sd - half) ** 2
    var_hi = (sd + half) ** 2
    return var_hi >= min_var - 1e-9 and var_lo <= max_var + 1e-9


def sprite_check(mean_sd_scale_pairs) -> list[Finding]:
    """mean_sd_scale_pairs: 可迭代 (mean, sd, n, vmin, vmax, page, label)。"""
    findings: list[Finding] = []
    for item in mean_sd_scale_pairs:
        mean, sd, n, vmin, vmax, page, label = item[:7]
        sdd = item[7] if len(item) > 7 else None
        if sprite_consistent(mean, sd, n, vmin, vmax, sdd):
            continue
        findings.append(Finding(
            detector="sprite", category=Category.STATS, severity=0.65,
            title="均值/标准差在该量表范围内无法重建（疑似不可能）",
            description=(
                f"{label}：在 [{vmin},{vmax}] 量表、n={n} 下，报告均值 {mean} 与标准差 {sd} "
                "落在可达方差区间之外，数学上无法由任何整数数据重建（常见于均值贴近边界却报大 SD）。"
                "疑似统计量编造或笔误，待核查。"
            ),
            evidence={"mean": mean, "sd": sd, "n": n, "scale": [vmin, vmax]},
            locations=[f"p.{page}" if page else label],
        ))
    return findings


def grim_check(mean_n_pairs) -> list[Finding]:
    """mean_n_pairs: 可迭代的 (mean, n, n_items, page, label) —— 对每个跑 GRIM。"""
    findings: list[Finding] = []
    for item in mean_n_pairs:
        mean, n, n_items, page, label = item[:5]
        decimals = item[5] if len(item) > 5 else None    # 原始字符串小数位（可靠）
        if grim_consistent(mean, n, n_items, decimals):
            continue
        findings.append(Finding(
            detector="grim",
            category=Category.STATS,
            severity=0.7,
            title="均值不满足 GRIM 一致性（整数数据下不可能取到该值）",
            description=(
                f"{label}：报告均值 {mean} 在样本量 n={n}"
                f"{'、每人 '+str(n_items)+' 题' if n_items>1 else ''} 的整数数据下数学上无法取到"
                f"（可行均值只能是整数/{n*n_items}）。疑似数据或样本量有误，待核查。"
            ),
            evidence={"mean": mean, "n": n, "n_items": n_items},
            locations=[f"p.{page}" if page else label],
        ))
    return findings


# ============================ 本福特定律 ============================

def _leading_digit(x: float) -> int | None:
    x = abs(x)
    if x == 0 or math.isnan(x) or math.isinf(x):
        return None
    while x < 1:
        x *= 10
    while x >= 10:
        x /= 10
    return int(x)


def benford_check(numbers, min_n: int = 60, min_orders: float = 2.0) -> Finding | None:
    """首位数分布 vs 本福特期望，卡方检验。

    重要前提：本福特只适用于**自然产生、跨多个数量级**的数据。数据不跨数量级时
    （如都是几十的百分比）即便诚实也不服从，故先 gate 数量级跨度再检验，且置信度偏保守。
    """
    digs = [d for x in numbers if (d := _leading_digit(x)) is not None]
    if len(digs) < min_n:
        return None
    positives = [abs(x) for x in numbers if x and not math.isnan(x)]
    span = math.log10(max(positives)) - math.log10(min(positives)) if positives else 0
    if span < min_orders:
        return None  # 不跨数量级，本福特不适用，跳过避免误报

    n = len(digs)
    obs = Counter(digs)
    expected = {d: n * math.log10(1 + 1 / d) for d in range(1, 10)}
    chi2 = sum((obs.get(d, 0) - expected[d]) ** 2 / expected[d] for d in range(1, 10))
    p = float(sps.chi2.sf(chi2, df=8))
    if p >= 0.01:
        return None

    obs_freq = {d: round(obs.get(d, 0) / n, 3) for d in range(1, 10)}
    sev = 0.6 if p < 0.001 else 0.45
    return Finding(
        detector="benford",
        category=Category.STATS,
        severity=sev,
        title="数字首位分布显著偏离本福特定律",
        description=(
            f"对 {n} 个数字做首位数分布卡方检验，χ²={chi2:.1f}, p={p:.2g}，显著偏离本福特期望。"
            " 自然数据通常服从本福特律，偏离*可能*提示人为编造，但也可能因数据性质本身不适用，"
            "属辅助线索，需结合其他证据人工核查。"
        ),
        evidence={"n": n, "chi2": round(chi2, 2), "p": round(p, 5),
                  "observed_freq": obs_freq,
                  "benford_expected": {d: round(math.log10(1 + 1 / d), 3) for d in range(1, 10)}},
        locations=[],
    )


# ============================ 重复数据点 ============================

def duplicate_values_check(tables, min_run: int = 4) -> list[Finding]:
    """检测表格里整行数字被复制粘贴（编造数据的常见痕迹）。

    把每张表的每一行的数值序列做指纹，若两行数值序列完全相同且长度 ≥ min_run，
    判为可疑重复行。
    """
    findings: list[Finding] = []
    for t in tables:
        seen: dict[tuple, int] = {}
        for ri, row in enumerate(t.rows):
            nums = tuple(_to_float(c) for c in row)
            nums = tuple(v for v in nums if v is not None)
            if len(nums) < min_run:
                continue
            if len(set(nums)) == 1:
                continue  # 整行同值(如全 0/全 NA 占位)不算复制粘贴造假
            if nums in seen:
                findings.append(Finding(
                    detector="dup_values",
                    category=Category.STATS,
                    severity=0.55,
                    title="表格内出现完全相同的数据行（疑似复制粘贴）",
                    description=(
                        f"{t.label()} 第 {seen[nums]+1} 行与第 {ri+1} 行的 {len(nums)} 个数值完全相同："
                        f"{list(nums)[:8]}{'…' if len(nums)>8 else ''}。整行数字一字不差地重复，"
                        "在真实实验数据里极少见，疑似复制粘贴造数，待核查。"
                    ),
                    evidence={"table": t.label(), "row_a": seen[nums] + 1, "row_b": ri + 1,
                              "values": list(nums)},
                    locations=[t.label()],
                ))
            else:
                seen[nums] = ri
    return findings


def _to_float(c):
    try:
        return float(str(c).replace(",", ""))
    except (ValueError, TypeError):
        return None


# ===================== 耿同学三大统计招牌 =====================

_NUMSTR = re.compile(r"^[-+]?\d+(?:\.\d+)?$")


def _numeric_strings(tables) -> list[str]:
    """收集表格里所有"纯数字"单元格的**原始字符串**（保留位数信息）。"""
    out = []
    for t in tables:
        for row in t.rows:
            for c in row:
                s = str(c).strip().replace(",", "")
                if _NUMSTR.match(s):
                    out.append(s)
    return out


def terminal_digit_check(tables, min_n: int = 50) -> Finding | None:
    """末位数字均匀性（耿同学"看小数点后几位分布"）。

    只看**有小数的数**的最后一位小数——测量值的最末位应近似均匀；编造数据
    常在末位露馅。整数末位易因取整偏向 0/5，故排除。
    """
    digits = []
    for s in _numeric_strings(tables):
        if "." in s:
            frac = s.split(".")[-1]
            # 跳过纯尾零的"整数写成小数"(如 12.0 / 100.00)——否则末位被拉向 0 造成误报
            if frac and frac.strip("0") != "" and frac[-1].isdigit():
                digits.append(frac[-1])
    if len(digits) < min_n:
        return None
    n = len(digits)
    c = Counter(digits)
    exp = n / 10.0
    chi2 = sum((c.get(str(d), 0) - exp) ** 2 / exp for d in range(10))
    p = float(sps.chi2.sf(chi2, df=9))
    if p >= 0.001:
        return None
    freq = {str(d): round(c.get(str(d), 0) / n, 3) for d in range(10)}
    return Finding(
        detector="terminal_digit", category=Category.STATS, severity=0.5,
        title="数据末位数字分布显著非均匀",
        description=(
            f"对 {n} 个小数的最末位做均匀性卡方检验，χ²={chi2:.1f}, p={p:.2g}，"
            "显著偏离均匀分布。测量值末位通常应近似均匀，异常集中*可能*提示人为编造或"
            "过度取整，属辅助线索，待人工核查。"
        ),
        evidence={"n": n, "chi2": round(chi2, 2), "p": round(p, 6), "last_digit_freq": freq},
        locations=[],
    )


def _numeric_columns(table) -> list[list[float]]:
    """按列取数值序列（行序保留），用于等差数列检测。"""
    cols = []
    ncol = max((len(r) for r in table.rows), default=0)
    for ci in range(ncol):
        vals = [_to_float(r[ci]) for r in table.rows if ci < len(r)]
        nums = [v for v in vals if v is not None]
        if len(nums) >= 5:
            cols.append(nums)
    return cols


def arithmetic_sequence_check(tables, min_run: int = 5, rel_tol: float = 1e-3) -> list[Finding]:
    """数据太规整：列里出现"公差恒定的等差数列"（如苏佳灿案公差 0.43）。

    真实测量值几乎不可能连续 5+ 个排成完美等差。排除明显是序号的整数列（公差=1）。
    """
    findings = []
    for t in tables:
        for col in _numeric_columns(t):
            i = 0
            while i < len(col) - 1:
                diff = col[i + 1] - col[i]
                j = i + 1
                while j < len(col) - 1 and abs((col[j + 1] - col[j]) - diff) <= max(1e-9, rel_tol * abs(diff)):
                    j += 1
                run = j - i + 1
                trivial_index = abs(diff - 1.0) < 1e-9 and all(float(x).is_integer() for x in col[i:j + 1])
                if run >= min_run and abs(diff) > 1e-9 and not trivial_index:
                    seq = [round(x, 4) for x in col[i:j + 1]]
                    findings.append(Finding(
                        detector="too_regular", category=Category.STATS, severity=0.7,
                        title="数据呈完美等差数列（疑似编造）",
                        description=(
                            f"{t.label()} 某列出现 {run} 个连续值构成公差≈{diff:.4g} 的完美等差数列："
                            f"{seq}。真实实验数据极难如此规整，'唯一的解释往往是编造'（耿同学语），待核查。"
                        ),
                        evidence={"table": t.label(), "common_diff": round(diff, 6),
                                  "run_length": run, "values": seq},
                        locations=[t.label()],
                    ))
                    i = j
                else:
                    i += 1
    return findings


def _aligned_columns(table):
    """按位置取列（行对齐，非数值处为 None），用于跨列比较。"""
    rows = table.rows
    ncol = max((len(r) for r in rows), default=0)
    return [[_to_float(r[ci]) if ci < len(r) else None for r in rows] for ci in range(ncol)]


def column_offset_check(tables, min_len: int = 5) -> list[Finding]:
    """跨列固定差值（苏佳灿案"两列数据固定相差0.3"）。

    两列在每一行的差值恒为同一非零常数 → 疑似把一列加常数伪造成另一列。
    排除"相等列"(归重复/小数巧合)和"序号列 vs 序号+1"(整数差=1)。
    """
    findings = []
    for t in tables:
        cols = _aligned_columns(t)
        usable = [(ci, c) for ci, c in enumerate(cols) if sum(v is not None for v in c) >= min_len]
        for (i, ca), (j, cb) in combinations(usable, 2):
            pairs = [(a, b) for a, b in zip(ca, cb) if a is not None and b is not None]
            if len(pairs) < min_len:
                continue
            diffs = [b - a for a, b in pairs]
            d0 = diffs[0]
            if abs(d0) < 1e-9:
                continue  # 相等列，另有检查管
            all_int = all(float(a).is_integer() and float(b).is_integer() for a, b in pairs)
            if all_int and abs(abs(d0) - 1.0) < 1e-9:
                continue  # 序号列 vs 序号+1
            if all(abs(d - d0) <= max(1e-9, 1e-3 * abs(d0)) for d in diffs):
                findings.append(Finding(
                    detector="column_offset", category=Category.STATS, severity=0.66,
                    title="两列数据呈固定差值（疑似一列+常数伪造另一列）",
                    description=(
                        f"{t.label()} 第 {i+1} 列与第 {j+1} 列在 {len(pairs)} 行上差值恒为 {d0:.4g}。"
                        "真实独立测量的两组数据极难处处相差同一常数，疑似把一列加常数造出另一列"
                        "（如苏佳灿案「两列固定相差0.3」），待核查。"
                    ),
                    evidence={"table": t.label(), "col_a": i + 1, "col_b": j + 1,
                              "constant_diff": round(d0, 6), "n_rows": len(pairs)},
                    locations=[t.label()],
                ))
    return findings


def decimal_coincidence_check(tables, min_n: int = 20, conc_thresh: float = 0.35) -> list[Finding]:
    """跨记录小数巧合（南开案"小数点后两位完全一致"）。

    取有 ≥2 位小数的数，看其末两位小数的分布。本应五花八门；若极度集中（少数
    几个尾数霸占大半），提示数据被复制/编造。
    """
    findings = []
    for t in tables:
        tails = []
        for s in _numeric_strings([t]):
            if "." in s:
                frac = s.split(".")[-1]
                if len(frac) >= 2:
                    tails.append(frac[-2:])
        if len(tails) < min_n:
            continue
        c = Counter(tails)
        top, cnt = c.most_common(1)[0]
        share = cnt / len(tails)
        distinct = len(c)
        if share >= conc_thresh and distinct < len(tails) / 2:
            findings.append(Finding(
                detector="decimal_coincidence", category=Category.STATS, severity=0.55,
                title="数据末两位小数异常集中（疑似抄数/编造）",
                description=(
                    f"{t.label()} 的 {len(tails)} 个小数中，末两位为「{top}」的占 {share:.0%}"
                    f"（仅 {distinct} 种不同尾数）。独立测量值的小数尾数本应分散，"
                    "异常一致提示复制或编造，待核查。"
                ),
                evidence={"table": t.label(), "n": len(tails), "top_tail": top,
                          "top_share": round(share, 3), "distinct_tails": distinct},
                locations=[t.label()],
            ))
    return findings


# ============================ 汇总入口 ============================

def run_stats(tables, reported_stats, mean_sd_n=None) -> list[Finding]:
    """跑全部统计检测，返回汇总 Finding 列表。

    mean_sd_n: extract_mean_sd_n 抽出的 (均值,SD,n,量表) 三元组（仅整数/量表语境）。
    据此跑 GRIM/GRIMMER/SPRITE——它们只对整数数据成立，已在提取层用语境闸门把关。
    """
    findings: list[Finding] = []
    findings += statcheck(reported_stats)
    if mean_sd_n:
        findings += grim_check([(x.mean, x.n, 1, x.page, x.context, x.mean_decimals)
                                for x in mean_sd_n])
        findings += grimmer_check([(x.mean, x.sd, x.n, 1, x.page, x.context,
                                    x.mean_decimals, x.sd_decimals)
                                   for x in mean_sd_n if x.sd is not None])
        findings += sprite_check([(x.mean, x.sd, x.n, x.scale[0], x.scale[1], x.page,
                                   x.context, x.sd_decimals)
                                  for x in mean_sd_n if x.sd is not None and x.scale])
    all_numbers = [v for t in tables for v in t.numbers]
    if all_numbers:
        bf = benford_check(all_numbers)
        if bf:
            findings.append(bf)
    findings += duplicate_values_check(tables)
    td = terminal_digit_check(tables)
    if td:
        findings.append(td)
    findings += arithmetic_sequence_check(tables)
    findings += decimal_coincidence_check(tables)
    findings += column_offset_check(tables)
    findings.sort(key=lambda f: f.severity, reverse=True)
    return findings
