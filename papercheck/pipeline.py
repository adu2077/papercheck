"""编排：把 extract → detect → (index) → report 串起来。

- analyze_paper：单篇论文全套检测 + 出报告
- index_papers ：批量把多篇论文的图片指纹入库
- cross_report ：跨论文比对 + 出报告
"""
from __future__ import annotations

from pathlib import Path

from papercheck.findings import Finding, Category
from papercheck.extract.pdf_images import extract_images
from papercheck.extract.pdf_tables import extract_tables
from papercheck.extract.pdf_text import extract_pages, extract_reported_stats, extract_mean_sd_n
from papercheck.extract.panels import panels_from_images
from papercheck.detect.image_dup import find_duplicate_pairs
from papercheck.detect.image_manip import find_copy_move
from papercheck.detect.splice import find_splice_seams
from papercheck.detect.chart_consistency import check_error_bars
from papercheck.detect.stats_anomaly import run_stats
from papercheck.detect.text_forensics import run_text_checks
from papercheck.detect.citation_check import check_retracted_citations
from papercheck.detect.llm_judge import judge_findings, holistic_review
from papercheck.report import render_report
from papercheck.index import fingerprint_db as fdb

# 图像类线索（交给 LLM 判官裁决的类别）
_IMAGE_CATS = {Category.IMAGE_DUP, Category.IMAGE_MANIP, Category.CHART, Category.CROSS_PAPER}


def analyze_paper(
    pdf_path: str | Path,
    out_dir: str | Path = "output",
    paper_id: str | None = None,
    db_path: str | Path | None = None,
    judge: bool = False,
    llm_review: bool = False,
    check_citations: bool = False,
    provider=None,
    log=print,
) -> dict:
    """单篇论文：抽取 → 图像/统计检测 →（可选 LLM 判官/通览）→ HTML 报告。

    judge=True：用 LLM 对图像类候选做语义裁决，调整严重度（降良性/升可疑）。
    llm_review=True：让 LLM 整图通览，找检测器没编进去的破绽（每图一次调用，较贵）。
    provider：自定义 LLM 后端；None 时用 get_provider()（厂商中立：环境命令模板 > 自动探测已装 CLI）。
    """
    pdf_path = Path(pdf_path)
    paper_id = paper_id or pdf_path.stem
    out_dir = Path(out_dir)
    viz_dir = out_dir / paper_id / "viz"

    log(f"[{paper_id}] 抽取图片/表格/文本…")
    images = extract_images(pdf_path, out_dir, paper_id)
    tables = extract_tables(pdf_path, paper_id)
    pages = extract_pages(pdf_path)
    stats = extract_reported_stats(pages)
    mean_sd_n = extract_mean_sd_n(pages)

    # 切面板 + 分类：只对印迹/照片面板做图像取证，图表面板跳过（避免图表结构误报）
    panels = panels_from_images(images, out_dir)
    photo = [p for p in panels if p.kind == "photo"]
    chart = [p for p in panels if p.kind == "chart"]
    log(f"[{paper_id}] 图片 {len(images)} → 面板 {len(panels)}"
        f"（印迹/照片 {len(photo)} 查 · 图表 {len(chart)} 跳过）· 表格 {len(tables)} · 统计量 {len(stats)}")

    findings: list[Finding] = []
    log(f"[{paper_id}] 跑图片查重（印迹面板两两比对）…")
    findings += find_duplicate_pairs(photo, viz_dir=viz_dir)
    log(f"[{paper_id}] 跑单图篡改检测…")
    for p in photo:
        f = find_copy_move(p, viz_dir=viz_dir)
        if f:
            findings.append(f)
    log(f"[{paper_id}] 跑拼接缝检测…")
    for p in photo:
        sf = find_splice_seams(p)
        if sf:
            findings.append(sf)
    for p in chart:                 # 误差棒异常只在图表面板上查
        ef = check_error_bars(p)
        if ef:
            findings.append(ef)
    log(f"[{paper_id}] 跑统计异常检测…（含 GRIM/GRIMMER/SPRITE，限整数/量表语境）")
    findings += run_stats(tables, stats, mean_sd_n=mean_sd_n)
    log(f"[{paper_id}] 跑文本/元数据检测…")
    findings += run_text_checks(pages)
    if check_citations:
        log(f"[{paper_id}] 查引用是否含撤稿论文（联网）…")
        findings += check_retracted_citations(pages, log=log)

    if llm_review:
        log(f"[{paper_id}] LLM 整图通览…")
        findings += holistic_review(photo, provider=provider, log=log)
    if judge:
        log(f"[{paper_id}] LLM 判官裁决图像候选…")
        judge_findings(findings, provider=provider, only_categories=_IMAGE_CATS, log=log)

    findings.sort(key=lambda f: f.severity, reverse=True)
    report_path = render_report(paper_id, findings, out_dir / f"{paper_id}.html")
    log(f"[{paper_id}] 发现 {len(findings)} 条可疑线索 → 报告 {report_path}")

    if db_path:
        con = fdb.connect(db_path)
        fdb.add_images(con, photo)        # 跨论文比对也只用印迹/照片面板
        con.close()
        log(f"[{paper_id}] 已入指纹库 {db_path}（{len(photo)} 个印迹面板）")

    return {"paper_id": paper_id, "findings": findings, "report": str(report_path),
            "n_images": len(images), "n_panels": len(panels),
            "n_photo": len(photo), "n_tables": len(tables)}


def index_papers(pdf_paths, out_dir="output", db_path="fingerprints.db", log=print) -> int:
    """批量把多篇论文的图片入指纹库（不出单篇报告，只为跨论文比对铺路）。"""
    con = fdb.connect(db_path)
    total = 0
    for p in pdf_paths:
        pid = Path(p).stem
        imgs = extract_images(p, out_dir, pid)
        photo = [pn for pn in panels_from_images(imgs, out_dir) if pn.kind == "photo"]
        total += fdb.add_images(con, photo)
        log(f"  + {pid}: {len(imgs)} 张图 → {len(photo)} 个印迹面板入库")
    n_papers = fdb.paper_count(con)
    con.close()
    log(f"指纹库 {db_path}：{n_papers} 篇论文 · {total} 张图")
    return total


def cross_report(db_path="fingerprints.db", out_dir="output", thorough=True, log=print) -> dict:
    """跨论文比对，出 HTML 报告。"""
    con = fdb.connect(db_path)
    n_papers = fdb.paper_count(con)
    log(f"跨论文比对：{n_papers} 篇论文…")
    findings = fdb.find_cross_paper_dups(con, thorough=thorough)
    con.close()
    report_path = render_report("跨论文盗图比对", findings, Path(out_dir) / "cross_report.html")
    log(f"发现 {len(findings)} 条跨论文疑似复用 → 报告 {report_path}")
    return {"findings": findings, "report": str(report_path), "n_papers": n_papers}
