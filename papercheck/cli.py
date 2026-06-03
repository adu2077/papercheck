"""命令行入口：analyze / index / cross。

  python -m papercheck analyze data/paper.pdf -o output/
  python -m papercheck index data/*.pdf --db fingerprints.db
  python -m papercheck cross --db fingerprints.db -o output/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from papercheck.pipeline import analyze_paper, index_papers, cross_report

_IMG_KEYS = ("image_a", "image_b", "image", "match_viz", "copymove_viz")


def _finding_brief(f) -> dict:
    """给调用方 agent 看的精简候选：含证据图路径，便于 agent 自己 Read 图来裁决。"""
    images = []
    for k in _IMG_KEYS:
        p = f.evidence.get(k)
        if p and p not in images and Path(p).exists():
            images.append(p)
    return {
        "severity": round(f.severity, 3),
        "category": f.category.value,
        "detector": f.detector,
        "title": f.title,
        "locations": f.locations,
        "images": images,            # 证据图绝对/相对路径，agent 直接 Read 来判
        "description": f.description,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="papercheck",
        description="程序化论文打假：图片查重/篡改 + 统计异常 + 跨论文盗图比对。",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pa = sub.add_parser("analyze", help="分析单篇论文并出报告")
    pa.add_argument("pdf", help="输入 PDF 路径")
    pa.add_argument("-o", "--out", default="output", help="输出目录（默认 output/）")
    pa.add_argument("--db", default=None, help="可选：同时把图片指纹入库")
    pa.add_argument("--json", action="store_true", dest="as_json",
                    help="输出机读候选 JSON（含证据图路径），供调起本工具的 agent 自行看图裁决")
    pa.add_argument("--judge", action="store_true",
                    help="[headless/无agent场景] 外接 LLM 裁决候选；被 agent 调用时不需要——调用方自己判即可")
    pa.add_argument("--llm-review", action="store_true", help="[headless] LLM 整图通览找破绽（每图一次调用，较贵）")
    pa.add_argument("--citations", action="store_true", help="联网查引用是否含撤稿论文（OpenAlex）")

    pi = sub.add_parser("index", help="批量把多篇论文图片入指纹库")
    pi.add_argument("pdfs", nargs="+", help="一个或多个 PDF")
    pi.add_argument("-o", "--out", default="output")
    pi.add_argument("--db", default="fingerprints.db")

    pc = sub.add_parser("cross", help="跨论文比对出报告")
    pc.add_argument("--db", default="fingerprints.db")
    pc.add_argument("-o", "--out", default="output")
    pc.add_argument("--fast", action="store_true",
                    help="大库提速：只用 phash 粗筛（会漏纯局部复用）")

    args = p.parse_args(argv)

    if args.cmd == "analyze":
        if not Path(args.pdf).exists():
            print(f"找不到文件: {args.pdf}", file=sys.stderr)
            return 2
        try:
            r = analyze_paper(args.pdf, args.out, db_path=args.db,
                              judge=args.judge, llm_review=args.llm_review,
                              check_citations=args.citations,
                              log=(lambda *a: None) if args.as_json else print)
        except Exception as e:
            if args.as_json:
                print(json.dumps({"error": str(e)}, ensure_ascii=False))
            else:
                print(f"分析失败（PDF 可能损坏/加密/非法）: {e}", file=sys.stderr)
            return 2
        if args.as_json:
            fs = sorted(r["findings"], key=lambda x: x.severity, reverse=True)
            print(json.dumps({
                "paper_id": r["paper_id"], "report": r["report"],
                "n_findings": len(fs),
                "note": "这些是程序产出的候选线索（疑似/待核查）。请逐条 Read images 里的证据图，"
                        "自己判断真复用/良性/不确定，切勿直接当定论。",
                "findings": [_finding_brief(f) for f in fs],
            }, ensure_ascii=False, indent=2))
        else:
            print(f"\n✓ {r['paper_id']}：{len(r['findings'])} 条线索 → {r['report']}")
        return 0

    if args.cmd == "index":
        index_papers(args.pdfs, args.out, args.db)
        return 0

    if args.cmd == "cross":
        r = cross_report(args.db, args.out, thorough=not args.fast)
        print(f"\n✓ {r['n_papers']} 篇论文，{len(r['findings'])} 条跨论文线索 → {r['report']}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
