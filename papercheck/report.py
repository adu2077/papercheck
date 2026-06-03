"""报告层：把 Finding 列表渲染成自包含的 HTML 证据报告。

图片用 base64 内嵌（生成缩略图控制体积），报告单文件即可分享。
按类别分组、按置信度排序，措辞统一"疑似/待核查"。
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

from jinja2 import Environment, BaseLoader
from PIL import Image

from papercheck.findings import Finding, Category, severity_label


_CATEGORY_CN = {
    Category.IMAGE_DUP: "① 图片复用",
    Category.IMAGE_MANIP: "① 单图篡改 (copy-move)",
    Category.STATS: "② 统计异常",
    Category.CHART: "③ 图表自洽性",
    Category.CROSS_PAPER: "跨论文盗图",
    Category.TEXT: "④ 文本/元数据",
}

# 每条 finding 里可能出现的图片证据键 → 展示标题
_IMG_KEYS = [
    ("match_viz", "特征匹配连线（绿线＝几何一致对应点）"),
    ("copymove_viz", "复制粘贴匹配（红线连接源/目标）"),
    ("image_a", "图 A"),
    ("image_b", "图 B"),
    ("image", "图"),
]


def _thumb_data_uri(path: str, max_side: int = 460) -> str | None:
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((max_side, max_side))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=82)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return None


def _finding_images(f: Finding) -> list[dict]:
    out, used = [], set()
    for key, caption in _IMG_KEYS:
        p = f.evidence.get(key)
        if p and p not in used and Path(p).exists():
            uri = _thumb_data_uri(p)
            if uri:
                out.append({"caption": caption, "uri": uri})
                used.add(p)
    return out


def _evidence_rows(f: Finding) -> list[tuple]:
    """非图片证据 → (键, 值) 行，给统计类等展示。"""
    skip = {k for k, _ in _IMG_KEYS} | {"sha1"}
    rows = []
    for k, v in f.evidence.items():
        if k in skip:
            continue
        if isinstance(v, float):
            v = f"{v:.4g}"
        rows.append((k, v))
    return rows


_TEMPLATE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>papercheck 报告 · {{ paper_id }}</title>
<style>
 body{font-family:-apple-system,"PingFang SC",Helvetica,Arial,sans-serif;margin:0;background:#f5f6f8;color:#1c2733}
 .wrap{max-width:980px;margin:0 auto;padding:28px 20px 80px}
 h1{font-size:22px;margin:0 0 4px} .sub{color:#6b7785;font-size:13px;margin-bottom:20px}
 .summary{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:24px}
 .chip{background:#fff;border:1px solid #e2e6ea;border-radius:10px;padding:10px 14px;font-size:13px}
 .chip b{font-size:18px;display:block}
 .disclaimer{background:#fff7e6;border:1px solid #ffe0a3;border-radius:8px;padding:10px 14px;font-size:12.5px;color:#8a6d3b;margin-bottom:24px}
 h2{font-size:16px;margin:28px 0 12px;border-left:4px solid #4a7;padding-left:8px}
 .card{background:#fff;border:1px solid #e2e6ea;border-radius:12px;padding:16px 18px;margin-bottom:14px;box-shadow:0 1px 2px rgba(0,0,0,.03)}
 .card-head{display:flex;align-items:center;gap:10px;margin-bottom:6px}
 .sev{font-size:12px;font-weight:700;padding:2px 9px;border-radius:20px;color:#fff;white-space:nowrap}
 .s-high{background:#d9534f}.s-mid{background:#e8923b}.s-low{background:#888}
 .title{font-weight:600;font-size:15px}
 .loc{color:#8b97a3;font-size:12px;margin-left:auto}
 .desc{font-size:13.5px;line-height:1.65;color:#384956;margin:6px 0 12px}
 .imgs{display:flex;gap:12px;flex-wrap:wrap}
 .imgbox{flex:0 0 auto;max-width:300px} .imgbox img{max-width:100%;border:1px solid #dde;border-radius:6px;display:block}
 .imgcap{font-size:11.5px;color:#8b97a3;margin-top:4px}
 table.ev{font-size:12px;border-collapse:collapse;margin-top:4px}
 table.ev td{border:1px solid #eee;padding:3px 8px;color:#566} table.ev td:first-child{color:#99a;font-family:monospace}
 .empty{background:#fff;border:1px dashed #cdd;border-radius:12px;padding:40px;text-align:center;color:#8b97a3}
</style></head><body><div class="wrap">
 <h1>📄 papercheck 程序化打假报告</h1>
 <div class="sub">论文：<b>{{ paper_id }}</b> · 共 {{ findings|length }} 条可疑线索</div>
 <div class="summary">
  <div class="chip"><b>{{ findings|length }}</b>线索总数</div>
  <div class="chip"><b>{{ n_high }}</b>高度可疑</div>
  {% for cat, items in groups %}<div class="chip"><b>{{ items|length }}</b>{{ cat }}</div>{% endfor %}
 </div>
 <div class="disclaimer">⚠️ 本报告由程序自动比对生成，仅提供<b>可疑线索 + 证据 + 置信度</b>，不构成"造假"定论。
  所有结论均为<b>疑似 / 待核查</b>，是否构成学术不端须由人工结合论文内容判断。</div>

 {% if not findings %}<div class="empty">未发现可疑线索 🎉</div>{% endif %}
 {% for cat, items in groups %}
  <h2>{{ cat }}（{{ items|length }}）</h2>
  {% for f in items %}
  <div class="card">
   <div class="card-head">
    <span class="sev {{ f.sev_class }}">{{ f.sev_label }} · {{ '%.0f'|format(f.severity*100) }}%</span>
    <span class="title">{{ f.title }}</span>
    {% if f.locations %}<span class="loc">{{ f.locations|join(' · ') }}</span>{% endif %}
   </div>
   <div class="desc">{{ f.description }}</div>
   {% if f.images %}<div class="imgs">{% for im in f.images %}
     <div class="imgbox"><img src="{{ im.uri }}"><div class="imgcap">{{ im.caption }}</div></div>
   {% endfor %}</div>{% endif %}
   {% if f.ev_rows %}<table class="ev">{% for k,v in f.ev_rows %}<tr><td>{{ k }}</td><td>{{ v }}</td></tr>{% endfor %}</table>{% endif %}
  </div>
  {% endfor %}
 {% endfor %}
</div></body></html>"""


def _sev_class(sev: float) -> str:
    return "s-high" if sev >= 0.7 else ("s-mid" if sev >= 0.45 else "s-low")


def render_report(paper_id: str, findings: list[Finding], out_path: str | Path) -> Path:
    findings = sorted(findings, key=lambda f: f.severity, reverse=True)

    # 按类别分组（保持出现顺序）
    groups: dict[Category, list] = {}
    for f in findings:
        groups.setdefault(f.category, []).append(f)

    def _prep(f: Finding) -> dict:
        return {
            "title": f.title, "description": f.description, "severity": f.severity,
            "locations": f.locations, "sev_label": severity_label(f.severity),
            "sev_class": _sev_class(f.severity),
            "images": _finding_images(f), "ev_rows": _evidence_rows(f),
        }

    rendered_groups = [
        (_CATEGORY_CN.get(cat, str(cat)), [_prep(f) for f in items])
        for cat, items in groups.items()
    ]

    env = Environment(loader=BaseLoader(), autoescape=True)
    html = env.from_string(_TEMPLATE).render(
        paper_id=paper_id,
        findings=findings,
        groups=rendered_groups,
        n_high=sum(1 for f in findings if f.severity >= 0.7),
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path
