"""统一的"可疑发现"数据结构。

所有检测器都产出 Finding，报告层只认 Finding，这样新增检测器不用动报告代码。

设计原则（见 README）：检测器只给"可疑线索 + 证据 + 置信度"，
措辞用"疑似/待核查"，不下"造假"定论——是否构成学术不端由人判断。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class Category(str, Enum):
    IMAGE_DUP = "image_dup"        # 图片查重：一图多用 / 局部复用 / 旋转翻转复用
    IMAGE_MANIP = "image_manip"    # 单图内复制粘贴篡改 (copy-move forgery)
    STATS = "stats"                # 统计异常：本福特 / GRIM / statcheck / 重复值
    CHART = "chart"                # 图表自洽性：柱状图条高 vs 报告数值
    CROSS_PAPER = "cross_paper"    # 跨论文/跨作者盗图
    TEXT = "text"                  # 文本/元数据：ChatGPT 指纹 / tortured phrases / 撤稿引用


@dataclass
class Finding:
    """一条可疑发现。

    severity: 0~1 的置信度（不是"造假概率"，是"这条线索有多硬"）。
    evidence: 自由字典，放证据材料（图片路径、坐标、数值、匹配点数等）。
              报告层按 category 决定怎么渲染。
    locations: 人类可读的定位（如 "p.3 Fig.2" / "Table 1"）。
    """
    detector: str
    category: Category
    severity: float
    title: str
    description: str
    evidence: dict[str, Any] = field(default_factory=dict)
    locations: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.severity = max(0.0, min(1.0, float(self.severity)))
        if isinstance(self.category, str):
            self.category = Category(self.category)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["category"] = self.category.value
        return d


def severity_label(severity: float) -> str:
    """把置信度数值映射成中文档位，供报告显示。"""
    if severity >= 0.8:
        return "高度可疑"
    if severity >= 0.55:
        return "可疑"
    if severity >= 0.3:
        return "待核查"
    return "轻微线索"
