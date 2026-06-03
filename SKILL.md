---
name: papercheck
description: 「耿同学」论文打假 skill——审查学术论文 PDF 的学术不端线索：图片复用/PS篡改/拼接、数据造假(等差数列/小数巧合/末位异常)、统计自相矛盾(p值/GRIM)、图表不一致、论文工厂特征(ChatGPT指纹/tortured phrases/引用撤稿)，并可跨论文查盗图。当用户想"用耿同学查/打假/审查一篇论文""查图片是否重复或PS""数据是不是编的""有没有学术不端"时使用。Detect research-misconduct signals in a paper PDF: image duplication/manipulation/splicing, data fabrication, statistical inconsistencies, chart mismatches, paper-mill text signals; cross-paper image reuse.
---

# papercheck —— 论文程序化打假

复现耿同学/Elisabeth Bik 式"靠形式化比对抓造假，不需读懂内容"的流水线：
PDF → 抽图/表/统计量 → 多类检测器 → （你这个 agent 看证据图语义裁决）→ HTML 铁证报告。

## ⚠️ 用前必读（每次给用户结论时都要带上）

本工具只产出**"可疑线索 + 证据 + 置信度"，措辞一律"疑似/待核查"，绝不下"造假"定论**。
程序层是**高召回低精度**的候选生成器——图表、上样对照、周期纹理常有误报。
**务必**：① 把发现当线索不当判决 ② **你自己 Read 证据图逐条裁决**（这就是判官，见下方流程），别把程序候选直接抛给用户 ③ 提醒用户人工复核、勿据此公开指控他人。

## 怎么用（核心：程序出候选 → **你这个 agent 亲自裁决**）

> 关键：**判官就是调起本 skill 的你**——一个会看图、能推理的大模型。程序只做廉价确定的
> "找候选"，**语义裁决你自己来，不要再去外接另一个 LLM**（那是 headless 无 agent 时才用的）。

脚本在本 skill 目录的 `scripts/` 下，会自行定位。命令里的 `<skill-dir>` 替换为**本 skill 基目录**
（运行时调起 skill 时会给出，如开头 "Base directory for this skill: …"），不依赖任何厂商变量。

**第一步：安装（首次/换机，幂等）**
```bash
bash "<skill-dir>/scripts/setup.sh"
```

**第二步：跑程序拿候选（机读 JSON，不要加 --judge）**
```bash
bash "<skill-dir>/scripts/run.sh" analyze "<论文.pdf>" -o "<输出目录>" --json
```
输出一段 JSON：每条候选含 `severity / category / title / locations / images（证据图路径）/ description`，
末尾有 HTML 报告路径。

**第三步：你（agent）亲自裁决** ⬅ 这一步就是"判官"，由你做
- 对 JSON 里**置信度高的候选**，逐条 **Read 它的 `images`**（并排原图 / 匹配连线图 / copy-move 连线图），
  用你的视觉+推理判断：**真复用 / 良性 / 不确定**，并给一句理由。
- 已知易误报（务必识别并降权）：共用坐标轴、loading control 自相似、周期纹理、拼接缝落在文字/标签上。
- 程序层是高召回低精度，很多候选是良性巧合——靠你这一步去伪存真。

**第四步：回复用户**
1. 先说免责（线索非定论、勿据此公开指控）。
2. 给出你裁决后的**可信短名单**（哪些值得人工核查、为什么），指向 HTML 报告看图证。

**跨论文/跨作者查盗图**：
```bash
bash "<skill-dir>/scripts/run.sh" index <一批pdf...> --db "<输出目录>/fp.db"
bash "<skill-dir>/scripts/run.sh" cross --db "<输出目录>/fp.db" -o "<输出目录>"
```
（`--citations` 联网查引用是否含撤稿论文。）

## 能抓什么（详见仓库 README）

图像：一图多用·旋转翻转缩放复用·copy-move·拼接缝·跨论文盗图·复用面板聚类
统计：statcheck·GRIM·GRIMMER·SPRITE·本福特·末位数字·等差数列·跨列固定差值·跨记录小数巧合·重复行
图表：柱高vs报告值·误差棒异常　文本：ChatGPT指纹·tortured phrases·引用撤稿核查
语义裁决：由调起本 skill 的 agent（你）直接看图判（headless 下才用 --judge 外接）

## 判官 = 你（调起本 skill 的 agent）

语义裁决由**你**直接看证据图完成——你本身就是那个会看图、能推理的模型，**无需外接任何大模型**。

仅当**没有 agent 在场**（headless/批量脚本），或想要**不同模型的第二意见**时，才用外接：
`analyze --judge`（自动探测 claude/codex，或设 `PAPERCHECK_LLM_CMD='…{prompt} {images}'` 接任意 CLI）。
被 agent 调用的正常场景下**不要用** `--judge`，那是多余的第二个模型。

## 局限

科研图表/复合图会有误报（靠判官+人兜底）；统计类需能抽出数据表/统计写法才触发，GRIM 族仅整数/量表语境生效；
付费墙论文下不到 PDF；扫描版/纯矢量图抽不出栅格图；不支持 AI 生成图检测、引用失真、作者网络异常。
