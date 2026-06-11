# DocPilot 复杂表格还原（RapidTable / SLANet-plus）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为本地 `deepdoc` 引擎新增一条纯 CPU/ONNX 的 RapidTable(SLANet-plus)表格识别路径，由 `DEEPDOC_TABLE_ENGINE` 开关控制，默认回退现有 TATR，显著提升复杂表格的 HTML 结构还原质量。

**Architecture:** 在 `_extract_table_figure` 生成表格 HTML 的唯一调用处（`pdf_parser.py:1676`）做引擎分流。`tatr`（默认）走现有 `construct_table` 几何拼装；`rapidtable` 把已裁好的表格图（跨页已由 `cropout` 拼接）+ 该表 OCR 框喂给新封装的 `RapidTableRecognizer` 直接得 HTML。单表级失败自动回退 TATR，最坏情况等于现状。

**Tech Stack:** Python 3.10、onnxruntime（已有）、`rapid-table`（新增可选依赖，纯 ONNX/CPU）、huggingface_hub（已有，模型分发）。

---

## 约定与前置说明（执行前必读）

1. **验证方式**：本仓库**无 pytest**（见 `AGENTS.md`）。本计划的"测试"统一用**独立可运行脚本 + API 冒烟 + `ruff`**表达，而非 pytest。每个任务给出确切命令与预期输出。
2. **git**：当前编排环境的 Bash 看不到 `.git`，无法提交。各任务的 commit 步骤需在 **git 可用的环境**执行；命令照写。
3. **核心不变量**：未设 `DEEPDOC_TABLE_ENGINE` 或设为 `tatr` 时，**输出与改造前逐字节一致**。每个改动 pdf_parser 的任务都要回归验证这一点。
4. **坐标系关键点**：传给 `construct_table` 的 `bxs` 用的是**页面累积坐标**（`top` 含 `page_cum_height`）；RapidTable 需要 OCR 框相对**裁剪表格图**的局部像素坐标。坐标转换在 Task 3 显式处理，务必先可视化校验对齐。
5. **依赖隔离**：`rapid-table` 作为 `[project.optional-dependencies]` 的 `table` extra，未安装时 `rapidtable` 引擎自动回退 TATR，不影响默认安装。

---

## File Structure

**创建：**
- `deepdoc/vision/rapid_table_recognizer.py` — RapidTable 封装：模型加载、OCR 框格式+坐标转换、`__call__(table_image, ocr_boxes, crop_origin, zoomin) -> html`
- `tools/probe_rapidtable.py` — Task 0 临时探针，核验真实 API/依赖（验证后可删）
- `deepdoc/vision/t_table.py` — 冒烟脚本：单张表格图 + 模拟 OCR 框 → 打印 HTML
- `tools/eval_table.py` — TEDS 基准对比脚本（tatr vs rapidtable）

**修改：**
- `pyproject.toml` — 新增 `table` 可选依赖
- `common/model_store.py` — `MODEL_GROUP_FILES` 增加 `table_v2` 组
- `download_models.py` — 命令白名单加入 `table_v2`
- `deepdoc/parser/pdf_parser.py` — `__init__` 惰性持有 recognizer；`_extract_table_figure:1676` 引擎分流；单表回退
- `docs/API.md` — 记录 `DEEPDOC_TABLE_ENGINE`
- `docs/PARSER_ENGINE_STRATEGY.md` — 补充表格引擎说明

---

## Task 0: 依赖声明 + RapidTable 真实 API/依赖探针（go/no-go）

**目的**：v3.0.0 接口变过、SLANet-plus 有 ONNX 版本坑、依赖可能拖入 torch/rapidocr——必须先用真实安装核验，否则后续代码建立在假设上。

**Files:**
- Modify: `pyproject.toml`
- Create: `tools/probe_rapidtable.py`

- [ ] **Step 1: 在 `pyproject.toml` 增加可选依赖**

在 `[project.optional-dependencies]` 下新增（版本号以安装时最新且支持 SLANet-plus ONNX 的为准）：
```toml
table = [
    "rapid-table>=1.0.0",
]
```

- [ ] **Step 2: 安装并写探针脚本 `tools/probe_rapidtable.py`**

```python
"""一次性探针：核验 RapidTable 真实 API、SLANet-plus ONNX 可用性、依赖是否纯 CPU。"""
import sys

def main():
    import rapid_table
    print("rapid_table version:", getattr(rapid_table, "__version__", "unknown"))
    # 核验：构造入口、模型类型选择、调用签名、返回结构
    from rapid_table import RapidTable  # 若类名/入参不同，记录真实名称
    print("RapidTable signature:", RapidTable.__init__.__doc__)
    # 核验依赖是否拖入 torch
    print("torch imported:", "torch" in sys.modules)
    # 若有现成 demo 图，跑一次，打印返回对象的字段（pred_html / cell_bboxes / elapse 等）

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 运行探针，记录真实事实**

Run: `uv run python tools/probe_rapidtable.py`
记录到本任务下方（供后续任务引用）：
- 入口类名与构造参数（如何选 `slanet_plus` ONNX、如何指定本地 `model_path`）
- `__call__`/`run` 签名：是否接受 `ocr_result`？格式？能否只传图让其内部 OCR？
- 返回结构字段名（HTML 字段、cell bbox 字段）
- `pip show rapid-table` 的依赖树是否含 `torch`（必须不含）；是否强依赖 `rapidocr`

- [ ] **Step 4: go/no-go 判定**

若 SLANet-plus 无法纯 ONNX/CPU 运行，或强制拖入 torch → **停止，回到 spec 重新选型**（如改用 `RapidAI/TableStructureRec` 的 ONNX 模型直连）。否则继续。

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml tools/probe_rapidtable.py
git commit -m "chore: 声明 rapid-table 可选依赖并核验 ONNX/CPU 可用性"
```

---

## Task 1: 模型分组 `table_v2`

**Files:**
- Modify: `common/model_store.py:12-31`（`MODEL_GROUP_FILES`）
- Modify: `download_models.py:44`（命令白名单）

- [ ] **Step 1: 在 `MODEL_GROUP_FILES` 增加 `table_v2`**

文件名以 Task 0 探针确认的实际模型文件为准，示例：
```python
    "table_v2": (
        "table/slanet_plus.onnx",
        "table/table_cls.onnx",
    ),
```

- [ ] **Step 2: `download_models.py` 命令白名单加入 `table_v2`**

`main` 中 `elif command in {"core", "formula", "seal", "all"}:` 改为：
```python
    elif command in {"core", "formula", "seal", "table_v2", "all"}:
```
并更新 Usage 字符串。

- [ ] **Step 3: 验证分组逻辑（不实际下载）**

Run: `uv run python download_models.py manifest`
Expected: 输出 JSON 中 `groups` 含 `table_v2`，其 `required_files` 为上面两项，`missing_files` 列出（因尚未上传）。

- [ ] **Step 4: Commit**

```bash
git add common/model_store.py download_models.py
git commit -m "feat: 新增 table_v2 模型分组（RapidTable SLANet-plus）"
```

> 模型上传到 HF repo `qwqqwq/deepdoc-standalone` 的 `table/` 子目录、用 `tools/publish_models_to_hf.py` 推送，作为部署准备步骤（非代码任务），在 Task 2 联调前完成。

---

## Task 2: `RapidTableRecognizer` 封装

**Files:**
- Create: `deepdoc/vision/rapid_table_recognizer.py`
- Create: `deepdoc/vision/t_table.py`

- [ ] **Step 1: 写冒烟脚本 `deepdoc/vision/t_table.py`（先定义期望行为）**

```python
"""冒烟：给定表格图 + 局部坐标 OCR 框，RapidTableRecognizer 输出非空 HTML <table>。"""
import argparse
from PIL import Image
from deepdoc.vision.rapid_table_recognizer import RapidTableRecognizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    args = ap.parse_args()
    img = Image.open(args.image).convert("RGB")
    # 用整图当一个文本框占位，仅验证管线连通（真实对齐在 PDF 端到端验证）
    w, h = img.size
    boxes = [{"text": "smoke", "x0": 0, "x1": w, "top": 0, "bottom": h, "score": 1.0}]
    rec = RapidTableRecognizer()
    html = rec(img, boxes, crop_origin=(0, 0), zoomin=1)
    print(html)
    assert "<table" in html, "expected an HTML table"

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行冒烟，确认失败（模块未实现）**

Run: `uv run python deepdoc/vision/t_table.py --image <任一表格截图>`
Expected: `ModuleNotFoundError` 或 `ImportError`（`rapid_table_recognizer` 不存在）。

- [ ] **Step 3: 实现 `rapid_table_recognizer.py`（API 以 Task 0 核验为准）**

```python
import os
import numpy as np
from common import logger
from common.model_store import ensure_groups

class RapidTableRecognizer:
    """RapidTable(SLANet-plus, ONNX) 表格识别封装：表格图 + OCR 框 -> HTML。"""

    def __init__(self):
        from rapid_table import RapidTable, RapidTableInput  # 名称以 Task0 为准
        model_root = ensure_groups("table_v2")
        model_path = os.path.join(model_root, "table", "slanet_plus.onnx")
        self._engine = RapidTable(
            RapidTableInput(model_type="slanet_plus", model_path=model_path)
        )

    @staticmethod
    def _to_local_ocr_result(ocr_boxes, crop_origin, zoomin):
        """把页面累积坐标的 box 转为相对裁剪图的局部像素四点框。
        crop_origin=(left, top) 为裁剪原点（页面局部坐标，未乘 ZM）。"""
        ox, oy = crop_origin
        result = []
        for b in ocr_boxes:
            text = (b.get("text") or "").strip()
            if not text:
                continue
            x0 = (b["x0"] - ox) * zoomin
            x1 = (b["x1"] - ox) * zoomin
            top = (b["top"] - oy) * zoomin
            bottom = (b["bottom"] - oy) * zoomin
            poly = [[x0, top], [x1, top], [x1, bottom], [x0, bottom]]
            result.append([poly, text, float(b.get("score", 1.0))])
        return result

    def __call__(self, table_image, ocr_boxes, crop_origin=(0, 0), zoomin=1):
        img = np.array(table_image) if not isinstance(table_image, np.ndarray) else table_image
        ocr_result = self._to_local_ocr_result(ocr_boxes, crop_origin, zoomin)
        if not ocr_result:
            return ""
        output = self._engine(img, ocr_results=ocr_result)  # 签名以 Task0 为准
        html = getattr(output, "pred_html", None) or ""
        return html or ""
```

- [ ] **Step 4: 运行冒烟，确认通过**

Run: `uv run python deepdoc/vision/t_table.py --image <表格截图>`
Expected: 打印含 `<table` 的 HTML，脚本退出码 0。

- [ ] **Step 5: lint + Commit**

```bash
uv run ruff check deepdoc/vision/rapid_table_recognizer.py deepdoc/vision/t_table.py
git add deepdoc/vision/rapid_table_recognizer.py deepdoc/vision/t_table.py
git commit -m "feat: 新增 RapidTableRecognizer(SLANet-plus ONNX) 封装与冒烟脚本"
```

---

## Task 3: `pdf_parser` 引擎分流（单表，含坐标对齐）

**Files:**
- Modify: `deepdoc/parser/pdf_parser.py`（`DeepDocPdfParser.__init__` 约 `:60-95`；`_extract_table_figure` 表格产出处 `:1661-1681`）

- [ ] **Step 1: `__init__` 惰性持有 recognizer + 读取引擎开关**

在 `DeepDocPdfParser.__init__`（`tbl_det = TableStructureRecognizer()` 之后）加入：
```python
        self.table_engine = os.getenv("DEEPDOC_TABLE_ENGINE", "tatr").strip().lower()
        self._rapid_table = None  # 惰性初始化

    def _get_rapid_table(self):
        if self._rapid_table is None:
            from deepdoc.vision.rapid_table_recognizer import RapidTableRecognizer
            self._rapid_table = RapidTableRecognizer()
        return self._rapid_table
```
（确认 `pdf_parser.py` 顶部已 `import os`；无则补。）

- [ ] **Step 2: 在表格产出处分流（`_extract_table_figure` 的 `for k, bxs in tables.items()` 循环内，`:1670-1681`）**

把 `cropout` 与 `construct_table` 部分改为：
```python
            img = cropout(bxs, "table", poss)
            if img is None:
                continue
            if self.table_engine == "rapidtable":
                html = self._table_html_rapid(img, bxs, poss, ZM)
            else:
                html = self.tbl_det.construct_table(
                    bxs, html=return_html, is_english=self.is_english
                )
            res.append((img, html))
            positions.append(poss)
```

并新增方法（坐标原点取该表裁剪框左上角；`poss` 末项为 `(pn, left, right, top, bott)`）：
```python
    def _table_html_rapid(self, img, bxs, poss, ZM):
        # 首版仅保证单页表的局部坐标对齐；跨页表（拼接图）的坐标映射在 Task 5 实现，
        # 在此之前跨页表回退 TATR，避免错位。ZM 由 _extract_table_figure 的同名参数传入。
        pages = {b.get("page_number") for b in bxs}
        is_cross_page = len(pages) > 1 or len(poss) > 1
        if not is_cross_page:
            try:
                left = poss[-1][1] if poss else float(np.min([b["x0"] for b in bxs]))
                top_local = poss[-1][3] if poss else 0.0
                html = self._get_rapid_table()(
                    img, bxs, crop_origin=(left, top_local), zoomin=ZM
                )
                if html and "<table" in html:
                    return html
                logger.warning("rapidtable empty/invalid output, fallback to TATR")
            except Exception as e:  # noqa: BLE001
                logger.warning("rapidtable failed (%s), fallback to TATR", e)
        return self.tbl_det.construct_table(bxs, html=True, is_english=self.is_english)
```
> 说明：`ZM` 直接取 `_extract_table_figure(self, need_image, ZM, ...)` 作用域内的同名参数，不新增方法。本任务只保证**单页表**坐标对齐；**跨页表**（`bxs` 跨多页或 `poss` 多项）首版回退 TATR，其正确的局部坐标映射在 Task 5 实现。

- [ ] **Step 3: 坐标对齐可视化校验（关键正确性步骤）**

写临时脚本：对一张含表格的单页 PDF，导出 `_table_html_rapid` 入参的 `img` 与转换后的局部 OCR 框，用 OpenCV 在 `img` 上画框，肉眼确认框落在对应文字上。对齐错位则修正 `crop_origin`/`zoomin` 公式后重试。
Expected: 画出的框与表格文字基本重合。

- [ ] **Step 4: 端到端 API 冒烟（单页表格 PDF）**

```bash
DEEPDOC_TABLE_ENGINE=rapidtable uv run python main.py &   # 或 uvicorn
curl -X POST "http://localhost:8000/api/v1/parse" \
  -F "parser_engine=deepdoc" -F "file=@<单页表格.pdf>"
```
Expected: 返回 JSON，`markdown` 含结构合理的 `<table>`（合并单元格/无线表优于 TATR）。

- [ ] **Step 5: 回归——默认 tatr 不变**

```bash
uv run python main.py &   # 不设 DEEPDOC_TABLE_ENGINE
curl -X POST "http://localhost:8000/api/v1/parse" -F "parser_engine=deepdoc" -F "file=@<样本.pdf>" -o tatr_after.json
```
Expected: 与改造前同样本输出一致（diff 为空）。

- [ ] **Step 6: lint + Commit**

```bash
uv run ruff check deepdoc/parser/pdf_parser.py
git add deepdoc/parser/pdf_parser.py
git commit -m "feat: pdf_parser 表格识别按 DEEPDOC_TABLE_ENGINE 分流（rapidtable/tatr）"
```

---

## Task 4: 单表级回退加固 + 可观测日志

**Files:**
- Modify: `deepdoc/parser/pdf_parser.py`（`_get_rapid_table` / `_table_html_rapid`）

- [ ] **Step 1: 依赖/模型缺失场景回退**

`_get_rapid_table` 捕获 `ImportError`（未装 `table` extra）与 `FileNotFoundError`（模型缺失，由 `ensure_groups` 抛出），记录一次 warning 并将 `self.table_engine` 降级为 `tatr`，使后续表格不再重试。

- [ ] **Step 2: 构造缺失场景验证回退**

在未安装 `rapid-table` 的环境（或临时改名模型文件）设 `DEEPDOC_TABLE_ENGINE=rapidtable` 跑样本 PDF。
Expected: 日志出现一次回退 warning；接口正常返回（等同 tatr 结果），不抛裸异常。

- [ ] **Step 3: Commit**

```bash
git add deepdoc/parser/pdf_parser.py
git commit -m "feat: rapidtable 引擎依赖/模型缺失时单表级回退 TATR 并记录日志"
```

---

## Task 5: 跨页表（拼接图直喂 + 表头去重降级）

**Files:**
- Modify: `deepdoc/parser/pdf_parser.py`（`_table_html_rapid` 跨页分支）

- [ ] **Step 1: 实现跨页拼接图的局部坐标映射（取消 Task 3 的跨页回退）**

`cropout` 跨页时把多页裁图按页序垂直拼接为长图（`pdf_parser.py:1610-1634`）。在调用 `RapidTableRecognizer` 前，把各页 box 的局部坐标映射到拼接图：按 `page_number` 排序，y 偏移 = 之前各页裁图的像素高度累加，x 用该页裁剪框 `left`；据此构造局部 OCR 框。完成后放开 Task 3 中"跨页回退 TATR"的限制。
Run: `DEEPDOC_TABLE_ENGINE=rapidtable` 跑一个**已知跨页表**的 PDF。
Expected: 得到一张合并的 `<table>`；按 Task 3 Step 3 方式做坐标对齐可视化校验通过；记录是否出现"续表表头重复成数据行"。

- [ ] **Step 2: 仅当出现表头重复时，加 HTML 层去重**

若 Step 1 出现重复表头：对输出 HTML 做后处理——识别与首个表头 `<tr>` 文本完全相同的后续 `<tr>` 并删除。列数/表头无法对齐时**保持原样**（不强行改写），并记 debug 日志。
> YAGNI：Step 1 若已可接受，跳过 Step 2，仅在 spec 验收子项不达标时再做。

- [ ] **Step 3: 跨页样本验证 + Commit**

Expected: 跨页表 TEDS 不低于 tatr；无明显错行。
```bash
git add deepdoc/parser/pdf_parser.py
git commit -m "feat: rapidtable 跨页表识别（拼接图直喂，按需表头去重）"
```

---

## Task 6: TEDS 基准评测脚本

**Files:**
- Create: `tools/eval_table.py`

- [ ] **Step 1: 写 `tools/eval_table.py`**

输入一批样本（OmniDocBench 表格子集 / PubTabNet 抽样，含 GT HTML），分别用 `DEEPDOC_TABLE_ENGINE=tatr` 和 `rapidtable` 解析，计算 TEDS（用 `apted` 或现成 TEDS 实现），输出对比表（总分 + 无线表/合并单元格分项 + CPU 单表耗时）。
```python
"""TEDS 基准：对比 tatr vs rapidtable。用法见 --help。"""
# 1) 遍历样本目录；2) 对每个引擎跑解析得 HTML；3) 与 GT 算 TEDS；4) 汇总打印表格
```

- [ ] **Step 2: 跑出对比并记录**

Run: `uv run python tools/eval_table.py --dataset <omnidocbench_table_subset> --engines tatr,rapidtable`
Expected: 打印对比表；`rapidtable` 总 TEDS 明显高于 `tatr`（向 ~82 看齐），无线表/合并子项不退化。

- [ ] **Step 3: Commit**

```bash
git add tools/eval_table.py
git commit -m "feat: 新增表格识别 TEDS 基准脚本(tatr vs rapidtable)"
```

---

## Task 7: 文档更新

**Files:**
- Modify: `docs/API.md`
- Modify: `docs/PARSER_ENGINE_STRATEGY.md`

- [ ] **Step 1: `docs/API.md` 记录新环境变量**

新增 `DEEPDOC_TABLE_ENGINE`（`tatr` 默认 / `rapidtable`）说明：仅作用于 `parser_engine=deepdoc` 的 PDF；缺依赖/模型时自动回退。

- [ ] **Step 2: `docs/PARSER_ENGINE_STRATEGY.md` 补充**

在 deepdoc 引擎一节说明：表格识别可切换 SLANet-plus(rapidtable) 提升复杂表还原，并附 Task 6 的 TEDS 对比数字与 CPU 耗时。

- [ ] **Step 3: Commit**

```bash
git add docs/API.md docs/PARSER_ENGINE_STRATEGY.md
git commit -m "docs: 记录 DEEPDOC_TABLE_ENGINE 与表格引擎选型/基准"
```

---

## 完成定义（Definition of Done）

- [ ] 默认（`tatr`）输出与改造前一致（回归通过）
- [ ] `DEEPDOC_TABLE_ENGINE=rapidtable` 端到端跑通单页与跨页表格 PDF
- [ ] 依赖/模型缺失时单表级回退、不报裸异常
- [ ] `tools/eval_table.py` 显示 rapidtable 在公开基准 TEDS 明显优于 tatr，无线表/合并子项不退化
- [ ] `uv run ruff check .` 通过
- [ ] 文档已更新
