#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
"""表格识别 TEDS 基准：对比 DEEPDOC_TABLE_ENGINE=tatr vs rapidtable。

数据约定：--dataset 目录下成对文件 <name>.pdf 与 <name>.gt.html
         （gt 为该 PDF 主表格的标准 HTML；多表格按出现顺序与 GT 列表对齐）。
依赖    ：pip install apted lxml
用法    ：python tools/eval_table.py --dataset ./samples --engines tatr,rapidtable

⚠️ 评测工具，未在本环境联调：
   - 解析链路按 main.py 的 deepdoc pipeline 调用，方法签名以实际为准；
   - TEDS 实现取自 PubTabNet 标准做法（apted + 编辑距离）。
"""
import argparse
import os
import time
from collections import deque
from pathlib import Path


# ----------------------------- TEDS (PubTabNet 标准) -----------------------------
def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _build_tree():
    from apted import Config
    from apted.helpers import Tree

    class TableTree(Tree):
        def __init__(self, tag, colspan=0, rowspan=0, content="", *children):
            self.tag = tag
            self.colspan = colspan
            self.rowspan = rowspan
            self.content = content
            self.children = list(children)

        def bracket(self):
            if self.tag == "td":
                result = '"tag": %s, "colspan": %d, "rowspan": %d, "text": %s' % (
                    self.tag,
                    self.colspan,
                    self.rowspan,
                    self.content,
                )
            else:
                result = '"tag": %s' % self.tag
            for child in self.children:
                result += child.bracket()
            return "{{{}}}".format(result)

    class CustomConfig(Config):
        @staticmethod
        def maximum(*sequences):
            return max(map(len, sequences))

        def normalized_distance(self, *sequences):
            m = self.maximum(*sequences)
            return float(_levenshtein(*sequences)) / m if m else 0.0

        def rename(self, node1, node2):
            if (
                node1.tag != node2.tag
                or node1.colspan != node2.colspan
                or node1.rowspan != node2.rowspan
            ):
                return 1.0
            if node1.tag == "td" and (node1.content or node2.content):
                return self.normalized_distance(node1.content or "", node2.content or "")
            return 0.0

    return TableTree, CustomConfig


def _html_to_tree(table_html, TableTree):
    from lxml import html as lxml_html

    root = lxml_html.fromstring(table_html)
    tables = root.xpath("//table")
    node = tables[0] if tables else root
    tree = TableTree("table")
    stack = deque([(node, tree)])
    while stack:
        el, parent = stack.popleft()
        for child in el.getchildren():
            tag = child.tag
            if tag in ("tr",):
                tnode = TableTree("tr")
                parent.children.append(tnode)
                stack.append((child, tnode))
            elif tag in ("td", "th"):
                colspan = int(child.get("colspan", 1) or 1)
                rowspan = int(child.get("rowspan", 1) or 1)
                content = "".join(child.itertext()).strip()
                parent.children.append(TableTree("td", colspan, rowspan, content))
            else:
                stack.append((child, parent))
    return tree


def _count(tree) -> int:
    return 1 + sum(_count(c) for c in tree.children)


def compute_teds(pred_html: str, gt_html: str) -> float:
    """Tree-Edit-Distance-based Similarity，范围 [0,1]，越大越好。"""
    from apted import APTED

    TableTree, CustomConfig = _build_tree()
    tp = _html_to_tree(pred_html, TableTree)
    tg = _html_to_tree(gt_html, TableTree)
    n = max(_count(tp), _count(tg))
    if n == 0:
        return 1.0
    dist = APTED(tp, tg, CustomConfig()).compute_edit_distance()
    return 1.0 - float(dist) / n


# ----------------------------- 解析（deepdoc pipeline） -----------------------------
def parse_pdf_tables(pdf_path: str, engine: str, zoomin: int = 3) -> list[str]:
    """按 main.py 的 deepdoc pipeline 解析 PDF，返回各表格 HTML 字符串列表。"""
    os.environ["DEEPDOC_TABLE_ENGINE"] = engine
    from deepdoc.parser.pdf_parser import DeepDocPdfParser

    parser = DeepDocPdfParser()
    max_pages = int(os.environ.get("DEEPDOC_PDF_MAX_PAGES", "10"))
    parser.__images__(pdf_path, zoomin, page_from=0, page_to=max_pages)
    parser._layouts_rec(zoomin)
    parser._table_transformer_job(zoomin)
    parser._text_merge()
    tbls = parser._extract_table_figure(False, zoomin, True, False)  # return_html=True

    htmls = []
    for item in tbls:
        # need_position=False -> item = (img, html)
        try:
            _img, payload = item
        except (TypeError, ValueError):
            continue
        if isinstance(payload, str) and "<table" in payload:
            htmls.append(payload)
    return htmls


def _load_dataset(dataset: str):
    """成对 <name>.pdf + <name>.gt.html。返回 [(name, pdf_path, [gt_html]), ...]。"""
    base = Path(dataset)
    samples = []
    for pdf in sorted(base.glob("*.pdf")):
        gt = pdf.with_suffix(".gt.html")
        if not gt.exists():
            continue
        samples.append((pdf.stem, str(pdf), [gt.read_text(encoding="utf-8")]))
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="含成对 *.pdf 与 *.gt.html 的目录")
    ap.add_argument("--engines", default="tatr,rapidtable")
    ap.add_argument("--limit", type=int, default=0, help="仅评测前 N 个样本(0=全部)")
    args = ap.parse_args()

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    samples = _load_dataset(args.dataset)
    if args.limit:
        samples = samples[: args.limit]
    if not samples:
        raise SystemExit(f"未找到成对样本（*.pdf + *.gt.html）于 {args.dataset}")

    # engine -> list[(teds, seconds)]
    results: dict[str, list[tuple[float, float]]] = {e: [] for e in engines}
    for name, pdf_path, gt_list in samples:
        for engine in engines:
            t0 = time.time()
            try:
                preds = parse_pdf_tables(pdf_path, engine)
            except Exception as e:  # noqa: BLE001
                print(f"[WARN] {name} engine={engine} 解析失败: {e}")
                results[engine].append((0.0, time.time() - t0))
                continue
            elapsed = time.time() - t0
            # 按顺序对齐 pred 与 gt，取可比对的最小长度
            scores = [
                compute_teds(p, g) for p, g in zip(preds, gt_list)
            ]
            avg = sum(scores) / len(scores) if scores else 0.0
            results[engine].append((avg, elapsed))
            print(f"{name:<28} {engine:<10} TEDS={avg:.4f} t={elapsed:.2f}s")

    print("\n===== 汇总（均值）=====")
    print(f"{'engine':<12}{'TEDS':<10}{'sec/pdf':<10}")
    for engine in engines:
        rows = results[engine]
        mteds = sum(r[0] for r in rows) / len(rows) if rows else 0.0
        msec = sum(r[1] for r in rows) / len(rows) if rows else 0.0
        print(f"{engine:<12}{mteds:<10.4f}{msec:<10.2f}")


if __name__ == "__main__":
    main()
