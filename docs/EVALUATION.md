# Evaluation Guide

本文档只覆盖评测、profile 和 CPU pipeline readiness gate。部署安装看 [DEPLOYMENT.md](DEPLOYMENT.md)，接口契约看 [API.md](API.md)。

## Dataset Layout

评测集目录按 PDF 样本组织，工具会递归发现 `--dataset` 下的 PDF。样本名使用 dataset-relative 路径，例如 `contracts/sample`，避免不同子目录里的同名 PDF 冲突。

推荐布局：

```text
tools/eval_datasets/biz_mini/
  contracts/
    sample.pdf
    sample.blocks.json
    sample.chunks.json
    sample.fields.json
    sample.formulas.json
    sample.tables.json
```

ground truth 只需要放对应评测要用到的字段，但必须能和同名 PDF 对齐。

## Dataset Contract

先跑 dataset contract 预检，再做正式 A/B：

```bash
python tools/eval_omnidocbench.py \
  --validate-dataset \
  --dataset tools/eval_datasets/biz_mini \
  --out eval_out/dataset-contract.json
```

输出 schema 是 `2026-06-08.cpu-pipeline-dataset-contract.v1`。它会检查：

- PDF 存在且非空
- 标注文件能匹配同名 PDF
- blocks / chunks / fields / formulas / table HTML 等 ground truth 具备可评测字段

正式 `--engine deepdoc --dataset ...` 评测也会先执行这一步；contract 失败时不会进入 PDF 解析。

## License Gate

候选组件进入本地解析主链路前，先跑 license gate：

```bash
python tools/eval_omnidocbench.py \
  --license-gate \
  --out eval_out/license-gate.json
```

输出 schema 是 `2026-06-08.cpu-pipeline-license-gate.v1`，状态必须是 `passed`。这个门禁用于阻止 AGPL / GPL 候选组件进入本地 CPU pipeline。

## A/B Evaluation

正式评测示例：

```bash
python tools/eval_omnidocbench.py \
  --engine deepdoc \
  --dataset tools/eval_datasets/biz_mini \
  --out eval_out/ocr-v4.json

python tools/eval_omnidocbench.py \
  --engine deepdoc \
  --dataset tools/eval_datasets/biz_mini \
  --layout-engine ppdoclayout \
  --reading-order-strategy rules \
  --out eval_out/layout-ppdoclayout-rules.json
```

报告必须和本次 readiness 的 `--dataset` 一致，并且要自带：

- `license_gate`
- `dataset_contract`
- `model_manifest`

同时要求：

- `engine` 必须是 `deepdoc`
- `samples` 必须是数组，且每项都是 JSON object
- `samples[*].pdf_path` 必填，且能对上 dataset contract
- `summary.sample_count == len(samples)`，且必须等于 dataset contract 的样本数
- 由 `samples` 生成的均值指标必须可重算且一致

## Readiness Gate

CPU pipeline readiness 汇总检查：

```bash
python tools/check_cpu_pipeline_readiness.py \
  --dataset tools/eval_datasets/biz_mini \
  --min-pages 100 \
  --license-gate-report eval_out/license-gate.json \
  --ocr-baseline-report eval_out/ocr-v4.json \
  --ocr-candidate-report eval_out/ocr-v5.json \
  --layout-baseline-report eval_out/layout-legacy.json \
  --layout-candidate-report eval_out/layout-ppdoclayout-rules.json \
  --table-baseline-report eval_out/table-tatr.json \
  --table-candidate-report eval_out/table-rapidtable.json \
  --formula-baseline-report eval_out/formula-rapidlatex.json \
  --formula-candidate-report eval_out/formula-pp_formula_net_s.json \
  --profile-report eval_out/profile.json
```

它会检查：

- `core_v5`、`layout_v2`、`table_v2`、`formula_v2` 模型组
- dataset contract
- license gate
- 真实 PDF 页数下限
- OCR / layout / table / formula 的 baseline / candidate A/B 报告
- profile 报告

任一关键项缺失或状态失败，整体 readiness 直接失败，不会把脚手架或旧报告误判成可切默认。

## Profiling

profile 示例：

```bash
python tools/profile_pipeline.py \
  tools/eval_datasets/biz_mini/contracts/sample.pdf \
  --dataset tools/eval_datasets/biz_mini \
  --layout-engine legacy \
  --ocr-version v4 \
  --formula-mode rapidlatex \
  --reading-order-strategy legacy \
  --out eval_out/profile.json
```

profile 报告必须包含：

- `pipeline_config`
- `model_manifest`
- `license_gate`
- `dataset_contract`
- `stage_summary`

阶段列表只能包含这 7 个阶段：

- `rasterize_ocr`
- `layout`
- `table`
- `text_merge`
- `cross_page_text`
- `reading_order`
- `extract_assets`

顶层 `dataset`、`sample_name` 和 `pdf_path` 也必须对上本次 dataset contract。

## Model Manifest

模型目录快照可单独输出：

```bash
python download_models.py manifest
```

`model_manifest` / `model_group_provenance` 用来固定模型来源、许可证和文件快照，避免模型目录变了却复用旧评测报告。

发布模型包后，可再校验一次远端 manifest：

```bash
python tools/ci/verify_hf_models.py --groups all
```

## Related Docs

- [Deployment Guide](DEPLOYMENT.md)
- [API Reference](API.md)
- [Parser Engine Strategy](PARSER_ENGINE_STRATEGY.md)
