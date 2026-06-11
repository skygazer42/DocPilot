# DocPilot Evaluation Datasets

This directory is the local landing zone for CPU pipeline evaluation samples.
Do not commit real PDFs, labeled corpora, customer documents, or large generated
reports here.

Samples are discovered recursively under the dataset root. Root-level samples use
the filename stem as `name`; nested samples use a dataset-relative name such as
`contracts/contract-001`. Ground-truth files are matched by the same PDF stem in
the same directory.

Each sample is matched by filename stem:

| File | Required | Purpose | Metrics |
|---|---:|---|---|
| `<name>.pdf` | yes | Source PDF to parse | `elapsed_seconds`, `text_length` |
| `<name>.gt.txt` | no | Full-text ground truth | `character_error_rate`, `word_error_rate`, `text_normalized_edit_distance` |
| `<name>.gt.blocks.json` | no | Block ground truth as a block array or structured JSON with a `blocks` field | `block_type_f1`, `reading_order_normalized_edit_distance`, `cross_page_merge_accuracy` |
| `<name>.gt.tables.html` | no | Table HTML ground truth | `mean_table_teds`, `mean_table_cell_f1` |
| `<name>.gt.html` | no | Backward-compatible single-table HTML ground truth | `mean_table_teds`, `mean_table_cell_f1` |
| `<name>.gt.formulas.json` | no | Expected formulas as an array or JSON object with `formulas`, `equations`, or `expected_formulas`; items may be strings or contain `latex` / `formula` / `text` | `formula_normalized_edit_distance`, `formula_exact_match_rate` |
| `<name>.gt.chunks.json` | no | Expected reusable chunk texts as an array or JSON object with `chunks` | `chunk_text_coverage` |
| `<name>.gt.fields.json` | no | Expected business fields as an array or JSON object with `fields`; each item may include `name`, `value`, `page_numbers` | `business_field_location_hit_rate` |

Example:

```text
biz_mini/
  contracts/
    contract-001.pdf
    contract-001.gt.txt
    contract-001.gt.blocks.json
    contract-001.gt.tables.html
    contract-001.gt.formulas.json
    contract-001.gt.chunks.json
    contract-001.gt.fields.json
```

Validate the dataset contract before running expensive parser A/B jobs:

```bash
python tools/eval_omnidocbench.py \
  --validate-dataset \
  --dataset tools/eval_datasets/biz_mini \
  --out ./eval_out/dataset-contract.json
```

The report schema is `2026-06-08.cpu-pipeline-dataset-contract.v1`. A failed
contract also fails `tools/check_cpu_pipeline_readiness.py` through the
`dataset_contract_failed` gate.

Run:

```bash
python tools/eval_omnidocbench.py \
  --engine deepdoc \
  --dataset tools/eval_datasets/biz_mini \
  --out ./eval_out/baseline.json
```

For A/B runs, pass temporary pipeline switches:

```bash
python tools/eval_omnidocbench.py \
  --engine deepdoc \
  --dataset tools/eval_datasets/biz_mini \
  --ocr-version v5 \
  --layout-engine ppdoclayout \
  --reading-order-strategy rules \
  --table-engine rapidtable \
  --out ./eval_out/ppocrv5-ppdoclayout-rules.json
```
