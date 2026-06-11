# Unified Model Hub And Parser Cleanup Implementation Plan

> Status: partially implemented. Keep this document aligned with the current product boundary: document parsing, structured artifacts, chunks, assets, and optional ingest publication.

**Goal:** Consolidate all required local DeepDoc models into one Hugging Face repository and make parser semantics consistent: `deepdoc` / `mineru` are document parsers; OCR, seal, formula, and table recognition are parsing capabilities rather than separate product layers.

**Architecture:** Introduce one canonical model root, one model repository ID, and one downloader/validator layer used by every local model consumer. Keep `mineru` and `paddleocr_vl` as external document parsing adapters. Keep the public contract focused on parsed Markdown, structured JSON, chunks, ingest export records, and assets.

**Tech Stack:** Python 3.10, `huggingface_hub`, ONNX Runtime, Flask, Gradio, Docker Compose.

---

## 0. Current State Inventory

### Current canonical local model directory

- [common/setting.py](/data/temp49/kd-brain/deepdoc-standalone/common/setting.py:11) defines `MODELS_DIR = resources/models`.
- Docker uses `/app/resources/models`.

### Current local core models

Required core files under `resources/models/`:

- `det.onnx`
- `rec.onnx`
- `ocr.res`
- `layout.onnx`
- `layout.manual.onnx`
- `layout.paper.onnx`
- `layout.laws.onnx`
- `tsr.onnx`
- `updown_concat_xgb.model`

### Current model-source fragmentation

- Base OCR/layout/TSR models: `InfiniFlow/deepdoc`
- XGBoost concat model: `InfiniFlow/text_concat_xgb_v1.0`
- Formula model: `SWHL/RapidLaTeXOCR`
- Seal detection model: `RapidAI/PP-OCRv4_server_seal_det`

### Current semantic cleanup target

- `deepdoc`: local document parser
- `mineru`: remote document parser
- `paddleocr_vl`: remote OCR/VL document parser adapter
- `plain`: plain text extractor
- `markitdown`: non-PDF office/text parser path

Feature switches belong under parsing options:

- `enable_formula`
- `enable_seal`
- `return_images`
- `return_structured`
- `include_chunks`
- `persist_artifacts`

---

## 1. Target Design

### 1.1 Canonical model root

All local models must resolve from exactly one directory:

- `resources/models/` locally
- `/app/resources/models` in Docker

Runtime code should not keep hidden alternate model roots.

### 1.2 Canonical model repository

Create one Hugging Face repository for local DeepDoc assets. Target repository:

- `qwqqwq/deepdoc-standalone`

This repo should contain all assets needed for local parsing.

### 1.3 Repository layout

```text
deepdoc-models/
  manifest.json
  det.onnx
  rec.onnx
  ocr.res
  layout.onnx
  layout.manual.onnx
  layout.paper.onnx
  layout.laws.onnx
  tsr.onnx
  updown_concat_xgb.model
  formula/
    encoder.onnx
    decoder.onnx
    tokenizer.json
  seal/
    seal_det.onnx
```

### 1.4 Compatibility policy

Do not hard-remove compatible parser aliases in the same change. When an alias is kept, docs and UI must identify whether it is local, remote, or legacy.

---

## 2. Implementation Phases

### Phase A: Unify model download and lookup

**Files to create or maintain**

- `common/model_store.py`

**Files to modify**

- `common/setting.py`
- `download_models.py`

**Design**

Create a small model-store abstraction responsible for:

- resolving the canonical model root
- checking whether a required model group is already present
- downloading a model group from the unified Hugging Face repo
- exposing exact required-file lists for `core`, `formula`, and `seal`

**Environment variables**

- `DEEPDOC_MODEL_PATH`
- `DEEPDOC_MODEL_REPO`
- `DEEPDOC_AUTO_DOWNLOAD`
- `DEEPDOC_DOWNLOAD_GROUPS`

**Required file groups**

- `core`: OCR, layout, table structure, and concat models
- `formula`: formula OCR models
- `seal`: seal detection model

**Validation**

```bash
python download_models.py manifest
python download_models.py core
find resources/models -maxdepth 2 -type f | sort
```

`python download_models.py manifest` also reports `ocr_dictionary` with `ocr.res` sha256, line count, unique character count, duplicate/empty-line checks, and required Chinese/digit/English character coverage.

### Phase B: Remove alternate runtime model roots

**Files to modify**

- `deepdoc/vision/recognizer.py`
- `deepdoc/vision/ocr.py`
- `deepdoc/vision/layout_recognizer.py`
- `deepdoc/vision/table_structure_recognizer.py`
- `deepdoc/parser/pdf_parser.py`
- `deepdoc/vision/formula_recognizer.py`
- `deepdoc/vision/seal_recognizer.py`

**Design**

Replace direct model path construction with the shared resolver from `common/model_store.py` or `common/setting.py`.

This applies to:

- OCR models
- layout models
- TSR models
- XGBoost concat model
- formula model
- seal model

**Rules**

- Runtime code must log exactly which group is missing.
- Runtime code may auto-download only when `DEEPDOC_AUTO_DOWNLOAD=1`.
- Otherwise it must fail with an actionable error.

**Validation**

```bash
rg -n "res/deepdoc|alternate model root|legacy model root" common deepdoc main.py download_models.py
python - <<'PY'
from deepdoc.vision import OCR, LayoutRecognizer
ocr = OCR()
lay = LayoutRecognizer("layout")
print("ok", bool(ocr), bool(lay))
PY
```

### Phase C: Docker startup auto-download

**Files to maintain**

- `docker/entrypoint.sh`
- `Dockerfile`
- `docker-compose.yml`

**Design**

Make Docker boot use one fixed model path:

- `/app/resources/models`

The container entrypoint should:

1. resolve the target model root
2. check required groups
3. auto-download missing groups when enabled
4. then start the requested service command

**Environment defaults**

- `DEEPDOC_MODEL_PATH=/app/resources/models`
- `DEEPDOC_AUTO_DOWNLOAD=1`
- `DEEPDOC_DOWNLOAD_GROUPS=published`

**Validation**

```bash
docker compose down
rm -rf resources/models/*
docker compose up --build
curl http://127.0.0.1:60005/health
```

### Phase D: Parser semantics cleanup

**Files to modify**

- `main.py`
- `gradio_app.py`
- `docs/API.md`
- `docs/PARSER_ENGINE_STRATEGY.md`

**Design**

Public messaging should say:

- `deepdoc`: local document parser
- `mineru`: remote document parser
- `paddleocr_vl`: remote OCR/VL parser adapter
- `plain`: plain extractor

**API policy**

- Keep current request compatibility.
- Update docs when aliases are retained for compatibility.
- Do not silently change defaults.

### Phase E: Parsing capability contract

Define the parse-result contract around document parsing:

```json
{
  "filename": "contract.pdf",
  "type": "pdf",
  "content": "...",
  "structured": {
    "document": {},
    "blocks": [],
    "assets": [],
    "chunks": []
  },
  "artifact_urls": {
    "structured_url": "...",
    "chunks_url": "..."
  }
}
```

Future feature flags should fit naturally:

- markdown persistence
- local image persistence
- object storage upload
- seal text extraction
- parser trace JSON

---

## 3. Concrete File-Level Change List

### Core files

- `common/model_store.py`
- `download_models.py`
- `main.py`
- `Dockerfile`
- `docker-compose.yml`
- `gradio_app.py`
- `docs/API.md`
- `docs/PARSER_ENGINE_STRATEGY.md`
- `deepdoc/vision/recognizer.py`
- `deepdoc/vision/ocr.py`
- `deepdoc/vision/layout_recognizer.py`
- `deepdoc/vision/table_structure_recognizer.py`
- `deepdoc/parser/pdf_parser.py`
- `deepdoc/vision/formula_recognizer.py`
- `deepdoc/vision/seal_recognizer.py`

## 4. What Not To Do In This Change

- Do not merge database persistence and object storage into model-store cleanup.
- Do not remove compatible parser aliases without an explicit API migration.
- Do not keep multiple local model roots alive long-term.
- Do not add new remote parser types while the model-store cleanup is unfinished.

## 5. Acceptance Criteria

- All local models come from one configured model repository.
- All local runtime code reads from one canonical model directory.
- Docker can start from an empty model volume and self-bootstrap.
- External parser adapters remain clearly labeled as external.
- Documentation reflects the current product model: document parsing, structured artifacts, chunks, and assets.
