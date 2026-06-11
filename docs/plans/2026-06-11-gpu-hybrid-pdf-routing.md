# GPU Hybrid PDF Routing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a GPU-focused hybrid PDF pipeline that skips unnecessary OCR on born-digital pages, keeps visual parsing for complex blocks, and prepares the async plane for page-level multi-GPU execution.

**Architecture:** Keep the public `/api/v1/parse` and `/api/v1/parse/async` surfaces stable, but extend `deepdoc` with a new `deepdoc_pdf_mode=hybrid` and `execution_profile=cpu|gpu|auto`. The implementation is staged: first decouple rasterization from OCR inside `DeepDocPdfParser`, then introduce page-level routing using native PDF text boxes as OCR substitutes for digital pages, then add a local multi-GPU page worker pool behind the existing async task plane. Mixed pages remain visually parsed for tables/formulas/seals; only ordinary text blocks bypass OCR.

**Tech Stack:** Flask API, existing `deepdoc` parser stack, `pdfplumber`/PyMuPDF text extraction, existing async task broker/store, Python multiprocessing/process pools pinned to GPU devices, repo `unittest` suite, `ruff`.

---

## Scope and assumptions

- Target the `deepdoc` local PDF engine first. Do not redesign `paddleocr_vl`, `mineru`, or `plain`.
- Keep `CPU` behavior conservative. The aggressive routing and worker-pool path is for `GPU`.
- First production target is **single node, multi-GPU**. Do not make cross-node scheduling a prerequisite for the first cut.
- Preserve current API parameters and defaults unless the new option is explicitly selected.
- Treat `26 pages/s` as a **cluster throughput** target for warm GPU services, not a blanket single-document SLA.
- Stage delivery:
  1. `hybrid` contract + parser refactor
  2. hybrid v1: `digital_clean` pages skip OCR entirely
  3. GPU local page worker pool
  4. hybrid v2: `digital_mixed` pages use selective complex-block OCR

## Current-state constraints that drive the plan

- `main.py:5466-5570` currently routes `deepdoc` PDF parsing as a whole-document choice: `auto/native/ocr`.
- `deepdoc/parser/pdf_parser.py:2568-2745` performs rasterization and OCR together inside `DeepDocPdfParser.__images__`. This is the main structural blocker to selective OCR.
- `deepdoc/parser/pdf_parser.py:1055-1064` shows layout currently expects page images plus OCR-style boxes.
- `tests/test_pdf_native_detection.py` already locks the existing `auto/native/ocr` behavior and must be extended carefully.
- `main.py:6541-6718` shows the async task plane is file-oriented today. It can be reused, but page-level execution needs an internal worker layer instead of a public API rewrite.

## End-state requirements

1. API supports `deepdoc_pdf_mode=hybrid` and `execution_profile=cpu|gpu|auto`.
2. `deepdoc` can build layout from native PDF text boxes without invoking OCR on `digital_clean` pages.
3. `deepdoc` emits per-page route metadata: `digital_clean`, `digital_mixed`, `scanned`.
4. Async GPU execution can fan out a document into page-level jobs across local GPUs while keeping the current task API stable.
5. Mixed-page complex blocks (`table`, `equation`, `seal`, visual fragments) remain on the visual path.
6. Docs, OpenAPI, structured metadata, and tests reflect the new contract.

---

### Task 1: Lock the public contract for `hybrid` mode and GPU execution profile

**Files:**
- Modify: `main.py:1522-1634`
- Modify: `docs/API.md`
- Modify: `docs/PARSER_ENGINE_STRATEGY.md`
- Modify: `openapi.json`
- Modify: `plans/optimization-roadmap.md`
- Modify: `tests/test_pdf_native_detection.py`

**Step 1: Write the failing tests**

Add tests that assert:

```python
self.assertEqual(["auto", "native", "ocr", "hybrid"], pdf_mode.get("enum"))
execution_profile = schema["properties"].get("execution_profile")
self.assertEqual(["auto", "cpu", "gpu"], execution_profile.get("enum"))
self.assertEqual("auto", execution_profile.get("default"))
```

and document checks such as:

```python
self.assertIn("hybrid", api_doc)
self.assertIn("execution_profile", api_doc)
```

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest tests.test_pdf_native_detection.PdfNativeDetectionTest.test_deepdoc_pdf_mode_contract_is_documented
```

Expected: FAIL because `hybrid` and `execution_profile` are not present.

**Step 3: Write minimal implementation**

- Extend `_normalize_deepdoc_pdf_mode` to accept `hybrid`.
- Add `_normalize_execution_profile`.
- Include `execution_profile` in `_build_parse_options`.
- Update API docs/OpenAPI/roadmap copy without changing existing defaults.

**Step 4: Run test to verify it passes**

Run:

```bash
python -m unittest tests.test_pdf_native_detection.PdfNativeDetectionTest.test_deepdoc_pdf_mode_contract_is_documented
```

Expected: PASS.

**Step 5: Commit**

```bash
git add main.py docs/API.md docs/PARSER_ENGINE_STRATEGY.md openapi.json plans/optimization-roadmap.md tests/test_pdf_native_detection.py
git commit -m "feat: add hybrid pdf mode contract and execution profile"
```

---

### Task 2: Extract native PDF text with real positions, not full-width synthetic lines

**Files:**
- Modify: `deepdoc/parser/pdf_parser.py:231-340`
- Test: `tests/test_pdf_native_detection.py`
- Create: `tests/test_pdf_hybrid_routing.py`

**Step 1: Write the failing tests**

Create tests that build a born-digital PDF with positioned words and assert the new extractor returns OCR-like boxes with geometry:

```python
boxes, meta = extract_native_pdf_text(..., preserve_geometry=True)
assert boxes[0]["x1"] > boxes[0]["x0"]
assert boxes[0]["bottom"] > boxes[0]["top"]
assert boxes[0]["layout_type"] == "text"
```

Also add a test for page-level extraction:

```python
pages = inspect_pdf_pages(...)
assert pages[0]["native_text_char_count"] > 0
```

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest tests.test_pdf_hybrid_routing
```

Expected: FAIL because positioned extraction/page inspection does not exist.

**Step 3: Write minimal implementation**

- Enhance `extract_native_pdf_text` so it can emit positioned line/word boxes using real PDF coordinates.
- Add a page inspection helper that returns per-page text/image features; keep it conservative and CPU-only.
- Do not change the existing `auto/native` behavior yet; only add new primitives.

**Step 4: Run test to verify it passes**

Run:

```bash
python -m unittest tests.test_pdf_hybrid_routing
```

Expected: PASS.

**Step 5: Commit**

```bash
git add deepdoc/parser/pdf_parser.py tests/test_pdf_native_detection.py tests/test_pdf_hybrid_routing.py
git commit -m "feat: add positioned native pdf text extraction primitives"
```

---

### Task 3: Split `DeepDocPdfParser.__images__` into rasterization and selectable OCR stages

**Files:**
- Modify: `deepdoc/parser/pdf_parser.py:343-375`
- Modify: `deepdoc/parser/pdf_parser.py:2568-2745`
- Test: `tests/test_pdf_hybrid_routing.py`

**Step 1: Write the failing tests**

Add parser-unit tests with a fake OCR implementation to prove OCR can be limited to selected pages:

```python
parser.prepare_pages(path, zoomin=3, page_from=0, page_to=4)
parser.seed_page_boxes(native_boxes_by_page)
parser.run_page_ocr(page_numbers={2, 4})
assert fake_ocr.called_pages == [2, 4]
```

Also assert that layout can still run when some pages are pre-seeded with native boxes.

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest tests.test_pdf_hybrid_routing.DeepDocHybridParserRefactorTest
```

Expected: FAIL because OCR is still hard-wired inside `__images__`.

**Step 3: Write minimal implementation**

Refactor `DeepDocPdfParser` into explicit stages:

- `prepare_pages(...)` for rasterization, outlines, page chars, and page metadata
- `seed_page_boxes(...)` for native-text-derived OCR-style boxes
- `run_page_ocr(page_numbers=...)` for OCR only on selected pages
- `finalize_page_boxes()` for cumulative height and downstream invariants

Keep the existing `__images__` entrypoint as a compatibility wrapper that calls the new staged methods with “OCR all pages”.

**Step 4: Run test to verify it passes**

Run:

```bash
python -m unittest tests.test_pdf_hybrid_routing.DeepDocHybridParserRefactorTest
```

Expected: PASS.

**Step 5: Commit**

```bash
git add deepdoc/parser/pdf_parser.py tests/test_pdf_hybrid_routing.py
git commit -m "refactor: split deepdoc pdf rasterization and page ocr stages"
```

---

### Task 4: Implement hybrid v1 page routing for `digital_clean` pages

**Files:**
- Create: `deepdoc/parser/pdf_hybrid_router.py`
- Modify: `main.py:5466-5570`
- Modify: `common/parse_builders.py`
- Test: `tests/test_pdf_hybrid_routing.py`
- Test: `tests/test_pdf_native_detection.py`

**Step 1: Write the failing tests**

Add routing tests for three page classes:

```python
plan = build_pdf_hybrid_plan(...)
assert plan["pages"][0]["route"] == "digital_clean"
assert plan["pages"][1]["route"] == "scanned"
```

Add an integration test for `main._parse_pdf_from_tmp(...)`:

```python
meta = result_meta
assert meta["pdf_parse_mode"] == "hybrid"
assert meta["page_routes"][0]["route"] == "digital_clean"
assert ocr_parser.constructed == 0
```

for a born-digital PDF when `deepdoc_pdf_mode=hybrid` and `execution_profile=gpu`.

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest tests.test_pdf_hybrid_routing.HybridRoutingIntegrationTest
```

Expected: FAIL because `hybrid` is not implemented.

**Step 3: Write minimal implementation**

- Add `build_pdf_hybrid_plan(...)` in `deepdoc/parser/pdf_hybrid_router.py`.
- For `digital_clean` pages, use native positioned boxes as layout input and skip OCR entirely.
- Preserve current whole-document `auto/native/ocr` behavior for non-hybrid modes.
- Populate `parse_meta["pdf_parse_mode"] = "hybrid"` and add `page_routes`.
- Update structured metadata so downstream artifact builders can expose the page-route source cleanly.

**Step 4: Run test to verify it passes**

Run:

```bash
python -m unittest tests.test_pdf_hybrid_routing.HybridRoutingIntegrationTest
python -m unittest tests.test_pdf_native_detection
```

Expected: PASS.

**Step 5: Commit**

```bash
git add deepdoc/parser/pdf_hybrid_router.py main.py common/parse_builders.py tests/test_pdf_hybrid_routing.py tests/test_pdf_native_detection.py
git commit -m "feat: add hybrid routing for digital clean pdf pages"
```

---

### Task 5: Reuse the current OCR path for `digital_mixed` and `scanned` pages, without public API churn

**Files:**
- Modify: `deepdoc/parser/pdf_hybrid_router.py`
- Modify: `deepdoc/parser/pdf_parser.py`
- Modify: `main.py:5466-5570`
- Test: `tests/test_pdf_hybrid_routing.py`

**Step 1: Write the failing tests**

Add tests that assert:

```python
assert page_routes[0]["route"] == "digital_mixed"
assert page_routes[1]["route"] == "scanned"
assert fake_ocr.called_pages == [1, 2]
```

where mixed/scanned pages still invoke OCR, while clean pages do not.

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest tests.test_pdf_hybrid_routing.MixedAndScannedRoutingTest
```

Expected: FAIL because hybrid v1 only handles `digital_clean`.

**Step 3: Write minimal implementation**

- Route `digital_mixed` and `scanned` pages into `run_page_ocr(page_numbers=...)`.
- Keep current table/formula/seal behavior unchanged for those pages.
- Emit per-page reasons in route metadata, e.g. `image_area_ratio_high`, `native_text_sparse`, `complex_visual_page`.

This is the first production-viable win: most born-digital pages bypass OCR, mixed/scanned pages stay safe.

**Step 4: Run test to verify it passes**

Run:

```bash
python -m unittest tests.test_pdf_hybrid_routing.MixedAndScannedRoutingTest
```

Expected: PASS.

**Step 5: Commit**

```bash
git add deepdoc/parser/pdf_hybrid_router.py deepdoc/parser/pdf_parser.py main.py tests/test_pdf_hybrid_routing.py
git commit -m "feat: route mixed and scanned pages through selective page ocr"
```

---

### Task 6: Add GPU execution profile and a local multi-GPU page worker pool behind the existing async task plane

**Files:**
- Create: `common/gpu_page_pool.py`
- Modify: `main.py:6541-6718`
- Modify: `common/async_tasks.py`
- Modify: `common/metrics.py`
- Test: `tests/test_gpu_hybrid_execution.py`
- Test: `tests/test_operational_surfaces.py`

**Step 1: Write the failing tests**

Create tests that prove:

```python
result = dispatch_gpu_page_jobs(task, page_jobs=[...], devices=[0, 1])
assert result["submitted_job_count"] == 8
assert result["worker_device_ids"] == [0, 1]
```

and an async-task integration test that asserts the public task lifecycle is unchanged while internal execution becomes page-based.

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest tests.test_gpu_hybrid_execution
```

Expected: FAIL because no GPU page pool exists.

**Step 3: Write minimal implementation**

- Introduce a local process-based GPU worker pool pinned by device ID.
- Keep `/api/v1/parse/async` and task polling APIs stable.
- When `parser_engine=deepdoc`, `execution_profile=gpu`, and `deepdoc_pdf_mode=hybrid`, decompose the document into page jobs internally.
- Emit task events for page progress, not just file progress, but keep file-level terminal behavior intact.
- Add metrics for queue depth, per-device page throughput, and per-stage timing.

Do **not** build multi-node orchestration in this task. This task is single-node, multi-GPU only.

**Step 4: Run test to verify it passes**

Run:

```bash
python -m unittest tests.test_gpu_hybrid_execution
python -m unittest tests.test_operational_surfaces
```

Expected: PASS.

**Step 5: Commit**

```bash
git add common/gpu_page_pool.py main.py common/async_tasks.py common/metrics.py tests/test_gpu_hybrid_execution.py tests/test_operational_surfaces.py
git commit -m "feat: add local multi-gpu page worker pool for hybrid pdf tasks"
```

---

### Task 7: Implement hybrid v2 selective complex-block OCR for `digital_mixed` pages

**Files:**
- Modify: `deepdoc/parser/pdf_hybrid_router.py`
- Modify: `deepdoc/parser/pdf_parser.py`
- Modify: `main.py:5466-5570`
- Test: `tests/test_pdf_hybrid_routing.py`
- Create: `tools/bench_hybrid_pdf.py`

**Step 1: Write the failing tests**

Add tests for the mixed-page target behavior:

```python
assert route["route"] == "digital_mixed"
assert route["ocr_scope"] == "complex_blocks_only"
assert route["complex_block_types"] == ["table", "equation"]
```

and assert ordinary text blocks on a mixed page are still sourced from native PDF text.

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest tests.test_pdf_hybrid_routing.SelectiveComplexBlockOcrTest
```

Expected: FAIL because mixed pages still OCR the full page.

**Step 3: Write minimal implementation**

- After layout on a mixed page, identify complex blocks (`table`, `equation`, `seal`, optionally figure captions if needed).
- Crop only those regions into the OCR/visual path.
- Preserve native text for `text/title/list/reference` blocks.
- Add a benchmark script that reports:
  - page classification counts
  - OCRed page count
  - OCRed block count
  - per-stage timings

This is the step that aligns the codebase with the intended production story: “正文读原文，复杂块走视觉”.

**Step 4: Run test to verify it passes**

Run:

```bash
python -m unittest tests.test_pdf_hybrid_routing.SelectiveComplexBlockOcrTest
python tools/bench_hybrid_pdf.py --input /path/to/sample.pdf --mode hybrid --profile gpu
```

Expected: PASS for tests; benchmark prints per-stage timings and route counts without crashing.

**Step 5: Commit**

```bash
git add deepdoc/parser/pdf_hybrid_router.py deepdoc/parser/pdf_parser.py main.py tests/test_pdf_hybrid_routing.py tools/bench_hybrid_pdf.py
git commit -m "feat: add selective complex-block ocr for mixed digital pdf pages"
```

---

### Task 8: Final docs, metrics, and performance verification

**Files:**
- Modify: `docs/API.md`
- Modify: `docs/PARSER_ENGINE_STRATEGY.md`
- Modify: `openapi.json`
- Modify: `plans/optimization-roadmap.md`
- Modify: `tests/test_pdf_native_detection.py`
- Modify: `tests/test_operational_surfaces.py`

**Step 1: Write the failing tests**

Extend docs/tests to assert:

```python
self.assertIn("hybrid", api_doc)
self.assertIn("execution_profile", api_doc)
self.assertIn("digital_clean", api_doc)
self.assertIn("digital_mixed", api_doc)
```

and that OpenAPI includes the new request fields.

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest tests.test_pdf_native_detection tests.test_operational_surfaces
```

Expected: FAIL until docs/OpenAPI are updated.

**Step 3: Write minimal implementation**

- Document the new route behavior, fallback rules, and GPU-only performance posture.
- Update operational docs with new metrics and worker health expectations.
- Record benchmark guidance:
  - born-digital corpus
  - mixed corpus
  - scanned corpus
  - warm vs cold service timings

**Step 4: Run full verification**

Run:

```bash
python -m unittest \
  tests.test_pdf_native_detection \
  tests.test_pdf_hybrid_routing \
  tests.test_gpu_hybrid_execution \
  tests.test_operational_surfaces

uv run ruff check main.py deepdoc/parser/pdf_parser.py deepdoc/parser/pdf_hybrid_router.py common/gpu_page_pool.py tests/test_pdf_native_detection.py tests/test_pdf_hybrid_routing.py tests/test_gpu_hybrid_execution.py
```

Expected: All tests pass; `ruff` reports no issues.

**Step 5: Commit**

```bash
git add docs/API.md docs/PARSER_ENGINE_STRATEGY.md openapi.json plans/optimization-roadmap.md tests/test_pdf_native_detection.py tests/test_operational_surfaces.py
git commit -m "docs: document hybrid pdf routing and gpu execution profile"
```

---

## Notes for execution

- Do not try to implement selective complex-block OCR before Task 3 is complete. The current parser architecture cannot support it cleanly while OCR is buried inside `__images__`.
- The first real latency win comes from **hybrid v1**: skipping OCR on `digital_clean` pages. Do not wait for v2 to start measuring.
- Keep API behavior stable for non-hybrid callers. `auto/native/ocr` must continue to work exactly as they do now.
- Use synthetic PDFs in tests to lock routing behavior before using real customer samples.
- Treat performance numbers as evidence, not intuition. The benchmark script is part of the deliverable.

