# Deployment Guide

本文档只覆盖部署和运行相关内容：安装、模型下载、启动方式、Docker、artifact backend、ingest 和运行参数。接口契约请看 [API.md](API.md)，健康检查和运维面请看 [OPERATIONS.md](OPERATIONS.md)，评测与 readiness gate 请看 [EVALUATION.md](EVALUATION.md)。

## Prerequisites

- Python 3.10-3.12
- Java Runtime，用于部分 Tika 解析路径
- 系统库：`libgl1`、`libglib2.0-0`

## Local Installation

```bash
conda create -n deepdoc python=3.10
conda activate deepdoc

pip install -e .
```

可选依赖：

```bash
# Gradio 控制台
pip install -e ".[gradio]"

# S3 / MinIO artifact backend
pip install -e ".[artifact-s3]"

# PostgreSQL ingest backend
pip install -e ".[ingest-postgres]"
```

## Model Download

默认下载到 `resources/models`。

```bash
# 已发布模型组
python download_models.py published

# 最小本地 PDF 解析模型组
python download_models.py core

# 可选能力模型
python download_models.py formula
python download_models.py seal

# CPU pipeline staged upgrade groups
python download_models.py core_v5
python download_models.py layout_v2
python download_models.py table_v2
python download_models.py formula_v2

# 全部声明模型组
python download_models.py all

# 查看模型 manifest、缺失文件和 OCR 字典/识别对齐
python download_models.py manifest
```

模型目录说明：

- 宿主机默认：`resources/models`
- 容器内统一：`/app/resources/models`
- 默认模型仓库：`qwqqwq/deepdoc-standalone`

CPU pipeline 现代化相关环境变量：

| 变量 | 默认值 | 可选值 | 说明 |
|---|---|---|---|
| `DEEPDOC_OCR_VERSION` | `v4` | `v4` / `v5` | `v5` 使用 `core_v5` 中的 OCR 模型和字典 |
| `DEEPDOC_REC_IMAGE_SHAPE` | 空 | `C,H,W` | 通用 OCR 识别输入尺寸覆盖 |
| `DEEPDOC_OCR_V4_REC_IMAGE_SHAPE` | 空 | `C,H,W` | v4 OCR 输入尺寸覆盖 |
| `DEEPDOC_OCR_V5_REC_IMAGE_SHAPE` | 空 | `C,H,W` | v5 OCR 输入尺寸覆盖 |
| `DEEPDOC_LAYOUT_ENGINE` | `legacy` | `legacy` / `ppdoclayout` | `ppdoclayout` 需要 `layout_v2` |
| `DEEPDOC_TABLE_ENGINE` | `tatr` | `tatr` / `rapidtable` | `rapidtable` 需要 `table_v2` |
| `DEEPDOC_FORMULA_MODE` | `rapidlatex` | `rapidlatex` / `pp_formula_net_s` | `pp_formula_net_s` 需要 `formula_v2` |
| `DEEPDOC_READING_ORDER_STRATEGY` | `legacy` | `legacy` / `rules` | 多栏排序和更严格的跨页规则 |

示例：

```bash
python download_models.py core_v5
DEEPDOC_OCR_VERSION=v5 python main.py

python download_models.py layout_v2
DEEPDOC_LAYOUT_ENGINE=ppdoclayout python main.py

python download_models.py table_v2
DEEPDOC_TABLE_ENGINE=rapidtable python main.py
```

## Run Locally

### API Service

```bash
export DEEPDOC_MODEL_PATH=./resources/models
python main.py
```

开发调试：

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Gradio Console

```bash
python gradio_app.py
```

默认监听 `0.0.0.0:7860`。Gradio 只作为本地 Python 调试入口，不进入默认 Docker 部署。

## Docker Deployment

### Single Container

```bash
docker build -f Dockerfile.cpu -t deepdoc-service:v1 .
docker run -d \
  -p 8000:8000 \
  -e DEEPDOC_MODEL_PATH=/app/resources/models \
  -e DEEPDOC_AUTO_DOWNLOAD=1 \
  -e DEEPDOC_DOWNLOAD_GROUPS=published \
  -v ./resources/models:/app/resources/models \
  deepdoc-service:v1
```

### Docker Compose

```bash
docker compose up -d
```

`docker-compose.yml` 只启动 `deepdoc` API 服务，并使用 `Dockerfile.cpu`。

GPU 镜像：

```bash
DEEPDOC_DOCKERFILE=Dockerfile.gpu DEEPDOC_ONNX_PROVIDER=auto docker compose up -d --build
```

也可以直接构建两个明确入口：

```bash
docker build -f Dockerfile.cpu -t deepdoc-standalone-cpu:0.1 .
docker build -f Dockerfile.gpu -t deepdoc-standalone-gpu:0.1 .
```

生产环境运行 CPU 镜像：

```bash
docker run -d \
  --name deepdoc \
  --env-file .env \
  -p 8000:8000 \
  -e DEEPDOC_MODEL_PATH=/app/resources/models \
  -v "$(pwd)/resources/models:/app/resources/models" \
  -v "$(pwd)/resources/artifacts:/app/resources/artifacts" \
  -v "$(pwd)/resources/logs:/app/resources/logs" \
  -v "$(pwd)/resources/temp:/app/resources/temp" \
  deepdoc-standalone-cpu:0.1
```

生产环境运行 GPU 镜像：

```bash
docker run -d \
  --name deepdoc \
  --gpus all \
  --env-file .env \
  -p 8000:8000 \
  -e DEEPDOC_MODEL_PATH=/app/resources/models \
  -e DEEPDOC_ONNX_PROVIDER=auto \
  -v "$(pwd)/resources/models:/app/resources/models" \
  -v "$(pwd)/resources/artifacts:/app/resources/artifacts" \
  -v "$(pwd)/resources/logs:/app/resources/logs" \
  -v "$(pwd)/resources/temp:/app/resources/temp" \
  deepdoc-standalone-gpu:0.1
```

如果模型仓库需要 token：

```bash
export HF_TOKEN=...
docker compose up -d --build
```

如果宿主机已预置模型，可关闭自动下载：

```bash
python download_models.py published
export DEEPDOC_AUTO_DOWNLOAD=0
docker compose up -d
```

## Artifact Backend

默认 backend 是 local。S3 / MinIO / compatible object storage：

```bash
pip install -e ".[artifact-s3]"
export DEEPDOC_ARTIFACT_BACKEND=s3
export DEEPDOC_ARTIFACT_BUCKET=deepdoc-artifacts
export DEEPDOC_ARTIFACT_PREFIX=prod
export DEEPDOC_ARTIFACT_ENDPOINT_URL=http://minio:9000
export DEEPDOC_ARTIFACT_REGION=us-east-1
export DEEPDOC_ARTIFACT_ACCESS_KEY_ID=minioadmin
export DEEPDOC_ARTIFACT_SECRET_ACCESS_KEY=minioadmin
export DEEPDOC_ARTIFACT_ADDRESSING_STYLE=path
export DEEPDOC_ARTIFACT_PUBLIC_BASE_URL=https://cdn.example.com/deepdoc-artifacts
```

## Ingest Publishing

File sink：

```bash
export DEEPDOC_INGEST_PUBLISHER=file
export DEEPDOC_INGEST_FILE_PATH=/data/deepdoc/published-ingest.jsonl
```

HTTP sink：

```bash
export DEEPDOC_INGEST_PUBLISHER=http
export DEEPDOC_INGEST_HTTP_URL=https://ingest.example.com/v1/documents
export DEEPDOC_INGEST_HTTP_AUTH_HEADER=Authorization
export DEEPDOC_INGEST_HTTP_AUTH_TOKEN='Bearer xxxxx'
export DEEPDOC_INGEST_HTTP_MAX_ATTEMPTS=3
export DEEPDOC_INGEST_HTTP_BACKOFF_SECONDS=1
export DEEPDOC_INGEST_HTTP_MAX_BACKOFF_SECONDS=8
export DEEPDOC_INGEST_HTTP_RETRY_STATUS_CODES=429,500,502,503,504
export DEEPDOC_INGEST_RETRY_BASE_DELAY_SECONDS=60
export DEEPDOC_INGEST_RETRY_MAX_DELAY_SECONDS=3600
export DEEPDOC_INGEST_RETRY_MAX_FAILURES=5
```

PostgreSQL sink：

```bash
export DEEPDOC_INGEST_PUBLISHER=postgres
export DEEPDOC_INGEST_PG_DSN=postgresql://user:password@postgres:5432/deepdoc
export DEEPDOC_INGEST_PG_SCHEMA=deepdoc_ingest
export DEEPDOC_INGEST_PG_CONNECT_TIMEOUT=10
```

## Advanced Runtime Configuration

CPU pipeline 现代化开关默认保持历史行为，不会静默改变解析结果：

| 变量 | 默认值 | 可选值 | 作用 | 需要的模型组 |
|---|---|---|---|---|
| `DEEPDOC_OCR_VERSION` | `v4` | `v4` / `v5` | 切换 deepdoc 本地 OCR 检测/识别模型 | `core` / `core_v5` |
| `DEEPDOC_REC_IMAGE_SHAPE` | 空 | `C,H,W` | 通用 OCR 识别输入尺寸覆盖，空值保持默认 `3,48,320` | `core` / `core_v5` |
| `DEEPDOC_OCR_V4_REC_IMAGE_SHAPE` | 空 | `C,H,W` | v4 OCR 识别输入尺寸覆盖，优先于 `DEEPDOC_REC_IMAGE_SHAPE` | `core` |
| `DEEPDOC_OCR_V5_REC_IMAGE_SHAPE` | 空 | `C,H,W` | v5 OCR 识别输入尺寸覆盖，优先于 `DEEPDOC_REC_IMAGE_SHAPE` | `core_v5` |
| `DEEPDOC_LAYOUT_ENGINE` | `legacy` | `legacy` / `ppdoclayout` | 切换 deepdoc PDF OCR + Layout 路径的版面识别后端 | `core` / `layout_v2` |
| `DEEPDOC_TABLE_ENGINE` | `tatr` | `tatr` / `rapidtable` | 切换 deepdoc PDF 表格识别后端；`rapidtable` 单表失败回退 `tatr` | `core` / `table_v2` |
| `DEEPDOC_FORMULA_MODE` | `rapidlatex` | `rapidlatex` / `pp_formula_net_s` | 切换公式识别模型模式 | `formula` / `formula_v2` |
| `DEEPDOC_READING_ORDER_STRATEGY` | `legacy` | `legacy` / `rules` | 切换阅读顺序和跨页段落规则；`rules` 启用多栏排序、重复页眉页脚去重、caption 绑定和标题/缩进误合并拦截 | 无 |

启用新模型前先下载或预置对应模型组：

```bash
python download_models.py core_v5
python download_models.py layout_v2
python download_models.py table_v2
python download_models.py formula_v2

# 或一次下载已发布模型组
python download_models.py published

# 全部声明模型组；包含未发布的 handwriting 时需要自行提供 rec_handwriting.onnx
python download_models.py all
```

`DEEPDOC_FORMULA_MODE=pp_formula_net_s` 会通过 PaddleX PP-FormulaNet-S 适配器执行公式识别，需安装 `pip install -e ".[formula-v2]"` 并预置 `formula_v2` 模型组。默认仍为 `DEEPDOC_FORMULA_MODE=rapidlatex`；切换生产默认前必须先补真实业务/论文公式子集 A/B 指标和 CPU 耗时报告。

需要启用 ONNX INT8 动态量化时，先生成量化模型：

```bash
python tools/quantize_models.py --model-dir resources/models
```

该脚本会递归扫描 `resources/models` 及 `layout/`、`formula/`、`table/` 等模型组子目录，为非 INT8 `.onnx` 模型生成同名 `.int8.onnx` 文件，不覆盖原始模型。默认批量扫描会跳过高风险序列/decoder 模型（如 `rec.onnx`、`rec_v5.onnx`、`formula/decoder.onnx`），这些模型需要先做逐模块精度校准；确认达标后再显式传 `--include-risky-sequence-models`，或用 `--model path/to/model.onnx` 单独量化。服务默认 `DEEPDOC_QUANT=fp32`，继续加载原始模型；显式设置 `DEEPDOC_QUANT=int8` 后加载 `.int8.onnx`。如果量化模型不存在，服务会直接报错而不是静默回退。INT8 动态量化路径当前固定使用 CPU Execution Provider，适合 CPU 推理或 GPU 不稳定时的轻量化部署。

ONNX Runtime provider 默认 `DEEPDOC_ONNX_PROVIDER=auto`，会按现有逻辑在 CUDA 可用时使用 `CUDAExecutionProvider`，否则回退 `CPUExecutionProvider`。需要启用 TensorRT 时显式设置 `DEEPDOC_ONNX_PROVIDER=tensorrt`；服务会要求 ONNX Runtime 同时提供 `TensorrtExecutionProvider` 和 `CUDAExecutionProvider`，并使用 TensorRT → CUDA → CPU 的 provider fallback 顺序。TensorRT engine cache 默认写入 `DEEPDOC_TENSORRT_CACHE_DIR=/app/resources/temp/tensorrt_engine_cache`，可用 `DEEPDOC_TENSORRT_FP16=1` 开启 FP16，并通过 `DEEPDOC_TENSORRT_MAX_WORKSPACE_SIZE` 控制 workspace size。默认不会自动切换到 TensorRT，避免普通 GPU 镜像在缺 TensorRT provider 时启动失败。

OCR 识别阶段默认启用动态批处理：当 rec 模型输入宽度为动态维度时，会按文本行目标宽度 bucket 分组，减少同批次被极宽文本行拖高的 padding 宽度；固定宽度模型保持原有固定 batch 行为。可通过 `DEEPDOC_REC_DYNAMIC_BATCHING=0` 关闭；`DEEPDOC_REC_WIDTH_BUCKET_STEP` 控制 bucket 粒度，默认 `64`。Layout 识别会把同一个 `batch_size` 内可合并的多页输入合并成一次 ONNX Runtime 调用，不能合并时自动回退到逐页推理。

手写体回退默认关闭。设置 `DEEPDOC_HANDWRITING_FALLBACK=1` 后，OCR rec 主模型分数低于 `DEEPDOC_HANDWRITING_FALLBACK_THRESHOLD` 的文本行会懒加载手写识别模型重跑；默认模型名为 `DEEPDOC_HANDWRITING_MODEL_NAME=rec_handwriting`，对应模型文件 `rec_handwriting.onnx`。只有手写结果分数至少高出 `DEEPDOC_HANDWRITING_MIN_SCORE_DELTA` 且文本非空时才替换主识别结果。该能力需要先下载 `handwriting` 模型组：

```bash
python download_models.py handwriting
```

API 服务和异步解析 worker 启动时默认执行一次 OCR/Layout 预热：加载模型后跑一张白图触发 ONNX Runtime/CUDA kernel 首次执行，降低首个真实解析请求的冷启动抖动。可通过 `DEEPDOC_MODEL_WARMUP=0` 关闭；白图尺寸可用 `DEEPDOC_MODEL_WARMUP_IMAGE_SIZE` 调整，默认 `64`。

OCR 模型 session 使用 LRU 缓存并支持空闲释放，避免多 worker 或多设备配置下长期持有过多显存/内存。`DEEPDOC_MODEL_CACHE_MAX_SIZE` 控制单进程最大缓存模型数，默认 `8`；`DEEPDOC_MODEL_CACHE_IDLE_TTL_SECONDS` 控制空闲 TTL，默认 `3600` 秒。缓存命中会刷新最近访问时间；超过容量时淘汰最久未使用模型。

## Model Pack Publishing

默认模型仓库：

- `qwqqwq/deepdoc-standalone`

生成 manifest：

```bash
python tools/publish_models_to_hf.py --groups core --write-manifest-only
```

发布模型包：

```bash
export HF_TOKEN=<your_token>
python tools/publish_models_to_hf.py --groups all
```

校验远端文件：

```bash
export HF_TOKEN=<your_token>
python tools/ci/verify_hf_models.py --groups all
```

## Related Docs

- [API Reference](API.md)
- [Evaluation Guide](EVALUATION.md)
- [Operations Guide](OPERATIONS.md)
- [Parser Engine Strategy](PARSER_ENGINE_STRATEGY.md)
