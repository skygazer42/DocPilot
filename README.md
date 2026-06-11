# DocPilot Standalone

DocPilot Standalone 是一个文档解析服务，负责把 PDF、Office、HTML、Markdown、TXT 等文件解析成可直接消费的 Markdown、结构化 JSON、chunks 和图片/表格/印章/公式等资产。

它不负责问答、向量化、检索或回答生成。服务边界就是文档解析和结构化输出。

![DocPilot parsing pipeline](docs/assets/docpilot-pipeline.png)

## Features

- 文档输入：PDF、Office、HTML、Markdown、TXT 及常见归档/邮件格式
- PDF 路径：`deepdoc`、`paddleocr_vl`、`mineru`、`plain`
- 输出结果：Markdown、structured JSON、chunks、ingest records、assets
- 运行方式：本地 Python、CPU/GPU Docker、Gradio 控制台

## Quick Start

### 1. Install

```bash
conda create -n deepdoc python=3.10
conda activate deepdoc

pip install -e .
pip install -e ".[gradio]"
```

### 2. Download Models

```bash
python download_models.py core
```

### 3. Run

```bash
python main.py
python gradio_app.py
```

### 4. Smoke Test

```bash
curl -X POST "http://localhost:8000/api/v1/parse" \
  -F "file=@/path/to/document.pdf"
```

## Documentation

| Doc | Scope |
|---|---|
| [Deployment Guide](docs/DEPLOYMENT.md) | 安装、模型下载、Docker、运行参数 |
| [Evaluation Guide](docs/EVALUATION.md) | dataset contract、license gate、A/B、readiness、profile |
| [Operations Guide](docs/OPERATIONS.md) | health、ready、metrics、审计、自检、日志 |
| [API Reference](docs/API.md) | 解析接口、异步任务、artifact、ingest |
| [Parser Engine Strategy](docs/PARSER_ENGINE_STRATEGY.md) | 引擎选择与兼容策略 |

部署、评测和运维细节都放在 `docs/`，README 只保留入口信息。

## Notes

- 默认模型目录：`resources/models`
- 容器内模型目录：`/app/resources/models`
- GPU 镜像依赖宿主机 CUDA 驱动和可用的 ONNX Runtime GPU provider
- 部分非 PDF 路径依赖 Java / Tika，请保证 `java` 在 PATH 中

## License

Apache 2.0
