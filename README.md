# DocPilot Standalone

DocPilot Standalone 是一个文档解析服务，负责把 PDF、Office、HTML、Markdown、TXT 等文件解析成可直接消费的 Markdown、结构化 JSON、chunks 和图片/表格/印章/公式等资产。

它不负责问答、向量化、检索或回答生成。服务边界就是文档解析和结构化输出。

## Features

- 多格式解析：PDF、DOCX、XLSX、PPTX、HTML、JSON、Markdown、TXT、CSV、RTF、ODT、EML、MSG、XML、ZIP、EPUB、CAJ
- 多 PDF 引擎：`deepdoc`、`paddleocr_vl`、`mineru`、`plain`，非 PDF 可显式使用 `markitdown`
- 结构化产物：`markdown.md`、`structured.json`、`chunks.jsonl`、`ingest.jsonl`、`assets/`
- 可选能力：公式识别、印章识别、二维码/条形码识别、手写体回退、artifact 持久化、异步任务、ingest 发布
- 部署形态：本地 Python、CPU/GPU Docker、Gradio 调试控制台

## Quick Start

### 1. Install

```bash
conda create -n deepdoc python=3.10
conda activate deepdoc

pip install -e .
```

如需本地控制台：

```bash
pip install -e ".[gradio]"
```

### 2. Download Models

```bash
python download_models.py published
```

最小本地 PDF 解析模型组：

```bash
python download_models.py core
```

### 3. Run

启动 API：

```bash
python main.py
```

开发调试也可以用：

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

启动 Gradio 控制台：

```bash
python gradio_app.py
```

### 4. Smoke Test

```bash
curl -X POST "http://localhost:8000/api/v1/parse" \
  -F "file=@/path/to/document.pdf"
```

## Documentation

- [Deployment Guide](docs/DEPLOYMENT.md)
- [Operations Guide](docs/OPERATIONS.md)
- [API Reference](docs/API.md)
- [Parser Engine Strategy](docs/PARSER_ENGINE_STRATEGY.md)

部署、模型下载、Docker、发布模型包、运行参数、运维接口、artifact backend、ingest 发布等细节都放在 `docs/`，README 只保留入口信息。

## Notes

- 默认模型目录：`resources/models`
- 容器内模型目录：`/app/resources/models`
- GPU 镜像依赖宿主机 CUDA 驱动和可用的 ONNX Runtime GPU provider
- 部分非 PDF 路径依赖 Java / Tika，请保证 `java` 在 PATH 中

## License

Apache 2.0
