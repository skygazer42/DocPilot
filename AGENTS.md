# AGENTS Guide for `deepdoc-standalone`

面向 AI 编码代理的指引文件。目标：最小化、正确、可验证的变更。

## 1) 范围与优先级

- 范围：`/work/project-kingdon/kd-brain/deepdoc-standalone`。
- 优先级：用户直接指令 > 本文件 > 上层/全局默认值。
- 变更应严格限定在用户任务范围内。

## 2) 仓库结构

### 入口点

| 文件 | 用途 | 启动方式 |
|---|---|---|
| `main.py` | Flask REST API（`app`）；`__main__` 用 gevent WSGIServer | `python main.py` 或 `uvicorn main:app` |
| `gradio_app.py` | Gradio PDF 解析交互界面 | `python gradio_app.py`（端口 7860） |

### 模块边界

- `deepdoc/vision/`：OCR (`ocr.py`)、版面识别 (`layout_recognizer.py`)、表格结构识别 (`table_structure_recognizer.py`)、ONNX 推理
- `deepdoc/parser/`：各格式解析器（pdf/docx/excel/ppt/html/json/md/txt）+ 远程引擎适配器（`paddleocr_parser.py`、`mineru_parser.py`、`docling_parser.py`、`tcadp_parser.py`）
- `common/`：配置 (`setting.py`)、自定义日志 (`log.py`)、文件/字符串/markdown/token/chunk/artifact 工具函数
- `tools/`：独立脚本（如 `pdf_to_markdown.py`，纯文本层提取）

### PDF 解析引擎选择

`main.py` 使用两个 dict 实现动态解析器路由：
- `PARSER_IMPORTS`：文件扩展名 → `(module, class)` 映射
- `PDF_PARSER_OVERRIDES`：引擎名 → `(module, class)` 覆盖（仅 PDF）
- 支持的引擎：`deepdoc`（默认，本地 ONNX）、`paddleocr_vl`（兼容旧别名 `paddleocr`）、`markitdown`、`mineru`、`plain`
- 新增 PDF 引擎只需在这两个 dict 中加映射 + 对应 parser 模块

### 构建与打包

- `pyproject.toml`（`setuptools.build_meta`），Python `>=3.10,<3.13`
- 默认依赖使用 CPU `onnxruntime`；GPU 环境使用 `pip install -e ".[gpu]"` 或 `Dockerfile.gpu`
- 可选依赖：`pip install -e ".[gradio]"` 安装 Gradio UI

## 3) 开发命令

### 环境搭建

```bash
conda create -n deepdoc python=3.10 && conda activate deepdoc
uv pip install -e .          # 或 pip install -e .
uv pip install -e ".[gradio]" # 如需 Gradio UI
```

### 模型下载（首次）

```bash
# 默认下载到 resources/models/（由 setting.MODELS_DIR 决定）
python download_models.py published
# 最小 deepdoc 默认解析模型组：
python download_models.py core
# 如需自定义路径，先设置环境变量：
export DEEPDOC_MODEL_PATH=./models
```

注意：`setting.MODELS_DIR` 的值为 `<项目根>/resources/models`。`main.py` 启动时使用 `os.environ.setdefault("DEEPDOC_MODEL_PATH", setting.MODELS_DIR)`，因此显式设置的 `DEEPDOC_MODEL_PATH`（例如 Docker 中的 `/app/resources/models`）会被保留。

### 运行服务

```bash
# API 服务（gevent WSGI）
python main.py                                     # 默认 0.0.0.0:8000

# API 服务（uvicorn，开发调试）
uvicorn main:app --host 0.0.0.0 --port 8000

# Gradio UI
python gradio_app.py                               # 默认 0.0.0.0:7860
```

### Docker

```bash
# 单服务 CPU 镜像
docker build -f Dockerfile.cpu -t deepdoc-service:v1 .
docker run -d -p 8000:8000 -v ./resources/models:/app/resources/models deepdoc-service:v1
```

`docker-compose.yml` 只启动 `deepdoc`（API :8000）。Docker 部署只保留 `Dockerfile.cpu` 和 `Dockerfile.gpu` 两个入口，容器使用 `/app/resources/models` 作为统一模型目录，并可在启动时按 `DEEPDOC_DOWNLOAD_GROUPS` 自动补齐缺失模型。

### Lint（最佳实践）

```bash
uv run ruff check .
```

无 `.ruff.toml`，`pyproject.toml` 中也无 `[tool.ruff]` 配置。

### 验证（无 pytest）

本仓库**无标准测试套件**，验证方式：

```bash
# 脚本级验证
python deepdoc/vision/t_ocr.py --inputs /path/to/image.png --output_dir ./debug_ocr
python deepdoc/vision/t_recognizer.py --inputs /path/to/file.pdf --mode layout --output_dir ./debug_layout
python deepdoc/vision/t_recognizer.py --inputs /path/to/image.png --mode tsr --output_dir ./debug_tsr

# API 冒烟测试
curl -X POST "http://localhost:8000/api/v1/ocr" -F "file=@sample.png"
curl -X POST "http://localhost:8000/api/v1/parse" -F "file=@sample.pdf"
```

## 4) 关键约定

### 日志

使用仓库自定义 logger，**不要**用 `logging.getLogger(__name__)`：

```python
from common import logger
logger.info("message %s", value)
```

`logger` 是 `common.log.Log` 单例，自动按日期/小时轮转写入 `resources/logs/`。

### 配置

- 路径常量在 `common/setting.py`（`MODELS_DIR`、`WORK_DIR`、`LOG_DIR`、`TIKTOKEN_CACHE_DIR`）
- 运行时配置通过环境变量 + `.env`（`python-dotenv`，`load_dotenv(override=False)`）
- `.env` 中含 `SECRET_ACCESS_KEY`——不要提交真实密钥

### 重要环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `DEEPDOC_MODEL_PATH` | `resources/models` | 模型路径（被 `main.py` 启动时覆盖为 `setting.MODELS_DIR`） |
| `DEEPDOC_PDF_PARSER` | `deepdoc` | 默认 PDF 引擎 |
| `DEEPDOC_PDF_MAX_PAGES` | `10` | deepdoc 引擎最大解析页数 |
| `DEEPDOC_REQUEST_TIMEOUT` | `600` | 远程引擎请求超时（秒） |
| `DEEPDOC_CLEANUP_OUTPUT` | `1` | 是否清理临时文件 |
| `SECRET_ACCESS_KEY` | - | API 鉴权密钥（可选） |
| `DEEPDOC_LAYOUT_MODEL` | `manual` | deepdoc 版面模型：`manual/paper/laws/general` |
| `DEEPDOC_TABLE_ENGINE` | `tatr` | deepdoc 表格识别引擎：`tatr`(几何拼装) / `rapidtable`(SLANet-plus ONNX)；缺依赖或模型时自动回退 `tatr` |

### 代码风格

- 4 空格缩进，目标行宽 120
- 新代码优先使用双引号
- 类名 `PascalCase`、函数/变量 `snake_case`、常量 `SCREAMING_SNAKE_CASE`
- 新公共函数添加类型注解，优先 `str | None` 而非 `Optional[str]`
- `docs/coding_style.md` 来自另一个项目（Dify），**不适用于本仓库**——勿照搬其 SQLAlchemy/Pydantic/配置管理规范

### API 行为不变性

- 保持请求参数行为稳定（`parser_engine`、`compute_device`、`return_images`、`strict_text`）
- 保持环境变量 fallback 行为
- 不静默变更默认值；行为变更需更新 `docs/API.md`

## 5) 架构安全约束

- 尊重模块边界：`vision`、`parser`、`common` 之间避免交叉耦合
- 避免循环导入
- `main.py` 中的解析器选择/模型初始化流程是核心路径，修改需谨慎
- 保持 `common/markdown_utils.py` 中的文本后处理和清洗流程
- 保持文件扩展名校验和临时文件清理行为

## 6) 已知反模式

1. 硬编码机器特定路径
2. 执行与用户任务无关的大规模重构
3. API 端点返回非 JSON 或裸异常
4. 静默变更行为但不更新文档
5. 将 `docs/coding_style.md`（Dify 项目规范）的约束应用到本仓库

## 7) 代理执行清单

1. 编辑前先读取相关代码
2. 优先最小化、局部变更
3. 至少运行一项验证：脚本级测试 / API 冒烟 / `uv run ruff check .`
4. 报告已验证内容和未验证内容
5. 不要声称存在 pytest 覆盖（当前没有）
6. 未经用户明确要求，不提交/推送
