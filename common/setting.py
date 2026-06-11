#!/usr/bin/env python
# _*_ coding:utf-8 _*_

import os

BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
DEFAULT_MODELS_DIR = os.path.join(BASE_DIR, "resources", "models")

# 日志目录
LOG_DIR = os.path.join(BASE_DIR, "resources", "logs")
# 模型文件存放目录
MODELS_DIR = os.path.abspath(os.environ.get("DEEPDOC_MODEL_PATH", DEFAULT_MODELS_DIR))
# 结构化解析产物目录
ARTIFACTS_DIR = os.path.abspath(
    os.environ.get("DEEPDOC_ARTIFACTS_DIR", os.path.join(BASE_DIR, "resources", "artifacts"))
)
# 异步任务目录
TASKS_DIR = os.path.abspath(
    os.environ.get("DEEPDOC_TASKS_DIR", os.path.join(BASE_DIR, "resources", "tasks"))
)
# 审计日志目录
AUDIT_DIR = os.path.abspath(
    os.environ.get("DEEPDOC_AUDIT_DIR", os.path.join(BASE_DIR, "resources", "audit"))
)
# 生产自检目录
SELF_CHECKS_DIR = os.path.abspath(
    os.environ.get("DEEPDOC_SELF_CHECKS_DIR", os.path.join(BASE_DIR, "resources", "self_checks"))
)
# 镜像内构建元数据文件
BUILD_INFO_PATH = os.path.abspath(
    os.environ.get("DEEPDOC_BUILD_INFO_PATH", os.path.join(BASE_DIR, "build-info.json"))
)
# tiktoken缓存目录
TIKTOKEN_CACHE_DIR = os.path.join(BASE_DIR, "resources", "tiktoken_cache")
# 临时目录
WORK_DIR = os.path.join(BASE_DIR, "resources", "temp")

# GPU 并行设备数（运行时可通过 torch 自动检测更新）
PARALLEL_DEVICES = int(os.environ.get("PARALLEL_DEVICES", 0))

# 是否使用 Infinity 引擎（standalone 模式下默认关闭）
DOC_ENGINE_INFINITY = os.environ.get("DOC_ENGINE_INFINITY", "").lower() in ("true", "1", "yes")
