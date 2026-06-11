# DocPilot Standalone 安全审计报告

日期: 2026-01-23
范围: deepdoc-standalone (静态审计)

## 1. 概览
本报告基于静态代码审查与配置核查，聚焦身份认证、文件处理、反序列化、异常处理和依赖风险。

## 2. 范围与方法
- 范围文件: `.env`, `main.py`, `deepdoc/parser/mineru_parser.py`, `deepdoc/vision/__init__.py`
- 方法: 手动代码审查 + 关键字扫描

## 3. 发现摘要

| ID | 严重级别 | 问题 | 位置 | 建议 |
| :-- | :-- | :-- | :-- | :-- |
| F-02 | High | ZIP 路径穿越 (Zip Slip) | `deepdoc/parser/mineru_parser.py:141-173` | 解压前校验路径 | 
| F-03 | High | 明文敏感信息提交 | `.env:10` | 移除密钥并轮换 | 
| F-04 | Medium | 密钥缺失时跳过鉴权 | `main.py:59-64` | 启动时强制鉴权配置 | 
| F-05 | Medium | 上传校验与隔离不足 | `main.py:71-102`, `main.py:154-157` | MIME/魔数校验、限制解码 | 
| F-06 | Medium | CORS 全开放 | `main.py:33-34` | 限制允许域名 |
| F-07 | Low | 详细错误/堆栈泄露 | `main.py:187-189`, `main.py:332`, `main.py:382`, `deepdoc/vision/__init__.py:69` | 统一错误响应 |
| F-08 | Low | 健康检查泄露内部路径 | `main.py:123-130` | 限制返回内容/鉴权 |

## 4. 详细发现

### F-02 ZIP 路径穿越 (High)
- 描述: MinerU ZIP 解压在存在 root 目录或无 root 目录时均未验证路径安全性。
- 风险: ZIP 内包含 `../` 可覆盖任意路径。
- 证据: `deepdoc/parser/mineru_parser.py:141-173` (含 `extractall` 与 `os.path.join`).
- 修复建议: 解压前对每个成员执行 `os.path.abspath` + `os.path.commonpath` 校验，拒绝绝对路径与 `..`。

### F-03 明文敏感信息提交 (High)
- 描述: `.env` 内含真实 `SECRET_ACCESS_KEY`，并包含内部地址与路径。
- 风险: 代码泄露即密钥泄露，攻击者可直接访问 API。
- 证据: `.env:10`。
- 修复建议: 移除 `.env` 并加入 `.gitignore`，立刻轮换密钥，部署侧使用 Secrets/KMS。

### F-04 密钥缺失时跳过鉴权 (Medium)
- 描述: `SECRET_ACCESS_KEY` 未设置时直接放行请求。
- 风险: 生产环境配置错误会导致鉴权失效。
- 证据: `main.py:59-64`。
- 修复建议: 启动时强制校验密钥存在，或提供显式开关 (如 `DEEPDOC_AUTH_DISABLED=false`)。

### F-05 上传校验与隔离不足 (Medium)
- 描述: 主要依赖扩展名验证，未校验 MIME/魔数；图片解码未限制像素规模；文件落盘未隔离。
- 风险: 伪造扩展名导致异常或 DoS；超大图片触发解码爆炸。
- 证据: `main.py:71-102`, `main.py:154-157`。
- 修复建议: 引入 `python-magic` 校验 MIME，限制 `PIL.Image.MAX_IMAGE_PIXELS`，使用专用受限目录或沙箱。

### F-06 CORS 全开放 (Medium)
- 描述: 默认 `CORS(app)` 未限制来源。
- 风险: 恶意站点可通过浏览器调用接口。
- 证据: `main.py:33-34`。
- 修复建议: 限制 `origins` 白名单，生产环境关闭通配符。

### F-07 详细错误/堆栈泄露 (Low)
- 描述: 异常通过 `traceback.print_exc()` 输出并直接返回 `str(e)`。
- 风险: 泄露内部路径、依赖与实现细节。
- 证据: `main.py:187-189`, `main.py:332`, `main.py:382`, `deepdoc/vision/__init__.py:69`。
- 修复建议: 返回统一错误码与简短描述，详细日志仅写入服务端。

### F-08 健康检查泄露内部路径 (Low)
- 描述: `/health` 返回 `model_path` 与加载状态。
- 风险: 暴露部署路径与组件状态。
- 证据: `main.py:123-130`。
- 修复建议: 仅返回简单状态或加入鉴权。

## 5. 依赖风险 (基于 pip-audit 结果)
建议升级以下依赖以修复已知漏洞:

| 依赖包 | 建议最低版本 |
| :-- | :-- |
| `urllib3` | `>= 2.6.3` |
| `requests` | `>= 2.32.4` |
| `brotli` | `>= 1.2.0` |
| `uv` | `>= 0.9.6` |
| `wheel` | `>= 0.46.2` |
| `pip` | `>= 25.3` |

## 6. 修复优先级
1. P0: F-02
2. P1: F-03, F-04
3. P2: F-05, F-06, F-07, F-08

## 7. 备注
本报告基于静态审计，未执行动态渗透测试或运行期验证。建议在 CI 中引入 Bandit/Trivy 等自动化扫描。
