import json
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch


class ApiBoundaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        openapi_path = Path(__file__).resolve().parents[1] / "openapi.json"
        cls.openapi = json.loads(openapi_path.read_text(encoding="utf-8"))
        cls.paths = set(cls.openapi.get("paths", {}))

    @staticmethod
    def _parse_sse_events(body: str) -> list[dict[str, object]]:
        events: list[dict[str, object]] = []
        current_event: str | None = None
        current_data_lines: list[str] = []

        def flush() -> None:
            nonlocal current_event, current_data_lines
            if not current_event:
                current_data_lines = []
                return
            payload_text = "\n".join(current_data_lines).strip()
            payload: object = None
            if payload_text:
                payload = json.loads(payload_text)
            events.append({"event": current_event, "data": payload})
            current_event = None
            current_data_lines = []

        for line in body.splitlines():
            if not line.strip():
                flush()
                continue
            if line.startswith("event: "):
                current_event = line[len("event: ") :].strip()
                continue
            if line.startswith("data: "):
                current_data_lines.append(line[len("data: ") :])
        flush()
        return events

    def test_public_api_has_no_rag_or_chat_provider_surface(self):
        forbidden_prefixes = ("/api/v1/rag", "/api/v1/chat-provider")
        forbidden_paths = {"/api/v1/ingest/embeddings/backfill"}
        leaked_paths = sorted(
            path
            for path in self.paths
            if path.startswith(forbidden_prefixes) or path in forbidden_paths
        )
        self.assertEqual([], leaked_paths)

    def test_document_parse_artifact_and_chunk_surfaces_remain_public(self):
        required_paths = {
            "/api/v1/parse",
            "/api/v1/parse/stream",
            "/api/v1/parse/async",
            "/api/v1/artifacts/{parse_id}/structured",
            "/api/v1/artifacts/{parse_id}/markdown",
            "/api/v1/artifacts/{parse_id}/chunks",
        }
        missing_paths = sorted(required_paths - self.paths)
        self.assertEqual([], missing_paths)

    def test_chunk_strategy_is_public_and_flows_into_artifact_builders(self):
        repo_root = Path(__file__).resolve().parents[1]
        main_source = (repo_root / "main.py").read_text(encoding="utf-8")
        builders_source = (repo_root / "common/parse_builders.py").read_text(encoding="utf-8")
        api_doc = (repo_root / "docs/API.md").read_text(encoding="utf-8")
        parse_schema = (
            self.openapi["paths"]["/api/v1/parse"]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]
        )
        async_schema = (
            self.openapi["paths"]["/api/v1/parse/async"]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]
        )

        self.assertIn('request.form.get("chunk_strategy")', main_source)
        self.assertIn('"chunk_strategy": chunk_strategy', main_source)
        self.assertIn('"chunk_strategy": str(parse_options.get("chunk_strategy")', main_source)
        self.assertIn("chunk_strategy=chunk_strategy", main_source)
        self.assertIn("chunk_strategy: str = DEFAULT_CHUNK_STRATEGY", builders_source)
        self.assertIn("strategy=chunk_strategy", builders_source)
        self.assertIn("structure_aware", api_doc)
        self.assertIn("page_aware", api_doc)
        self.assertIn("asset_aware", api_doc)
        for schema in (parse_schema, async_schema):
            chunk_strategy = schema["properties"].get("chunk_strategy")
            self.assertIsInstance(chunk_strategy, dict)
            self.assertEqual(["structure_aware", "page_aware", "asset_aware"], chunk_strategy.get("enum"))

    def test_formula_and_seal_are_deepdoc_only_optional_parse_capabilities(self):
        repo_root = Path(__file__).resolve().parents[1]
        main_source = (repo_root / "main.py").read_text(encoding="utf-8")
        pdf_source = (repo_root / "deepdoc/parser/pdf_parser.py").read_text(encoding="utf-8")
        model_store_source = (repo_root / "common/model_store.py").read_text(encoding="utf-8")
        api_doc = (repo_root / "docs/API.md").read_text(encoding="utf-8")
        roadmap = (repo_root / "plans/optimization-roadmap.md").read_text(encoding="utf-8")

        self.assertIn('request.form.get("enable_formula")', main_source)
        self.assertIn('request.form.get("enable_seal")', main_source)
        self.assertIn('enable_formula and parser_engine != "deepdoc"', main_source)
        self.assertIn('enable_seal and parser_engine != "deepdoc"', main_source)
        self.assertIn('parser._recognize_formulas(zoomin)', main_source)
        self.assertIn('parser._recognize_seals(zoomin)', main_source)
        self.assertIn("def _recognize_formulas", pdf_source)
        self.assertIn("def _recognize_seals", pdf_source)
        self.assertIn('"formula"', model_store_source)
        self.assertIn('"seal"', model_store_source)
        self.assertIn("enable_formula", api_doc)
        self.assertIn("enable_seal", api_doc)
        self.assertIn("| A1 | **公式识别(LaTeX OCR)** | 已落地", roadmap)
        self.assertIn("| A2 | **印章识别** | 已落地", roadmap)

    def test_docs_metrics_health_and_readiness_surfaces_are_available(self):
        import main

        client = main.app.test_client()

        docs_response = client.get("/docs/")
        self.assertEqual(200, docs_response.status_code)
        docs_html = docs_response.get_data(as_text=True).lower()
        self.assertIn("swagger", docs_html)

        openapi_response = client.get("/openapi.json")
        docs_openapi_response = client.get("/docs/openapi.json")
        for response in (openapi_response, docs_openapi_response):
            self.assertEqual(200, response.status_code)
            payload = response.get_json()
            self.assertIsInstance(payload, dict)
            self.assertEqual("3.1.0", payload.get("openapi"))
            self.assertIn("/api/v1/parse", payload.get("paths", {}))
            self.assertIn("/health", payload.get("paths", {}))
            self.assertIn("/ready", payload.get("paths", {}))

        metrics_response = client.get("/metrics")
        self.assertEqual(200, metrics_response.status_code)
        self.assertIn("text/plain", metrics_response.content_type)
        metrics_body = metrics_response.get_data(as_text=True)
        self.assertTrue(
            "# metrics disabled" in metrics_body
            or "deepdoc_http_requests_total" in metrics_body
        )
        if "# metrics disabled" not in metrics_body:
            self.assertIn("deepdoc_backend_info", metrics_body)

        health_response = client.get("/health")
        self.assertEqual(200, health_response.status_code)
        health = health_response.get_json()
        for field in (
            "status",
            "api_docs",
            "artifact_backend",
            "request_protection",
            "async_tasks",
            "self_checks",
            "retention_janitor",
        ):
            self.assertIn(field, health)
        self.assertIn("/docs", health["api_docs"]["docs_url"])
        self.assertIn("/openapi.json", health["api_docs"]["openapi_url"])

        ready_response = client.get("/ready")
        self.assertIn(ready_response.status_code, {200, 503})
        readiness = ready_response.get_json()
        for field in (
            "status",
            "required_model_groups",
            "missing_model_files",
            "api_docs",
            "artifact_backend",
            "ingest_store",
            "request_protection",
            "async_tasks",
        ):
            self.assertIn(field, readiness)

    def test_roadmap_marks_verified_operational_surfaces_as_done(self):
        repo_root = Path(__file__).resolve().parents[1]
        roadmap = (repo_root / "plans/optimization-roadmap.md").read_text(encoding="utf-8")

        self.assertIn("| F1 | **Prometheus 指标** | 已落地", roadmap)
        self.assertIn("| G6 | **健康检查增强** | 已落地", roadmap)
        self.assertIn("| H1 | **Swagger UI** | 已落地", roadmap)
        self.assertIn("| G4 | **限流 + 配额** | 已落地", roadmap)
        self.assertIn("| C5 | **并发控制** | 已落地：请求级 admission/in-flight 准入控制", roadmap)
        self.assertIn("| C3 | **流式输出 SSE** | 已落地：`POST /api/v1/parse/stream`", roadmap)
        self.assertIn("| C6 | **OCR/Layout 全局单例** | 已落地", roadmap)
        self.assertIn("| H2 | **SSE 流式解析端点** | 已落地：同步 `POST /api/v1/parse/stream`", roadmap)
        self.assertIn("| H4 | **批量上传 + 异步进度** | 已落地：同步 `/api/v1/parse` 支持多文件上传", roadmap)
        self.assertNotIn("同步 `/api/v1/parse/stream` 仍归 C3/H2", roadmap)
        self.assertNotIn("**C6 全局单例**待做", roadmap)

    def test_request_guard_routes_rate_limits_admission_and_health_surfaces(self):
        import main

        repo_root = Path(__file__).resolve().parents[1]
        main_source = (repo_root / "main.py").read_text(encoding="utf-8")
        ratelimit_source = (repo_root / "common/ratelimit.py").read_text(encoding="utf-8")

        self.assertIn("REQUEST_RATE_LIMITER = create_request_rate_limiter()", main_source)
        self.assertIn("INFLIGHT_ADMISSION = create_inflight_admission_controller()", main_source)
        self.assertIn('if path.startswith("/api/v1/parse"):', main_source)
        self.assertIn('if path.startswith("/api/v1/artifacts"):', main_source)
        self.assertIn('if path.startswith("/api/v1/ingest"):', main_source)
        self.assertIn("REQUEST_RATE_LIMITER.evaluate(", main_source)
        self.assertIn("INFLIGHT_ADMISSION.acquire(scope)", main_source)
        self.assertIn("response.status_code = rate_limit.denied_decision.rule.status_code", main_source)
        self.assertIn("response.status_code = 503", main_source)
        self.assertIn("request_protection = _request_protection_health()", main_source)
        self.assertIn("RedisRateLimitStore", ratelimit_source)
        self.assertIn("InMemoryRateLimitStore", ratelimit_source)
        self.assertIn("DEEPDOC_RATE_LIMIT_PARSE_BYTES", ratelimit_source)
        self.assertIn("DEEPDOC_MAX_INFLIGHT_PARSE", ratelimit_source)

        client = main.app.test_client()
        for path in ("/health", "/ready"):
            response = client.get(path)
            self.assertIn(response.status_code, {200, 503})
            payload = response.get_json()
            self.assertIn("request_protection", payload)
            self.assertIn("rate_limit", payload["request_protection"])
            self.assertIn("admission", payload["request_protection"])

    def test_parse_endpoint_accepts_multiple_files_without_rag_side_effects(self):
        import main

        def fake_parse_single_file(file, parse_options):
            return {
                "filename": file.filename,
                "markdown": f"# {file.filename}",
                "parser_engine": parse_options.get("parser_engine"),
            }

        with patch.object(main, "_parse_single_file", side_effect=fake_parse_single_file), patch.object(
            main, "_append_ops_audit_event", return_value=None
        ):
            response = main.app.test_client().post(
                "/api/v1/parse",
                data={
                    "file": [
                        (BytesIO(b"alpha"), "alpha.txt"),
                        (BytesIO(b"beta"), "beta.txt"),
                    ],
                    "parser_engine": "plain",
                },
                content_type="multipart/form-data",
            )

        self.assertEqual(200, response.status_code)
        payload = response.get_json()
        self.assertEqual(["alpha.txt", "beta.txt"], [item["filename"] for item in payload["results"]])
        self.assertEqual(["plain", "plain"], [item["parser_engine"] for item in payload["results"]])

    def test_parse_stream_endpoint_emits_file_progress_and_final_results(self):
        import main

        def fake_parse_single_file(file, parse_options):
            return {
                "filename": file.filename,
                "markdown": f"# {file.filename}",
                "parser_engine": parse_options.get("parser_engine"),
                "parse_id": f"parse-{file.filename}",
            }

        with patch.object(main, "_parse_single_file", side_effect=fake_parse_single_file), patch.object(
            main, "_append_ops_audit_event", return_value=None
        ):
            response = main.app.test_client().post(
                "/api/v1/parse/stream",
                data={
                    "file": [
                        (BytesIO(b"alpha"), "alpha.txt"),
                        (BytesIO(b"beta"), "beta.txt"),
                    ],
                    "parser_engine": "plain",
                },
                content_type="multipart/form-data",
                buffered=True,
            )

        self.assertEqual(200, response.status_code, response.get_data(as_text=True))
        self.assertIn("text/event-stream", response.content_type)
        events = self._parse_sse_events(response.get_data(as_text=True))
        self.assertEqual(
            ["start", "file_started", "file_completed", "file_started", "file_completed", "done"],
            [event["event"] for event in events],
        )
        self.assertEqual({"file_count": 2, "parser_engine": "plain"}, events[0]["data"])
        self.assertEqual({"index": 1, "total": 2, "filename": "alpha.txt"}, events[1]["data"])
        first_completed = events[2]["data"]
        self.assertEqual(1, first_completed["index"])
        self.assertEqual("alpha.txt", first_completed["filename"])
        self.assertFalse(first_completed["has_error"])
        self.assertEqual("parse-alpha.txt", first_completed["parse_id"])
        self.assertEqual("alpha.txt", first_completed["result"]["filename"])
        done = events[-1]["data"]
        self.assertEqual("ok", done["status"])
        self.assertEqual(2, done["file_count"])
        self.assertEqual(2, done["result_count"])
        self.assertEqual(0, done["error_count"])
        self.assertEqual(["alpha.txt", "beta.txt"], [item["filename"] for item in done["results"]])

    def test_parse_stream_endpoint_is_documented_as_sync_sse_parse_surface(self):
        repo_root = Path(__file__).resolve().parents[1]
        api_doc = (repo_root / "docs/API.md").read_text(encoding="utf-8")
        readme = (repo_root / "README.md").read_text(encoding="utf-8")
        openapi = json.loads((repo_root / "openapi.json").read_text(encoding="utf-8"))

        parse_stream = openapi["paths"].get("/api/v1/parse/stream")
        self.assertIsInstance(parse_stream, dict)
        stream_response = parse_stream["post"]["responses"]["200"]["content"]
        self.assertIn("text/event-stream", stream_response)
        self.assertIn('curl -N -X POST "http://localhost:8000/api/v1/parse/stream"', api_doc)
        self.assertIn("`start`、`file_started`、`file_completed`、`done`", api_doc)
        self.assertIn('curl -N -X POST "http://localhost:8000/api/v1/parse/stream"', readme)

    def test_async_task_progress_events_and_sse_stream_contract_are_present(self):
        repo_root = Path(__file__).resolve().parents[1]
        main_source = (repo_root / "main.py").read_text(encoding="utf-8")
        api_doc = (repo_root / "docs/API.md").read_text(encoding="utf-8")
        openapi_text = (repo_root / "openapi.json").read_text(encoding="utf-8")

        self.assertIn("progress_total = max(1, len(running.input_files))", main_source)
        self.assertIn('"progress": {"current": index, "total": progress_total}', main_source)
        self.assertIn('"file_completed"', main_source)
        self.assertIn('@app.route("/api/v1/tasks/<task_id>/stream", methods=["GET"])', main_source)
        self.assertIn('mimetype="text/event-stream"', main_source)
        self.assertIn("/api/v1/tasks/<task_id>/stream", api_doc)
        self.assertIn("/api/v1/parse/stream", api_doc)
        self.assertIn('"text/event-stream"', openapi_text)

    def test_gradio_console_has_no_rag_chat_or_embedding_controls(self):
        gradio_path = Path(__file__).resolve().parents[1] / "gradio_app.py"
        source = gradio_path.read_text(encoding="utf-8")
        forbidden_terms = [
            "/api/v1/rag",
            "/api/v1/chat-provider",
            "embeddings/backfill",
            "Chat Provider",
            "chat_provider_",
            "_run_rag_query",
            "_clear_rag_console",
            "_refresh_rag_query_center",
            "_prefill_rag_target",
            "ingest_record_mode",
            "_run_ingest_backfill",
            "rag_",
        ]
        leaked_terms = [term for term in forbidden_terms if term in source]
        self.assertEqual([], leaked_terms)

    def test_user_facing_docs_do_not_advertise_rag_chat_or_vector_retrieval(self):
        repo_root = Path(__file__).resolve().parents[1]
        docs = {
            "README.md": (repo_root / "README.md").read_text(encoding="utf-8"),
            "docs/API.md": (repo_root / "docs/API.md").read_text(encoding="utf-8"),
        }
        forbidden_terms = [
            "/api/v1/rag",
            "/api/v1/chat-provider",
            "embeddings/backfill",
            "DEEPDOC_CHAT",
            "DEEPDOC_EMBEDDING",
            "DEEPDOC_INGEST_PG_VECTOR",
            "pgvector",
            "fastembed",
            "Chat Provider",
            "chat provider",
            "RAG 查询",
            "vector retrieval",
            "mode=vector",
            "mode=hybrid",
        ]
        leaks = {
            filename: [term for term in forbidden_terms if term in content]
            for filename, content in docs.items()
        }
        leaks = {filename: terms for filename, terms in leaks.items() if terms}
        self.assertEqual({}, leaks)

    def test_docker_observability_stack_configs_are_not_packaged(self):
        repo_root = Path(__file__).resolve().parents[1]
        removed_paths = [
            repo_root / "docker" / "grafana",
            repo_root / "docker" / "prometheus.yml",
            repo_root / "docker" / "prometheus-alerts.yml",
            repo_root / "docker" / "otel-collector-config.yaml",
            repo_root / "docker" / "tempo.yaml",
        ]

        for path in removed_paths:
            with self.subTest(path=path):
                self.assertFalse(path.exists(), str(path))

    def test_core_parser_and_ingest_code_uses_neutral_chunk_export_names(self):
        repo_root = Path(__file__).resolve().parents[1]
        source_paths = [
            "main.py",
            "common/parse_artifacts.py",
            "common/ingest_publisher.py",
            "common/ingest_postgres.py",
            "common/file_utils.py",
            "common/constants.py",
            "common/nlp/tokenizer.py",
            "deepdoc/parser/pdf_parser.py",
            "deepdoc/parser/html_parser.py",
            "deepdoc/parser/json_parser.py",
            "deepdoc/parser/excel_parser.py",
            "deepdoc/parser/docx_parser.py",
            "deepdoc/parser/utils.py",
            "deepdoc/parser/resume/step_two.py",
            "deepdoc/parser/resume/entities/corporations.py",
            "deepdoc/vision/table_structure_recognizer.py",
        ]
        forbidden_terms = [
            "RagChunkRecord",
            "RagIngestRecord",
            "build_rag_chunk_records",
            "build_rag_ingest_records",
            "from rag.nlp",
            "import rag.nlp",
            "rag_tokenizer",
            "RAG_FLOW_SERVICE_NAME",
            "rag_flow",
            "RAG_PROJECT_BASE",
            "RAG_DEPLOY_BASE",
        ]
        leaks = {}
        for relative_path in source_paths:
            content = (repo_root / relative_path).read_text(encoding="utf-8")
            terms = [term for term in forbidden_terms if term in content]
            if terms:
                leaks[relative_path] = terms
        self.assertEqual({}, leaks)

    def test_document_tokenizer_is_parser_local_not_rag_adapter(self):
        from common.nlp import tokenizer

        self.assertIsInstance(tokenizer.tokenize("中国 ABC 123"), str)
        self.assertIsInstance(tokenizer.fine_grained_tokenize("中国 ABC 123"), str)
        self.assertIsInstance(tokenizer.tag("合同"), str)
        self.assertIsInstance(tokenizer.freq("合同"), dict)
        self.assertTrue(tokenizer.is_chinese("中"))
        self.assertTrue(tokenizer.is_number("3"))
        self.assertTrue(tokenizer.is_alphabet("A"))
        self.assertEqual("ＡB １２", tokenizer.naive_qie("  ＡB   １２  "))

    def test_source_tree_has_no_rag_package(self):
        repo_root = Path(__file__).resolve().parents[1]
        self.assertFalse((repo_root / "rag").exists())

    def test_generated_metadata_and_model_cache_have_no_rag_or_embedding_surfaces(self):
        repo_root = Path(__file__).resolve().parents[1]
        forbidden_paths = [
            repo_root / "common" / "__pycache__" / "rag_query.cpython-310.pyc",
            repo_root / "common" / "__pycache__" / "embedding_provider.cpython-310.pyc",
            repo_root / "common" / "__pycache__" / "chat_provider.cpython-310.pyc",
            repo_root / "common" / "__pycache__" / "self_check_http_chat.cpython-310.pyc",
            repo_root / "docker" / "__pycache__" / "run_chat_provider_probe_worker.cpython-310.pyc",
            repo_root / "docker" / "__pycache__" / "run_mock_openai_chat.cpython-310.pyc",
            repo_root / "resources" / "models" / "embeddings",
        ]
        for path in forbidden_paths:
            self.assertFalse(path.exists(), str(path))

        egg_info = repo_root / "deepdoc.egg-info"
        if egg_info.exists():
            metadata_files = [
                egg_info / "top_level.txt",
                egg_info / "SOURCES.txt",
                egg_info / "PKG-INFO",
            ]
            forbidden_terms = ["\nrag\n", "rag/", "rag.", "RAGFlow", "embedding"]
            leaks = {}
            for path in metadata_files:
                if not path.exists():
                    continue
                content = f"\n{path.read_text(encoding='utf-8', errors='ignore')}\n"
                terms = [term for term in forbidden_terms if term in content]
                if terms:
                    leaks[path.name] = terms
            self.assertEqual({}, leaks)

    def test_runtime_parse_caches_have_no_rag_vector_or_embedding_artifacts(self):
        repo_root = Path(__file__).resolve().parents[1]
        cache_roots = [
            repo_root / "resources" / "artifacts",
            repo_root / "resources" / "tasks",
            repo_root / "resources" / "self_checks",
            repo_root / "resources" / "temp",
        ]
        forbidden_path_terms = ("rag", "vector", "embedding", "embeddings")
        forbidden_content_terms = (
            "/api/v1/rag",
            "RAG retrieval",
            "document RAG",
            "vector retrieval",
            "vector_enabled",
            "embedding_model",
            "LLM answers",
            "chat provider",
            "Chat Provider",
            "pgvector",
            "fastembed",
        )
        text_suffixes = {".json", ".jsonl", ".md", ".txt", ".yaml", ".yml", ".html", ".csv"}
        leaks: dict[str, list[str]] = {}

        for cache_root in cache_roots:
            if not cache_root.exists():
                continue
            for path in cache_root.rglob("*"):
                relative_path = str(path.relative_to(repo_root))
                path_terms = [term for term in forbidden_path_terms if term in relative_path.lower()]
                if path_terms:
                    leaks.setdefault(relative_path, []).extend(path_terms)
                if not path.is_file() or path.suffix.lower() not in text_suffixes:
                    continue
                content = path.read_text(encoding="utf-8", errors="ignore")
                content_terms = [term for term in forbidden_content_terms if term in content]
                if content_terms:
                    leaks.setdefault(relative_path, []).extend(content_terms)

        self.assertEqual({}, leaks)

    def test_container_runtime_has_no_rag_chat_or_embedding_entrypoints(self):
        repo_root = Path(__file__).resolve().parents[1]
        files = {
            "Dockerfile.cpu": repo_root / "Dockerfile.cpu",
            "Dockerfile.gpu": repo_root / "Dockerfile.gpu",
            "docker-compose.yml": repo_root / "docker-compose.yml",
            "tools/ci/strict_config_guard_smoke.py": repo_root / "tools/ci/strict_config_guard_smoke.py",
        }
        forbidden_terms = [
            "/api/v1/rag",
            "chat-mock",
            "mock-chat",
            "chat_provider",
            "chat-provider",
            "DEEPDOC_CHAT",
            "DEEPDOC_EMBEDDING",
            "generate_answer",
            "mode=vector",
            "mode=hybrid",
            "pgvector",
            "fastembed",
        ]
        leaks = {}
        for label, path in files.items():
            if not path.exists():
                continue
            content = path.read_text(encoding="utf-8")
            terms = [term for term in forbidden_terms if term in content]
            if terms:
                leaks[label] = terms
        self.assertEqual({}, leaks)

    def test_env_files_do_not_document_rag_chat_or_embedding_runtime(self):
        repo_root = Path(__file__).resolve().parents[1]
        files = {
            ".env": repo_root / ".env",
        }
        forbidden_terms = [
            "/api/v1/rag",
            "DEEPDOC_CHAT",
            "DEEPDOC_EMBEDDING",
            "DEEPDOC_INGEST_PG_VECTOR",
            "DEEPDOC_RATE_LIMIT_RAG",
            "DEEPDOC_MAX_INFLIGHT_RAG",
            "RAG",
            "rag",
            "vector retrieval",
            "pgvector",
            "fastembed",
            "embeddings",
        ]
        leaks = {}
        for label, path in files.items():
            if not path.exists():
                continue
            content = path.read_text(encoding="utf-8")
            terms = [term for term in forbidden_terms if term in content]
            if terms:
                leaks[label] = terms
        self.assertEqual({}, leaks)

    def test_project_guidance_and_plans_do_not_steer_to_rag(self):
        repo_root = Path(__file__).resolve().parents[1]
        files = {
            "AGENTS.md": repo_root / "AGENTS.md",
            "SECURITY_AUDIT.md": repo_root / "SECURITY_AUDIT.md",
            "plans/optimization-roadmap.md": repo_root / "plans/optimization-roadmap.md",
            "plans/2026-06-06-unified-model-hub-and-parser-cleanup.md": (
                repo_root / "plans/2026-06-06-unified-model-hub-and-parser-cleanup.md"
            ),
            "docs/superpowers/specs/2026-06-08-deepdoc-table-rapidtable-design.md": (
                repo_root / "docs/superpowers/specs/2026-06-08-deepdoc-table-rapidtable-design.md"
            ),
            "docs/superpowers/plans/2026-06-08-deepdoc-table-rapidtable.md": (
                repo_root / "docs/superpowers/plans/2026-06-08-deepdoc-table-rapidtable.md"
            ),
        }
        forbidden_terms = [
            "rag/",
            "rag.",
            "RAGFlow",
            "ragflow",
            "RAG 增强",
            "RAG 闭环",
            "接入 RAG",
            "对接 RAG",
            "RAG 查询",
        ]
        leaks = {}
        for label, path in files.items():
            content = path.read_text(encoding="utf-8")
            terms = [term for term in forbidden_terms if term in content]
            if terms:
                leaks[label] = terms
        self.assertEqual({}, leaks)


if __name__ == "__main__":
    unittest.main()
