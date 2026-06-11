import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from common.audit_log import AuditEvent, LocalAuditLogStore
from common.parse_artifacts import S3ArtifactStore


class FakeS3Body:
    def __init__(self, payload: bytes):
        self.payload = payload

    def read(self) -> bytes:
        return self.payload


class FakeS3Client:
    def __init__(self):
        self.objects: dict[tuple[str, str], dict[str, object]] = {}

    def put_object(self, *, Bucket, Key, Body, ContentType):
        self.objects[(Bucket, Key)] = {"Body": bytes(Body), "ContentType": ContentType}

    def get_object(self, *, Bucket, Key):
        item = self.objects[(Bucket, Key)]
        return {"Body": FakeS3Body(item["Body"]), "ContentType": item["ContentType"]}

    def list_objects_v2(self, **kwargs):
        bucket = kwargs["Bucket"]
        prefix = kwargs.get("Prefix") or ""
        contents = [
            {"Key": key, "LastModified": index}
            for index, ((item_bucket, key), _value) in enumerate(self.objects.items())
            if item_bucket == bucket and key.startswith(prefix)
        ]
        return {"Contents": contents, "IsTruncated": False}

    def generate_presigned_url(self, operation, *, Params, ExpiresIn):
        return f"signed://{operation}/{Params['Bucket']}/{Params['Key']}?expires={ExpiresIn}"

    def delete_object(self, *, Bucket, Key):
        self.objects.pop((Bucket, Key), None)


class OperationalSurfacesTest(unittest.TestCase):
    def test_openapi_and_docs_expose_async_task_audit_artifact_and_ops_surfaces(self):
        repo_root = Path(__file__).resolve().parents[1]
        openapi = json.loads((repo_root / "openapi.json").read_text(encoding="utf-8"))
        paths = set(openapi.get("paths", {}))
        docs = (repo_root / "docs/API.md").read_text(encoding="utf-8")
        readme = (repo_root / "README.md").read_text(encoding="utf-8")

        for path in (
            "/api/v1/parse/async",
            "/api/v1/tasks",
            "/api/v1/tasks/{task_id}",
            "/api/v1/tasks/{task_id}/events",
            "/api/v1/tasks/{task_id}/stream",
            "/api/v1/tasks/{task_id}/retry",
            "/api/v1/tasks/retry",
            "/api/v1/tasks/cleanup",
            "/api/v1/audit/events",
            "/api/v1/audit/events/{event_id}",
            "/api/v1/audit/events/cleanup",
        ):
            self.assertIn(path, paths)

        self.assertIn("DEEPDOC_ASYNC_ENABLED: \"0\"", (repo_root / "docker-compose.yml").read_text(encoding="utf-8"))
        self.assertIn("DEEPDOC_ARTIFACT_BACKEND=s3", docs)
        self.assertIn("DEEPDOC_ARTIFACT_BACKEND=s3", readme)
        self.assertIn("DEEPDOC_TRACING_ENABLED=1", docs)
        self.assertIn("DEEPDOC_TRACING_ENABLED=1", readme)
        self.assertIn("/api/v1/audit/events", docs)

    def test_metrics_source_mentions_gpu_page_pool_execution(self):
        repo_root = Path(__file__).resolve().parents[1]
        metrics_source = (repo_root / "common/metrics.py").read_text(encoding="utf-8")

        self.assertIn("deepdoc_gpu_page_pool_jobs_total", metrics_source)
        self.assertIn("deepdoc_gpu_page_pool_device_jobs", metrics_source)

    def test_s3_artifact_store_writes_reads_and_resolves_urls(self):
        client = FakeS3Client()
        with patch.dict(os.environ, {"DEEPDOC_ARTIFACT_PUBLIC_BASE_URL": "https://cdn.example.com/root"}):
            store = S3ArtifactStore(bucket="deepdoc-artifacts", prefix="prod", s3_client=client)

        paths = store.get_paths("parse-1", "sample.pdf")
        store.write_markdown(paths, "# Parsed")

        payload, media_type = store.read_file("parse-1", "markdown.md")
        self.assertEqual(b"# Parsed", payload)
        self.assertEqual("text/markdown; charset=utf-8", media_type)
        self.assertEqual(
            "https://cdn.example.com/root/prod/parse-1/assets/figure-1.png",
            store.resolve_asset_url(
                parse_id="parse-1",
                relative_path="assets/figure-1.png",
                download_path="/api/v1/artifacts/parse-1/assets/figure-1.png",
                mode="direct",
            ),
        )
        self.assertIn(
            "signed://get_object/deepdoc-artifacts/prod/parse-1/assets/figure-1.png",
            store.resolve_asset_url(
                parse_id="parse-1",
                relative_path="assets/figure-1.png",
                download_path="/api/v1/artifacts/parse-1/assets/figure-1.png",
                mode="signed",
                expires_in=3600,
            ),
        )

    def test_cors_and_tracing_runtime_contracts_are_configured(self):
        import main

        with patch.dict(
            os.environ,
            {
                "DEEPDOC_CORS_ALLOW_ALL": "0",
                "DEEPDOC_CORS_ALLOWED_ORIGINS": "https://console.example.com",
                "DEEPDOC_CORS_ALLOWED_HEADERS": "Authorization,X-Request-ID",
            },
        ):
            cors_state = main._cors_health_state()

        self.assertFalse(cors_state["allow_all"])
        self.assertEqual(["https://console.example.com"], cors_state["allowed_origins"])
        self.assertEqual(["Authorization", "X-Request-ID"], cors_state["allowed_headers"])

        tracing_source = (Path(__file__).resolve().parents[1] / "common/tracing.py").read_text(encoding="utf-8")
        main_source = (Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")
        self.assertIn("FlaskInstrumentor().instrument_app", tracing_source)
        self.assertIn("RequestsInstrumentor().instrument()", tracing_source)
        self.assertIn("BotocoreInstrumentor().instrument()", tracing_source)
        self.assertIn("PsycopgInstrumentor().instrument()", tracing_source)
        self.assertIn("trace_operation(", main_source)
        self.assertIn("deepdoc.parse_single_file", main_source)
        self.assertIn("deepdoc.build_structured_artifact", main_source)
        self.assertIn("deepdoc.ingest.publish", main_source)

    def test_audit_events_api_lists_gets_and_cleans_events(self):
        import main

        with tempfile.TemporaryDirectory(prefix="deepdoc-audit-api-") as temp_dir:
            original_store = main.AUDIT_LOG_STORE
            store = LocalAuditLogStore(root_dir=temp_dir)
            event = store.append_event(
                AuditEvent(
                    tenant_id="tenant-a",
                    actor_subject="tester",
                    request_id="req-1",
                    action="parse.sync",
                    resource_type="parse_request",
                    resource_id="parse-1",
                    status="ok",
                    payload={"file_count": 1, "parser_engine": "deepdoc"},
                    metadata={"filename": "sample.pdf"},
                )
            )
            main.AUDIT_LOG_STORE = store
            try:
                client = main.app.test_client()
                list_response = client.get("/api/v1/audit/events?limit=10&action=parse.sync")
                get_response = client.get(f"/api/v1/audit/events/{event.event_id}")
                cleanup_response = client.post(
                    "/api/v1/audit/events/cleanup",
                    json={"dry_run": True, "keep_latest": 0, "action": "parse.sync"},
                )
            finally:
                main.AUDIT_LOG_STORE = original_store

        self.assertEqual(200, list_response.status_code, list_response.get_data(as_text=True))
        listed = list_response.get_json()["results"]
        self.assertEqual(1, len(listed))
        self.assertEqual(event.event_id, listed[0]["event_id"])
        self.assertEqual("req-1", listed[0]["request_id"])

        self.assertEqual(200, get_response.status_code, get_response.get_data(as_text=True))
        self.assertEqual(event.event_id, get_response.get_json()["event"]["event_id"])

        self.assertEqual(200, cleanup_response.status_code, cleanup_response.get_data(as_text=True))
        cleanup = cleanup_response.get_json()
        self.assertTrue(cleanup["dry_run"])
        self.assertEqual(1, cleanup["candidate_count"])

    def test_roadmap_marks_verified_production_surfaces_as_done(self):
        roadmap = (Path(__file__).resolve().parents[1] / "plans/optimization-roadmap.md").read_text(encoding="utf-8")

        self.assertIn("| F2 | **OpenTelemetry tracing** | 已落地", roadmap)
        self.assertIn("| F4 | **审计日志** | 已落地", roadmap)
        self.assertIn("| G2 | **异步任务队列** | 代码层能力保留", roadmap)
        self.assertIn("默认 Docker 部署不再附带 Redis/worker 容器", roadmap)
        self.assertIn("| G3 | **S3/MinIO 对象存储抽象** | 已落地", roadmap)
        self.assertIn("| H3 | **异步任务接口** | 已落地", roadmap)
        self.assertIn("| H6 | **CORS 收紧** | 已落地", roadmap)


if __name__ == "__main__":
    unittest.main()
