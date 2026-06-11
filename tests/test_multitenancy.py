import io
import json
import logging
import tempfile
import unittest
from pathlib import Path

from common.audit_log import LocalAuditLogStore


class MultiTenancyTest(unittest.TestCase):
    def test_request_guard_identity_prefers_tenant_bucket_over_subject(self):
        import main

        with main.app.test_request_context("/api/v1/parse", headers={"X-Tenant-ID": "tenant-a"}):
            main.g.auth_context = main._build_auth_context(
                mode="jwt_hs256",
                subject="user-1",
                tenant_id="tenant-a",
                scopes=set(),
            )
            identity = main._request_guard_identity()

        self.assertEqual("tenant:tenant-a", identity)

    def test_request_guard_identity_falls_back_to_subject_without_tenant(self):
        import main

        with main.app.test_request_context("/api/v1/parse"):
            main.g.auth_context = main._build_auth_context(
                mode="api_key",
                subject="api_key",
                tenant_id=None,
                scopes=set(),
            )
            identity = main._request_guard_identity()

        self.assertEqual("subject:api_key", identity)

    def test_json_logs_include_request_tenant_and_auth_context(self):
        import main
        from common.log import JsonLogFormatter, Log

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JsonLogFormatter())
        logger = Log()
        logger.logger.addHandler(handler)
        try:
            with main.app.test_request_context("/api/v1/parse", headers={"X-Request-ID": "req-tenant"}):
                main.g.auth_context = main._build_auth_context(
                    mode="jwt_hs256",
                    subject="user-1",
                    tenant_id="tenant-a",
                    scopes={"parse"},
                )
                logger.info("tenant scoped parse")
        finally:
            logger.logger.removeHandler(handler)

        payload = json.loads(stream.getvalue())
        self.assertEqual("req-tenant", payload["request_id"])
        self.assertEqual("tenant-a", payload["tenant_id"])
        self.assertEqual("user-1", payload["auth_subject"])
        self.assertEqual("jwt_hs256", payload["auth_mode"])

    def test_ops_audit_event_defaults_to_current_tenant_context(self):
        import main

        with tempfile.TemporaryDirectory(prefix="deepdoc-tenant-audit-") as temp_dir:
            original_store = main.AUDIT_LOG_STORE
            store = LocalAuditLogStore(root_dir=temp_dir)
            main.AUDIT_LOG_STORE = store
            try:
                with main.app.test_request_context("/api/v1/parse", headers={"X-Request-ID": "req-audit"}):
                    main.g.auth_context = main._build_auth_context(
                        mode="jwt_hs256",
                        subject="user-1",
                        tenant_id="tenant-a",
                        scopes={"parse"},
                    )
                    event_id = main._append_ops_audit_event(
                        "parse.sync",
                        resource_type="parse_request",
                        payload={"file_count": 1},
                    )
            finally:
                main.AUDIT_LOG_STORE = original_store

            payload = json.loads((Path(temp_dir) / "events.jsonl").read_text(encoding="utf-8").splitlines()[0])

        self.assertIsNotNone(event_id)
        self.assertEqual("tenant-a", payload["tenant_id"])
        self.assertEqual("user-1", payload["actor_subject"])
        self.assertEqual("req-audit", payload["request_id"])

    def test_multitenancy_docs_and_roadmap_mark_g5_done(self):
        repo_root = Path(__file__).resolve().parents[1]
        api_doc = (repo_root / "docs/API.md").read_text(encoding="utf-8")
        roadmap = (repo_root / "plans/optimization-roadmap.md").read_text(encoding="utf-8")
        env_file = (repo_root / ".env").read_text(encoding="utf-8")

        self.assertIn("DEEPDOC_AUTH_MODE=jwt_hs256", api_doc)
        self.assertIn("X-Tenant-ID", api_doc)
        self.assertIn("tenant_id", api_doc)
        self.assertIn("DEEPDOC_DEFAULT_TENANT_ID", env_file)
        self.assertIn("| G5 | **多租户** | 已落地", roadmap)


if __name__ == "__main__":
    unittest.main()
