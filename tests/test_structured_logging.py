import io
import json
import logging
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch


class StructuredLoggingTest(unittest.TestCase):
    def test_json_formatter_emits_request_and_parse_context_fields(self):
        import main
        from common.log import JsonLogFormatter, Log

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JsonLogFormatter())
        logger = Log()
        logger.logger.addHandler(handler)
        try:
            with main.app.test_request_context("/api/v1/parse", headers={"X-Request-ID": "req-123"}):
                with logger.context(file_sha="abc123", engine="deepdoc"):
                    logger.info("parse finished")
        finally:
            logger.logger.removeHandler(handler)

        payload = json.loads(stream.getvalue())
        self.assertEqual("INFO", payload["level"])
        self.assertEqual("parse finished", payload["message"])
        self.assertEqual("req-123", payload["request_id"])
        self.assertEqual("abc123", payload["file_sha"])
        self.assertEqual("deepdoc", payload["engine"])
        self.assertIn("trace_id", payload)
        self.assertIn("span_id", payload)

    def test_parse_single_file_binds_file_sha_and_engine_to_log_context(self):
        import main

        seen_contexts: list[dict[str, object]] = []

        class CaptureContext:
            def __init__(self, fields):
                self.fields = fields

            def __enter__(self):
                seen_contexts.append(self.fields)

            def __exit__(self, exc_type, exc, tb):
                return False

        class Upload:
            filename = "sample.txt"

            def save(self, path):
                Path(path).write_bytes(b"hello")

        with patch.object(main.logger, "context", side_effect=lambda **fields: CaptureContext(fields)):
            with redirect_stderr(io.StringIO()):
                result = main._parse_single_file(Upload(), {"parser_engine": "plain"})

        self.assertEqual("sample.txt", result["filename"])
        self.assertTrue(any(item.get("engine") == "plain" for item in seen_contexts))
        self.assertTrue(
            any(
                item.get("file_sha")
                == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
                for item in seen_contexts
            )
        )

    def test_json_logging_is_documented_and_roadmap_marks_f3_done(self):
        repo_root = Path(__file__).resolve().parents[1]
        roadmap = (repo_root / "plans/optimization-roadmap.md").read_text(encoding="utf-8")
        api_doc = (repo_root / "docs/API.md").read_text(encoding="utf-8")
        readme = (repo_root / "README.md").read_text(encoding="utf-8")

        self.assertIn("| F3 | **结构化 JSON 日志** | 已落地", roadmap)
        self.assertIn("DEEPDOC_LOG_FORMAT=json", api_doc)
        self.assertIn("DEEPDOC_LOG_FORMAT=json", readme)


if __name__ == "__main__":
    unittest.main()
