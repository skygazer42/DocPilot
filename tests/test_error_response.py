import json
import unittest
from pathlib import Path


class ErrorResponseTest(unittest.TestCase):
    def test_error_payload_keeps_legacy_error_and_adds_code_locale_and_details(self):
        from common.errors import ErrorCode, build_error_payload

        payload = build_error_payload(
            ErrorCode.NO_FILE_PART,
            locale="zh-CN",
            details={"field": "file"},
        )

        self.assertEqual("NO_FILE_PART", payload["error_code"])
        self.assertEqual("No file part", payload["error"])
        self.assertEqual("No file part", payload["message"])
        self.assertEqual("缺少文件字段", payload["message_zh"])
        self.assertEqual("zh-CN", payload["locale"])
        self.assertEqual({"field": "file"}, payload["details"])

    def test_parse_endpoint_returns_structured_error_for_missing_file(self):
        import main

        client = main.app.test_client()
        response = client.post("/api/v1/parse", data={}, headers={"Accept-Language": "zh-CN"})

        self.assertEqual(400, response.status_code)
        payload = response.get_json()
        self.assertEqual("NO_FILE_PART", payload["error_code"])
        self.assertEqual("No file part", payload["error"])
        self.assertEqual("No file part", payload["message"])
        self.assertEqual("缺少文件字段", payload["message_zh"])
        self.assertEqual("zh-CN", payload["locale"])
        self.assertEqual({"field": "file"}, payload["details"])

    def test_parse_result_item_errors_carry_error_code_for_unsupported_extension(self):
        import main

        client = main.app.test_client()
        response = client.post(
            "/api/v1/parse",
            data={"file": (Path(__file__).open("rb"), "sample.unsupported")},
            content_type="multipart/form-data",
        )

        self.assertEqual(400, response.status_code)
        result = response.get_json()["results"][0]
        self.assertEqual("UNSUPPORTED_FILE_EXTENSION", result["error_code"])
        self.assertIn("Unsupported file extension", result["error"])
        self.assertEqual("不支持的文件扩展名", result["message_zh"])
        self.assertEqual({"filename": "sample.unsupported"}, result["details"])

    def test_openapi_documents_unified_error_response_schema(self):
        repo_root = Path(__file__).resolve().parents[1]
        openapi = json.loads((repo_root / "openapi.json").read_text(encoding="utf-8"))
        schema = openapi["components"]["schemas"]["ErrorResponse"]

        for field in ("error", "error_code", "message", "message_zh", "locale", "details"):
            self.assertIn(field, schema["properties"])
        self.assertIn("error_code", schema["required"])

        roadmap = (repo_root / "plans/optimization-roadmap.md").read_text(encoding="utf-8")
        api_doc = (repo_root / "docs/API.md").read_text(encoding="utf-8")
        self.assertIn("| H5 | **统一错误码 + i18n** | 已落地", roadmap)
        self.assertIn("error_code", api_doc)


if __name__ == "__main__":
    unittest.main()
