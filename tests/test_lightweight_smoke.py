import unittest

from tools.ci.lightweight_smoke import _validate_build_info


class LightweightSmokeTest(unittest.TestCase):
    def test_build_info_accepts_ok_status(self):
        _validate_build_info({"status": "ok", "build_source": "embedded"}, allow_runtime_build_info=False)

    def test_build_info_rejects_runtime_fallback_by_default(self):
        with self.assertRaisesRegex(RuntimeError, "build-info endpoint did not return status=ok"):
            _validate_build_info(
                {"status": "degraded", "build_source": "runtime-fallback"},
                allow_runtime_build_info=False,
            )

    def test_build_info_can_allow_source_runtime_fallback_for_local_smoke(self):
        _validate_build_info(
            {"status": "degraded", "build_source": "runtime-fallback"},
            allow_runtime_build_info=True,
        )

    def test_build_info_does_not_allow_other_degraded_sources(self):
        with self.assertRaisesRegex(RuntimeError, "build-info endpoint did not return status=ok"):
            _validate_build_info(
                {"status": "degraded", "build_source": "embedded"},
                allow_runtime_build_info=True,
            )


if __name__ == "__main__":
    unittest.main()
