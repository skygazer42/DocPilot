import unittest

from common.ratelimit import (
    FixedWindowRule,
    InMemoryRateLimitStore,
    InflightAdmissionController,
    RequestRateLimiter,
    parse_bytes_rule,
    parse_rate_rule,
)


class RequestProtectionTest(unittest.TestCase):
    def test_parse_request_rate_limit_rejects_second_request_in_window(self):
        limiter = RequestRateLimiter(
            enabled=True,
            store=InMemoryRateLimitStore(),
            rules_by_scope={
                "parse": [
                    FixedWindowRule(name="parse", scope="parse", limit=1, window_seconds=60),
                ],
            },
        )

        first = limiter.evaluate(scope="parse", identity="subject:demo", request_bytes=100)
        second = limiter.evaluate(scope="parse", identity="subject:demo", request_bytes=100)

        self.assertTrue(first.allowed)
        self.assertFalse(second.allowed)
        self.assertEqual("rate limit exceeded", second.error_payload()["error"])
        self.assertEqual("parse", second.error_payload()["scope"])
        self.assertIn("X-RateLimit-Parse-Limit", second.headers)
        self.assertIn("Retry-After", second.headers)

    def test_parse_byte_quota_rejects_oversized_window_consumption(self):
        quota_rule = parse_bytes_rule("parse-bytes", "parse", "10b/min")
        self.assertIsNotNone(quota_rule)
        limiter = RequestRateLimiter(
            enabled=True,
            store=InMemoryRateLimitStore(),
            rules_by_scope={"parse": [quota_rule]},
        )

        first = limiter.evaluate(scope="parse", identity="ip:127.0.0.1", request_bytes=6)
        second = limiter.evaluate(scope="parse", identity="ip:127.0.0.1", request_bytes=6)

        self.assertTrue(first.allowed)
        self.assertFalse(second.allowed)
        self.assertEqual("usage quota exceeded", second.error_payload()["error"])
        self.assertEqual("bytes", second.error_payload()["cost_name"])
        self.assertIn("X-Quota-ParseBytes-Limit", second.headers)

    def test_rate_and_byte_rule_parsers_support_runtime_env_specs(self):
        rate_rule = parse_rate_rule("parse", "parse", "3/min")
        byte_rule = parse_bytes_rule("parse-bytes", "parse", "1.5mb/day")

        self.assertEqual(3, rate_rule.limit)
        self.assertEqual(60, rate_rule.window_seconds)
        self.assertEqual("requests", rate_rule.cost_name)
        self.assertEqual(1_500_000, byte_rule.limit)
        self.assertEqual(86400, byte_rule.window_seconds)
        self.assertEqual("bytes", byte_rule.cost_name)

    def test_inflight_admission_controller_releases_parse_pool_capacity(self):
        controller = InflightAdmissionController({"parse": 1})

        first = controller.acquire("parse")
        second = controller.acquire("parse")
        controller.release(first.lease)
        third = controller.acquire("parse")

        self.assertTrue(first.allowed)
        self.assertFalse(second.allowed)
        self.assertEqual("server busy", second.error_payload()["error"])
        self.assertEqual("parse", second.error_payload()["pool"])
        self.assertTrue(third.allowed)
        controller.release(third.lease)


if __name__ == "__main__":
    unittest.main()
