#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys


IMPORT_SNIPPET = "import main\n"
POSITIVE_SNIPPET = """
import json
import main

print(json.dumps({
    "auth_mode": main._auth_mode(),
    "cors_allow_all": main.CORS_STATE["allow_all"],
    "strict": main.RUNTIME_CONFIG_STATE["strict"],
    "warnings": main.RUNTIME_CONFIG_STATE["warnings"],
}, sort_keys=True))
"""


COMMON_ENV = {
    "DEEPDOC_AUTO_DOWNLOAD": "0",
    "DEEPDOC_WAIT_FOR_INGEST_DB": "0",
    "DEEPDOC_INGEST_PUBLISHER": "none",
    "DEEPDOC_AUDIT_BACKEND": "file",
    "DEEPDOC_ASYNC_ENABLED": "0",
    "DEEPDOC_RATE_LIMIT_ENABLED": "0",
    "DEEPDOC_SELF_CHECK_AUTO_ENABLED": "0",
    "DEEPDOC_RETENTION_JANITOR_ENABLED": "0",
    "DEEPDOC_TRACING_ENABLED": "0",
}


def _docker_python(image: str, env: dict[str, str], snippet: str) -> subprocess.CompletedProcess[str]:
    args = ["docker", "run", "-i", "--rm", "--entrypoint", "python"]
    for key, value in env.items():
        args.extend(["-e", f"{key}={value}"])
    args.extend([image, "-"])
    return subprocess.run(
        args,
        input=snippet,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def _run_negative(image: str) -> None:
    env = {
        **COMMON_ENV,
        "DEEPDOC_CONFIG_STRICT": "1",
        "DEEPDOC_ENVIRONMENT": "production",
        "SECRET_ACCESS_KEY": "change-me",
        "DEEPDOC_CORS_ALLOW_ALL": "1",
    }
    completed = _docker_python(image, env, IMPORT_SNIPPET)
    print("--- strict negative output ---")
    print((completed.stdout or "").rstrip())
    print(f"exit={completed.returncode}")
    if completed.returncode == 0:
        raise RuntimeError("strict placeholder configuration unexpectedly imported successfully")
    output = completed.stdout or ""
    required_fragments = (
        "Invalid strict DeepDoc runtime configuration",
        "SECRET_ACCESS_KEY",
        "DEEPDOC_CORS_ALLOW_ALL",
    )
    missing = [fragment for fragment in required_fragments if fragment not in output]
    if missing:
        raise RuntimeError(f"strict negative output missing expected fragments: {', '.join(missing)}")


def _run_positive(image: str) -> None:
    env = {
        **COMMON_ENV,
        "DEEPDOC_CONFIG_STRICT": "1",
        "DEEPDOC_ENVIRONMENT": "production",
        "SECRET_ACCESS_KEY": "prod-secret-0123456789abcdef",
        "DEEPDOC_CORS_ALLOW_ALL": "0",
        "DEEPDOC_CORS_ALLOWED_ORIGINS": "https://console.example.com",
    }
    completed = _docker_python(image, env, POSITIVE_SNIPPET)
    print("--- strict positive output ---")
    print((completed.stdout or "").rstrip())
    print(f"exit={completed.returncode}")
    if completed.returncode != 0:
        raise RuntimeError("strict valid configuration failed to import")
    lines = [line for line in (completed.stdout or "").splitlines() if line.strip()]
    payload = json.loads(lines[-1])
    expected = {
        "auth_mode": "api_key",
        "cors_allow_all": False,
        "strict": True,
        "warnings": [],
    }
    if payload != expected:
        raise RuntimeError(f"strict positive payload mismatch: expected={expected!r} actual={payload!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify strict production config guard inside the Docker image.")
    parser.add_argument("--image", default="deepdoc-standalone:0.1")
    args = parser.parse_args()

    try:
        _run_negative(args.image)
        _run_positive(args.image)
    except Exception as exc:
        print(f"strict config guard smoke failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
