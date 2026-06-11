import unittest
from pathlib import Path


class DockerImageVariantTest(unittest.TestCase):
    def test_only_cpu_and_gpu_dockerfile_entrypoints_are_supported(self):
        repo_root = Path(__file__).resolve().parents[1]
        variants = {
            "Dockerfile.cpu": (
                "onnxruntime==1.23.2",
                'DEEPDOC_IMAGE_VARIANT="cpu"',
                'DEEPDOC_DEFAULT_ONNX_PROVIDER="cpu"',
                "-e .",
            ),
            "Dockerfile.gpu": (
                "onnxruntime-gpu==1.23.2",
                'DEEPDOC_IMAGE_VARIANT="gpu"',
                'DEEPDOC_DEFAULT_ONNX_PROVIDER="auto"',
                '-e ".[gpu]"',
            ),
        }

        for filename, expected_terms in variants.items():
            with self.subTest(filename=filename):
                content = (repo_root / filename).read_text(encoding="utf-8")
                self.assertIn("ARG DEEPDOC_ONNXRUNTIME_PACKAGE", content)
                self.assertNotIn("DEEPDOC_PIP_EXTRAS", content)
                self.assertNotIn("gradio_app.py", content)
                self.assertNotIn("run_gradio.py", content)
                self.assertNotIn("ingest-postgres", content)
                self.assertNotIn("wait_for_postgres.py", content)
                self.assertNotIn("run_parse_worker.py", content)
                self.assertIn("DEEPDOC_DOWNLOAD_GROUPS=published", content)
                for term in expected_terms:
                    self.assertIn(term, content)

        self.assertFalse((repo_root / "Dockerfile.trt").exists())

    def test_default_dockerfile_is_removed_to_keep_only_cpu_gpu_entrypoints(self):
        repo_root = Path(__file__).resolve().parents[1]
        self.assertFalse((repo_root / "Dockerfile").exists())

    def test_docker_bake_is_removed_to_avoid_variant_sprawl(self):
        repo_root = Path(__file__).resolve().parents[1]
        self.assertFalse((repo_root / "docker-bake.hcl").exists())

    def test_prod_compose_and_prod_env_are_removed_to_avoid_duplicate_deployment_paths(self):
        repo_root = Path(__file__).resolve().parents[1]

        self.assertFalse((repo_root / "docker-compose.prod.yml").exists())
        self.assertFalse((repo_root / ".env.prod.example").exists())

    def test_github_workflows_and_ci_runtime_directory_are_not_packaged(self):
        repo_root = Path(__file__).resolve().parents[1]

        self.assertFalse((repo_root / ".github").exists())
        self.assertFalse((repo_root / ".ci").exists())

    def test_docker_directory_only_contains_api_entrypoint(self):
        repo_root = Path(__file__).resolve().parents[1]
        docker_files = sorted(
            path.relative_to(repo_root).as_posix()
            for path in (repo_root / "docker").rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        )

        self.assertEqual(["docker/entrypoint.sh"], docker_files)

    def test_compose_is_parser_service_stack_without_platform_dependencies(self):
        repo_root = Path(__file__).resolve().parents[1]
        content = (repo_root / "docker-compose.yml").read_text(encoding="utf-8")

        self.assertIn("dockerfile: ${DEEPDOC_DOCKERFILE:-Dockerfile.cpu}", content)
        self.assertIn("deepdoc:", content)
        self.assertNotIn("deepdoc-gradio:", content)
        self.assertNotIn("profiles:", content)
        self.assertIn("DEEPDOC_INGEST_PUBLISHER: none", content)
        self.assertIn("DEEPDOC_ASYNC_ENABLED: \"0\"", content)
        self.assertIn("DEEPDOC_TRACING_ENABLED: \"0\"", content)
        self.assertIn("DEEPDOC_RATE_LIMIT_ENABLED: \"0\"", content)
        self.assertIn("DEEPDOC_DOWNLOAD_GROUPS: ${DEEPDOC_DOWNLOAD_GROUPS:-published}", content)

        forbidden_terms = [
            "deepdoc-redis",
            "deepdoc-pg",
            "postgres:",
            "redis:",
            "prometheus",
            "grafana",
            "tempo",
            "otel-collector",
            "run_parse_worker.py",
            "run_callback_redrive_worker.py",
            "run_self_check_worker.py",
            "run_retention_janitor.py",
            "Dockerfile.trt",
            "gradio",
            "GRADIO_PORT",
            "run_gradio.py",
        ]
        leaks = [term for term in forbidden_terms if term in content]
        self.assertEqual([], leaks)

    def test_docs_describe_only_cpu_gpu_docker_deployment(self):
        repo_root = Path(__file__).resolve().parents[1]
        api_doc = (repo_root / "docs/API.md").read_text(encoding="utf-8")
        readme = (repo_root / "README.md").read_text(encoding="utf-8")
        roadmap = (repo_root / "plans/optimization-roadmap.md").read_text(encoding="utf-8")

        for content in (api_doc, readme, roadmap):
            with self.subTest(content=content[:20]):
                self.assertIn("Dockerfile.cpu", content)
                self.assertIn("Dockerfile.gpu", content)
                self.assertNotIn("Dockerfile.trt", content)
                self.assertNotIn("docker-bake.hcl", content)
                self.assertNotIn("docker-compose.prod.yml", content)
                self.assertNotIn(".env.prod.example", content)
                self.assertNotIn("deepdoc-standalone-trt", content)
                self.assertNotIn("deepdoc-pg", content)
                self.assertNotIn("deepdoc-redis", content)
                self.assertNotIn("--profile gradio", content)
                self.assertNotIn("deepdoc-gradio", content)

        self.assertIn("TensorRT EP 仍是代码层显式运行时选项", roadmap)

    def test_project_default_dependency_does_not_force_gpu_runtime(self):
        repo_root = Path(__file__).resolve().parents[1]
        content = (repo_root / "pyproject.toml").read_text(encoding="utf-8")

        self.assertIn('"onnxruntime==1.23.2"', content)
        self.assertIn("gpu = [", content)
        self.assertIn('"onnxruntime-gpu==1.23.2"', content)
        self.assertNotIn('platform_machine == "x86_64"', content)


if __name__ == "__main__":
    unittest.main()
