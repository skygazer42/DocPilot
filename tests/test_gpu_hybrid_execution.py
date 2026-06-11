import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from common.async_tasks import AsyncTaskInput, AsyncTaskStore, build_async_task


class GpuPagePoolDispatchTest(unittest.TestCase):
    def test_dispatch_gpu_page_jobs_assigns_pages_round_robin(self):
        from common.gpu_page_pool import dispatch_gpu_page_jobs

        page_jobs = [
            {"job_id": f"page-{index}", "page_number": index, "route": "digital_clean"}
            for index in range(1, 9)
        ]

        result = dispatch_gpu_page_jobs(
            task_id="task-1",
            page_jobs=page_jobs,
            devices=[0, 1],
        )

        self.assertEqual(8, result["submitted_job_count"])
        self.assertEqual([0, 1], result["worker_device_ids"])
        self.assertEqual({0: 4, 1: 4}, result["device_job_counts"])
        self.assertEqual(
            [1, 2, 3, 4, 5, 6, 7, 8],
            [job["page_number"] for job in result["page_jobs"]],
        )


class AsyncGpuHybridExecutionTest(unittest.TestCase):
    def test_run_async_parse_task_records_gpu_page_pool_plan_without_changing_terminal_status(self):
        import main

        with tempfile.TemporaryDirectory(prefix="deepdoc-gpu-page-pool-task-") as temp_dir:
            store = AsyncTaskStore(root_dir=temp_dir)
            source_path = Path(temp_dir) / "sample.pdf"
            source_path.write_bytes(b"%PDF-1.4\n% async gpu hybrid test\n")
            input_file = AsyncTaskInput(
                filename="sample.pdf",
                file_type="pdf",
                size_bytes=source_path.stat().st_size,
                sha256="abc123",
                source_path=str(source_path),
            )
            task = build_async_task(
                queue_name="deepdoc:async:test",
                parser_engine="deepdoc",
                parse_options={
                    "parser_engine": "deepdoc",
                    "deepdoc_pdf_mode": "hybrid",
                    "execution_profile": "gpu",
                },
                input_files=[input_file],
                tenant_id=None,
                requested_by="test",
                auth_subject=None,
            )
            store.create_task(task)

            dispatch_summary = {
                "submitted_job_count": 8,
                "worker_device_ids": [0, 1],
                "device_job_counts": {0: 4, 1: 4},
                "page_jobs": [
                    {"job_id": f"page-{index}", "page_number": index, "route": "digital_clean"}
                    for index in range(1, 9)
                ],
                "page_count": 8,
                "ocr_page_numbers": [7, 8],
                "complex_block_page_numbers": [7],
            }

            with (
                patch.object(main, "ASYNC_TASK_STORE", store),
                patch.object(
                    main,
                    "_plan_gpu_page_pool_dispatch",
                    return_value=dispatch_summary,
                    create=True,
                ),
                patch.object(
                    main,
                    "_parse_single_file",
                    return_value={
                        "filename": "sample.pdf",
                        "type": "pdf",
                        "markdown": "# ok",
                        "parser_engine": "deepdoc",
                    },
                ),
            ):
                result = main.run_async_parse_task(task.task_id)

            self.assertEqual("succeeded", result["status"])

            persisted = store.load_task(task.task_id)
            self.assertEqual(
                dispatch_summary["submitted_job_count"],
                ((persisted.metadata or {}).get("gpu_page_pool") or {}).get("submitted_job_count"),
            )
            self.assertEqual(
                dispatch_summary["worker_device_ids"],
                ((persisted.result_summary or {}).get("gpu_page_pool") or {}).get("worker_device_ids"),
            )

            events = store.read_events(task.task_id)
            planned_events = [event for event in events if event.event_type == "gpu_page_jobs_planned"]
            self.assertEqual(1, len(planned_events))
            self.assertEqual(8, planned_events[0].payload["submitted_job_count"])
            self.assertEqual([0, 1], planned_events[0].payload["worker_device_ids"])


if __name__ == "__main__":
    unittest.main()
