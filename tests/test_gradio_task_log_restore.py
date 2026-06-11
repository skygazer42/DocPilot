import importlib
import sys
import types
import unittest
from unittest.mock import patch


def _import_gradio_app():
    sys.modules.pop("gradio_app", None)

    fake_gradio = types.ModuleType("gradio")
    fake_gradio.__version__ = "5.0.0"
    sys.modules["gradio"] = fake_gradio

    fake_gradio_pdf = types.ModuleType("gradio_pdf")
    fake_gradio_pdf.PDF = object
    sys.modules["gradio_pdf"] = fake_gradio_pdf

    return importlib.import_module("gradio_app")


class GradioTaskLogRestoreTest(unittest.TestCase):
    def test_restore_latest_task_events_uses_newest_task_log(self):
        gradio_app = _import_gradio_app()
        latest_events = [
            {
                "created_at": "2026-06-11T09:05:59Z",
                "event_type": "running",
                "payload": {"filename": "latest.pdf", "file_count": 1},
            }
        ]

        with (
            patch.object(gradio_app, "_fetch_async_tasks", return_value=[{"task_id": "task-new"}, {"task_id": "task-old"}]),
            patch.object(gradio_app, "_safe_fetch_async_task_events", return_value=latest_events) as fetch_events,
        ):
            restored = gradio_app._restore_latest_task_events()

        fetch_events.assert_called_once_with("task-new")
        self.assertEqual(gradio_app._render_task_events(latest_events), restored)


if __name__ == "__main__":
    unittest.main()
