import tempfile
import unittest
from pathlib import Path

from common.gradio_temp import cleanup_temp_on_startup, prepare_gradio_temp_dir


class GradioTempDirTest(unittest.TestCase):
    def test_prepare_gradio_temp_dir_uses_namespaced_subdirectory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gradio_tmp = prepare_gradio_temp_dir(root)

            self.assertEqual(root / "gradio", gradio_tmp)
            self.assertTrue(gradio_tmp.exists())
            self.assertTrue(gradio_tmp.is_dir())

    def test_cleanup_only_removes_items_inside_gradio_temp_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sibling_stage_dir = root / "deepdoc-stream-abc123"
            sibling_stage_dir.mkdir(parents=True)
            sibling_file = sibling_stage_dir / "upload.txt"
            sibling_file.write_text("keep me", encoding="utf-8")

            gradio_tmp = prepare_gradio_temp_dir(root)
            (gradio_tmp / "old-output.md").write_text("remove me", encoding="utf-8")
            (gradio_tmp / "old-dir").mkdir()

            removed_items = cleanup_temp_on_startup(gradio_tmp)

            self.assertEqual(2, removed_items)
            self.assertEqual([], list(gradio_tmp.iterdir()))
            self.assertTrue(sibling_stage_dir.exists())
            self.assertTrue(sibling_file.exists())


if __name__ == "__main__":
    unittest.main()
