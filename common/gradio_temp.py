from pathlib import Path


def prepare_gradio_temp_dir(root_dir: str | Path) -> Path:
    tmp_dir = Path(root_dir) / "gradio"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir


def cleanup_temp_on_startup(tmp_dir: str | Path) -> int:
    directory = Path(tmp_dir)
    if not directory.exists():
        return 0
    removed_items = 0
    for item in directory.iterdir():
        try:
            if item.is_dir():
                for child in item.rglob("*"):
                    if child.is_file():
                        child.unlink()
                for child_dir in sorted((path for path in item.rglob("*") if path.is_dir()), reverse=True):
                    child_dir.rmdir()
                item.rmdir()
            else:
                item.unlink()
            removed_items += 1
        except Exception:
            continue
    return removed_items
