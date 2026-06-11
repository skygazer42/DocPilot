import json
import os

from common.model_store import (
    MODEL_GROUP_FILES,
    PUBLISHED_MODEL_GROUPS,
    download_groups,
    get_model_group_provenance,
    get_model_repo,
    get_model_root,
    list_missing_files,
    validate_ocr_dictionaries,
    validate_ocr_dictionary,
    validate_ocr_recognition_model_alignments,
)

SUPPORTED_DOWNLOAD_COMMANDS = frozenset(MODEL_GROUP_FILES).union({"all", "published"})


def download(groups: str = "published"):
    model_dir = get_model_root()
    os.makedirs(model_dir, exist_ok=True)

    print(f"Downloading DocPilot model group(s) {groups} to {model_dir}...")
    print(f"Repository: {get_model_repo()}")
    download_groups(groups, model_root=model_dir)
    print("Download complete.")


def print_manifest():
    payload = {
        "model_root": get_model_root(),
        "repo_id": get_model_repo(),
        "published_model_groups": list(PUBLISHED_MODEL_GROUPS),
        "model_group_provenance": get_model_group_provenance("all"),
        "ocr_dictionary": validate_ocr_dictionary(),
        "ocr_dictionaries": validate_ocr_dictionaries(),
        "ocr_recognition_alignments": validate_ocr_recognition_model_alignments(),
        "groups": {
            group: {
                "required_files": list(files),
                "missing_files": list_missing_files(group),
            }
            for group, files in MODEL_GROUP_FILES.items()
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import sys

    command = sys.argv[1].strip().lower() if len(sys.argv) > 1 else "published"
    if command == "manifest":
        print_manifest()
    elif command in SUPPORTED_DOWNLOAD_COMMANDS:
        download(command)
    else:
        raise SystemExit(
            "Usage: python download_models.py [{}|manifest]".format(
                "|".join(sorted(SUPPORTED_DOWNLOAD_COMMANDS))
            )
        )
