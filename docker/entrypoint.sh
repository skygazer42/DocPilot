#!/bin/sh
set -eu

MODEL_ROOT="${DEEPDOC_MODEL_PATH:-/app/resources/models}"
export DEEPDOC_MODEL_PATH="$MODEL_ROOT"

mkdir -p "$MODEL_ROOT"

if [ "${DEEPDOC_AUTO_DOWNLOAD:-1}" = "1" ]; then
  if python - <<'PY'
from common.model_store import get_download_groups_from_env, list_missing_files

groups = get_download_groups_from_env()
missing = list_missing_files(groups)
if missing:
    print("[entrypoint] missing model files:", ", ".join(missing))
    raise SystemExit(0)
raise SystemExit(1)
PY
  then
    python download_models.py "${DEEPDOC_DOWNLOAD_GROUPS:-published}"
  fi
fi

exec "$@"
