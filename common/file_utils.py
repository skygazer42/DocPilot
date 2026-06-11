#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import os


def validate_file(file, allowed_extensions=None, check_image=False):
    if not file:
        return "No file provided"

    try:
        max_mb = int(os.environ.get("DEEPDOC_MAX_FILE_SIZE_MB", 100))
        file.seek(0, os.SEEK_END)
        size = file.tell()
        file.seek(0)
        if size > max_mb * 1024 * 1024:
            return f"File size exceeds {max_mb}MB limit"
    except Exception:
        pass

    filename = file.filename or ""
    if allowed_extensions and filename:
        ext = filename.split(".")[-1].lower() if "." in filename else ""
        if ext not in allowed_extensions:
            return f"Unsupported file extension: .{ext}. Allowed: {', '.join(allowed_extensions)}"

    if check_image:
        try:
            from PIL import Image

            img = Image.open(file)
            width, height = img.size
            file.seek(0)
            if width > 10000 or height > 10000:
                return f"Image dimensions too large: {width}x{height}. Max 10000x10000"
        except Exception as e:
            return f"Invalid image file: {str(e)}"

    return None


PROJECT_BASE = os.getenv("DEEPDOC_PROJECT_BASE") or os.getenv("DEEPDOC_DEPLOY_BASE")


def get_project_base_directory(*args):
    global PROJECT_BASE
    if PROJECT_BASE is None:
        PROJECT_BASE = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                os.pardir,
            )
        )

    if args:
        return os.path.join(PROJECT_BASE, *args)
    return PROJECT_BASE


def traversal_files(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            fullname = os.path.join(root, f)
            yield fullname
