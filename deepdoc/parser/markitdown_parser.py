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

from pathlib import Path


class MarkItDownParser:
    def __init__(self, enable_plugins: bool = False):
        try:
            from markitdown import MarkItDown
        except ImportError as exc:
            raise ImportError(
                "markitdown is not installed. Reinstall the project dependencies to "
                "enable parser_engine=markitdown."
            ) from exc

        # Official docs recommend convert_local() for server-side local file parsing.
        self._converter = MarkItDown(enable_plugins=enable_plugins)

    def __call__(self, fnm, binary=None):
        if binary is not None:
            raise ValueError(
                "MarkItDownParser only supports local file paths in this service path."
            )

        result = self._converter.convert_local(Path(fnm))
        markdown = getattr(result, "markdown", None) or getattr(
            result, "text_content", ""
        )
        return markdown or ""
