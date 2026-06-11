#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
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

import re
from collections import Counter


_TOKEN_RE = re.compile(
    r"[A-Za-z]+(?:[-_][A-Za-z0-9]+)*"
    r"|\d+(?:[.,]\d+)*"
    r"|[\u4e00-\u9fff]+"
    r"|[^\s]",
    flags=re.UNICODE,
)


def _as_text(value) -> str:
    return "" if value is None else str(value)


def is_chinese(s):
    text = _as_text(s)
    return bool(text) and "\u4e00" <= text[0] <= "\u9fff"


def is_number(s):
    text = _as_text(s)
    return bool(text) and text[0].isdigit()


def is_alphabet(s):
    text = _as_text(s)
    return bool(text) and text[0].isalpha()


def naive_qie(txt):
    return re.sub(r"\s+", " ", _as_text(txt)).strip()


def _split_tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(naive_qie(text))


def _token_tag(token: str) -> str:
    if not token:
        return ""
    if all(char.isdigit() or char in ".,%" for char in token):
        return "m"
    if all("\u4e00" <= char <= "\u9fff" for char in token):
        return "n"
    if all(char.isalpha() or char in "-_" for char in token):
        return "eng"
    return "x"


class TextTokenizer:
    """Small local tokenizer used by document parsers."""

    def tokenize(self, line: str) -> str:
        return " ".join(_split_tokens(self._tradi2simp(self._strQ2B(line))))

    def fine_grained_tokenize(self, tks: str) -> str:
        parts: list[str] = []
        for token in _split_tokens(self._tradi2simp(self._strQ2B(tks))):
            if all(is_chinese(char) for char in token) and len(token) > 1:
                parts.extend(token)
            else:
                parts.append(token)
        return " ".join(parts)

    def tag(self, token: str = "", *_args, **_kwargs) -> str:
        return _token_tag(_as_text(token).strip())

    def freq(self, text: str = "", *_args, **_kwargs) -> dict[str, int]:
        return dict(Counter(self.tokenize(text).split()))

    @staticmethod
    def _tradi2simp(s):
        return _as_text(s)

    @staticmethod
    def _strQ2B(s):
        chars = []
        for char in _as_text(s):
            code = ord(char)
            if code == 0x3000:
                chars.append(" ")
            elif 0xFF01 <= code <= 0xFF5E:
                chars.append(chr(code - 0xFEE0))
            else:
                chars.append(char)
        return "".join(chars)


# ---- 模块级便捷引用 ----

tokenizer = TextTokenizer()
tokenize = tokenizer.tokenize
fine_grained_tokenize = tokenizer.fine_grained_tokenize
tag = tokenizer.tag
freq = tokenizer.freq
tradi2simp = tokenizer._tradi2simp
strQ2B = tokenizer._strQ2B
