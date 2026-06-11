from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CajConversionResult:
    pdf_path: str
    output_dir: str
    converter: str
    command: list[str]
    output_size_bytes: int

    def model_dump(self) -> dict[str, object]:
        return {
            "pdf_path": self.pdf_path,
            "output_dir": self.output_dir,
            "converter": self.converter,
            "command": self.command,
            "output_size_bytes": self.output_size_bytes,
        }


class DeepDocCajParser:
    """Convert CAJ to PDF so the existing PDF parser pipeline can handle it."""

    def __init__(self, command_template: str | None = None):
        self.command_template = (
            command_template
            or os.environ.get("DEEPDOC_CAJ2PDF_COMMAND_TEMPLATE")
            or "caj2pdf {input} {output}"
        )

    def convert_to_pdf(
        self,
        input_path,
        *,
        output_dir=None,
        request_timeout: int | None = None,
    ) -> CajConversionResult:
        source = Path(input_path)
        if not source.exists():
            raise FileNotFoundError(f"CAJ source file not found: {source}")
        target_dir = Path(output_dir or source.parent)
        target_dir.mkdir(parents=True, exist_ok=True)
        output_path = target_dir / f"{source.stem}.pdf"
        command = self._build_command(source, output_path)
        timeout = int(request_timeout or os.environ.get("DEEPDOC_CAJ2PDF_TIMEOUT", "600"))
        completed = subprocess.run(
            command,
            cwd=str(target_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.decode("utf-8", errors="replace").strip()
            stdout = completed.stdout.decode("utf-8", errors="replace").strip()
            detail = stderr or stdout or f"exit code {completed.returncode}"
            raise RuntimeError(f"CAJ to PDF conversion failed: {detail}")
        if not output_path.exists():
            raise RuntimeError(f"CAJ to PDF conversion did not create output PDF: {output_path}")
        head = output_path.read_bytes()[:4]
        if head != b"%PDF":
            raise RuntimeError(f"CAJ converter output is not a PDF: {output_path}")
        return CajConversionResult(
            pdf_path=str(output_path),
            output_dir=str(target_dir),
            converter="caj2pdf",
            command=command,
            output_size_bytes=output_path.stat().st_size,
        )

    def _build_command(self, input_path: Path, output_path: Path) -> list[str]:
        template = (self.command_template or "").strip()
        if not template:
            raise ValueError("DEEPDOC_CAJ2PDF_COMMAND_TEMPLATE cannot be empty")
        rendered = template.format(
            input=shlex.quote(str(input_path)),
            output=shlex.quote(str(output_path)),
        )
        return shlex.split(rendered)
