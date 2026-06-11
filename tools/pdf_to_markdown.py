import argparse
from pathlib import Path

from pypdf import PdfReader


def extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    lines = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text:
            lines.extend(text.splitlines())
    return "\n".join(lines).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert PDF text layer to Markdown.")
    parser.add_argument("pdf", type=Path, help="Path to the PDF file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output Markdown path (default: <input>.md)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pdf_path = args.pdf
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    output_path = args.output or pdf_path.with_suffix(".md")
    markdown_text = extract_pdf_text(pdf_path)
    output_path.write_text(markdown_text, encoding="utf-8")
    print(f"Wrote Markdown to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
