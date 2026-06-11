import re

from bs4 import BeautifulSoup


def clean_text(text) -> str:
    # 清理 DeepDoc 位置标签，避免污染最终 Markdown 文本
    if not isinstance(text, str):
        return str(text)
    return re.sub(r"@@[0-9-]+\t[0-9.\t]+##", "", text)


def strip_markdown_images(markdown_text: str) -> str:
    # 同时移除 Markdown 图片语法和 HTML img 标签
    if not isinstance(markdown_text, str) or not markdown_text:
        return markdown_text
    cleaned = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", markdown_text)
    cleaned = re.sub(r"<img\b[^>]*>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def strip_html_tags(markdown_text: str) -> str:
    if not isinstance(markdown_text, str) or not markdown_text:
        return markdown_text
    text = BeautifulSoup(markdown_text, "html.parser").get_text("\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def html_table_to_rows(html_str: str) -> list:
    soup = BeautifulSoup(html_str, "html.parser")
    table = soup.find("table")
    if not table:
        return []
    rows = []
    for tr in table.find_all("tr"):
        cells = [
            cell.get_text(separator=" ", strip=True)
            for cell in tr.find_all(["th", "td"])
        ]
        if cells:
            rows.append(cells)
    return rows


def table_to_md(table_data) -> str:
    rows = table_data
    if isinstance(table_data, tuple):
        rows = table_data[1]
    if not rows:
        return ""
    if isinstance(rows, str) and "<table" in rows.lower():
        rows = html_table_to_rows(rows)
        if not rows:
            return ""
    if isinstance(rows, str):
        return clean_text(rows)
    if not isinstance(rows, list):
        return str(rows)
    if all(isinstance(r, str) for r in rows):
        return "\n".join(rows)
    try:
        max_cols = max(len(r) for r in rows) if rows else 0
        if max_cols == 0:
            return ""
        md_lines = []
        header = rows[0] + [""] * (max_cols - len(rows[0]))
        md_lines.append(
            "| "
            + " | ".join(clean_text(str(c)).replace("\n", "<br>") for c in header)
            + " |"
        )
        md_lines.append("| " + " | ".join(["---"] * max_cols) + " |")
        for row in rows[1:]:
            row = row + [""] * (max_cols - len(row))
            md_lines.append(
                "| "
                + " | ".join(clean_text(str(c)).replace("\n", "<br>") for c in row)
                + " |"
            )
        return "\n" + "\n".join(md_lines) + "\n"
    except Exception:
        return str(rows)


def results_to_markdown(results) -> str:
    # 统一把 parser 多种返回结构收敛成单一 Markdown 字符串
    if isinstance(results, str):
        return clean_text(results)
    if isinstance(results, tuple):
        text_content = results[0]
        tables = results[1] if len(results) > 1 else []
        parts = []
        if isinstance(text_content, str):
            parts.append(clean_text(text_content))
        elif isinstance(text_content, list):
            for box in text_content:
                if isinstance(box, dict) and box.get("text", "").strip():
                    parts.append(clean_text(box["text"]))
                elif isinstance(box, dict) and box.get("block_content", "").strip():
                    parts.append(clean_text(box["block_content"]))
                elif isinstance(box, str) and box.strip():
                    parts.append(clean_text(box))
                elif (
                    isinstance(box, (list, tuple))
                    and len(box) > 0
                    and isinstance(box[0], str)
                    and box[0].strip()
                ):
                    parts.append(clean_text(box[0]))
        if isinstance(tables, list) and tables:
            parts.append("\n\n### Tables\n")
            for tbl in tables:
                if (
                    isinstance(tbl, tuple)
                    and len(tbl) == 2
                    and isinstance(tbl[0], tuple)
                ):
                    tbl = tbl[0]
                parts.append(table_to_md(tbl))
        return "\n\n".join(parts)
    if isinstance(results, list):
        processed = []
        for line in results:
            if isinstance(line, str):
                processed.append(clean_text(line))
            elif (
                isinstance(line, (list, tuple))
                and len(line) > 0
                and isinstance(line[0], str)
            ):
                processed.append(clean_text(line[0]))
        return "\n\n".join(processed)
    return clean_text(str(results))


def post_process_markdown(
    markdown_content: str, return_images: bool, strict_text: bool
) -> str:
    if not return_images:
        markdown_content = strip_markdown_images(markdown_content)
    if strict_text:
        markdown_content = strip_html_tags(markdown_content)
    return markdown_content
