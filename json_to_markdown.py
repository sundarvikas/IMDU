# json_to_markdown.py
from pathlib import Path
from typing import Dict, Any, Optional, List


def json_to_markdown(data: Dict[Any, Any], md_path: Optional[str] = None) -> str:
    """
    Converts structured document JSON into well-organized Markdown with:
      - # Main Title
      - ## Section Headings
      - ### Subheadings / Subsections
      - Clean tables
      - Proper spacing and hierarchy

    Preserves reading order and logical structure.
    """
    # -----------------------------
    # Table Formatter (Clean & Aligned)
    # -----------------------------
    def format_table(headers: List[str], rows: List[List[str]]) -> str:
        if not headers and not rows:
            return ""

        all_rows = ([headers] if headers else []) + rows
        col_widths = [
            max(len(str(cell)) for cell in col)
            for col in zip(*all_rows)
        ]
        col_widths = [max(w, 3) for w in col_widths]  # min width

        def pad_row(row):
            padded = " | ".join(
                (str(cell) if cell is not None else "").ljust(width)
                for cell, width in zip(row, col_widths)
            )
            return f"| {padded} |"

        lines = []
        if headers:
            lines.append(pad_row(headers))
            lines.append(pad_row(["-" * w for w in col_widths]))
        for row in rows:
            padded_row = pad_row((row + [""] * len(col_widths))[:len(col_widths)])
            lines.append(padded_row)

        return "\n".join(lines) + "\n\n"

    # -----------------------------
    # Block to Markdown
    # -----------------------------
    def block_to_md(block: Dict[str, Any]) -> str:
        btype = block.get("type", "").lower()
        content = block.get("content", "")
        if not content:
            return ""

        # -----------------------------
        # HIERARCHY: # → ## → ### → text
        # -----------------------------

        if btype == "title":
            return f"# {content.strip()}\n\n"
        
        elif btype == "section_header":
            return f"## {content.strip()}\n\n"

        elif btype in ["subsection", "sub_header", "page_header"]:
            return f"### {content.strip()}\n\n"

        elif btype == "text":
            return f"{content.strip()}\n\n"

        elif btype == "list_item":
            return f"- {content.strip()}\n"

        elif btype == "table" and isinstance(content, dict):
            headers = [h or "" for h in content.get("headers", [])]
            rows = [[cell or "" for cell in r] for r in content.get("rows", [])]
            return format_table(headers, rows)

        elif btype == "caption":
            return f"*{content.strip()}*\n\n"
        elif btype == "footnote":
            return f"\n[^{content.strip()}]\n"
        elif btype == "formula":
            return f"\n```\n{content.strip()}\n```\n\n"
        elif btype == "footer":
            return f"\n\n---\n\n*{content.strip()}*\n"
        else:
            return f"{content.strip()}\n\n"

    # -----------------------------
    # MAIN: Build Structured Markdown
    # -----------------------------
    md_lines = []

    if "raw_text" in data:
        md_lines.append("# Parsed Document\n\n")
        md_lines.append("```\n")
        md_lines.append(data["raw_text"])
        md_lines.append("\n```\n")
    else:
        document = data.get("document", {})
        doc_name = document.get("doc-name", "Document")

        # Always start with a main title
        # md_lines.append(f"# {doc_name}\n\n")

        pages = document.get("pages", [])
        first_section_added = False

        for page in pages:
            blocks = page.get("blocks", [])

            for block in blocks:
                btype = block.get("type", "").lower()
                content = block.get("content", "")

                if not content:
                    continue

                # Promote first section_header to main title if no title exists yet
                if btype == "doc_title":
                    # Skip doc_title completely as per requirement
                    continue
                elif btype == "section_header" and not first_section_added:
                    # Promote first section_header to # Main Title
                    # Instead of unsafe md_lines[-2], just insert at top
                    md_lines.insert(0, f"# {content.strip()}\n\n")
                    first_section_added = True
                else:
                    md_lines.append(block_to_md(block))

    markdown_text = "".join(md_lines).strip() + "\n"

    # Save to file
    if md_path:
        Path(md_path).write_text(markdown_text, encoding="utf-8")

    return markdown_text