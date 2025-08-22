from pathlib import Path
from typing import Dict, Any, Optional, List

# -----------------------------
# HELPER FORMATTING FUNCTIONS
# -----------------------------

def format_table(headers: List[str], rows: List[List[str]]) -> str:
    """Formats a list of headers and rows into a clean Markdown table."""
    if not headers and not rows:
        return ""
    all_rows = ([headers] if headers else []) + rows
    if not all_rows: return ""

    num_cols = len(headers) if headers else (len(rows[0]) if rows else 0)
    if not num_cols: return ""
    
    col_widths = [0] * num_cols
    for row in all_rows:
        for i, cell in enumerate(row):
            if i < num_cols:
                col_widths[i] = max(col_widths[i], len(str(cell)))

    col_widths = [max(w, 3) for w in col_widths]

    def pad_row(row, widths):
        cells = []
        for i, cell in enumerate(row):
            if i < len(widths):
                cells.append((str(cell) if cell is not None else "").ljust(widths[i]))
        while len(cells) < len(widths):
            cells.append("".ljust(widths[len(cells)]))
        return f"| {' | '.join(cells)} |"

    lines = []
    if headers:
        lines.append(pad_row(headers, col_widths))
        lines.append(f"|{'|'.join(['-' * (w + 2) for w in col_widths])}|")
    for row in rows:
        lines.append(pad_row(row, col_widths))

    return "\n".join(lines) + "\n\n"

def format_prescription(content: dict) -> str:
    """Formats a prescription JSON object into readable Markdown."""
    lines = ["### Medical Prescription\n", "---"]
    lines.append(f"**Patient:** {content.get('patient_name', 'N/A')}")
    lines.append(f"**Doctor:** {content.get('doctor_name', 'N/A')}")
    lines.append(f"**Date:** {content.get('date', 'N/A')}\n")
    lines.append("**Medications:**")
    for med in content.get('medications', []):
        lines.append(f"- **{med.get('name', 'N/A')}**: {med.get('dosage', '')} - *{med.get('frequency', '')}*")
    return "\n".join(lines) + "\n\n---\n\n"

def format_flowchart(content: str) -> str:
    """Formats flowchart content into a Mermaid diagram block."""
    if not content: return ""
    return f"```mermaid\ngraph TD\n{content.strip()}\n```\n\n"

def format_handwritten(content: str) -> str:
    """Formats transcribed handwritten text into a blockquote."""
    if not content: return ""
    return f"> **Handwritten:** *{content.strip()}*\n\n"

def format_image_container(content: dict, indent_level: int) -> str:
    """Formats an image container and recursively renders its nested blocks."""
    description = content.get("description", "Embedded Image")
    nested_blocks = content.get("blocks", [])
    
    container_md = [f"\n---\n> **Embedded Image:** *{description}*"]
    # Recursively call the main renderer for the nested blocks
    nested_md = render_blocks(nested_blocks, indent_level + 1)
    container_md.append(nested_md)
    container_md.append("---\n")
    return "\n".join(container_md)

def render_blocks(blocks: List[Dict[str, Any]], indent_level=0) -> str:
    """Main recursive rendering function to convert a list of blocks to Markdown."""
    md_parts = []
    indent = "  " * indent_level
    
    for block in blocks:
        btype = block.get("type", "").lower()
        content = block.get("content")

        if not content:
            continue

        if btype in ['title', 'header', 'doc_title']:
            md_parts.append(f"{indent}# {content.strip()}\n\n")
        elif btype in ["pie_chart", "bar_chart", "line_graph"]:
            title = content.get("title", "Untitled Chart")
            md_parts.append(f"> **Chart Data Extracted:** *'{title}'* \n> (View the interactive version in the 'Charts & Graphs' tab.)\n\n")
        elif btype == "section_header":
            num = block.get("section_number")
            md_parts.append(f"{indent}## {num}. {content.strip()}\n\n" if num else f"{indent}## {content.strip()}\n\n")
        elif btype in ["subsection", "sub_header"]:
            md_parts.append(f"{indent}### {content.strip()}\n\n")
        elif btype == "paragraph" or btype == "text":
            md_parts.append(f"{indent}{content.strip()}\n\n")
        elif btype == "list_item":
            md_parts.append(f"{indent}- {content.strip()}\n")
        elif btype == "table" and isinstance(content, dict):
            md_parts.append(format_table(content.get("headers", []), content.get("rows", [])))
        elif btype == "prescription" and isinstance(content, dict):
            md_parts.append(format_prescription(content))
        elif btype == "flowchart":
            md_parts.append(format_flowchart(content))
        elif btype == "handwritten_text":
            md_parts.append(format_handwritten(content))
        elif btype == "figure_description":
            md_parts.append(f"![{content.strip()}](placeholder.png)\n\n")
        elif btype == "image_container" and isinstance(content, dict):
            md_parts.append(format_image_container(content, indent_level))
        else:
            md_parts.append(f"{indent}{str(content).strip()}\n\n")
    
    return "".join(md_parts)

# -----------------------------
# MAIN EXPORTED FUNCTION
# -----------------------------
def json_to_markdown(data: Dict[Any, Any], md_path: Optional[str] = None) -> str:
    """
    Converts a structured document JSON, including nested blocks, into well-organized Markdown.
    """
    document = data.get("document", {})
    all_blocks = []
    for page in document.get("pages", []):
        all_blocks.extend(page.get("blocks", []))
        
    markdown_text = render_blocks(all_blocks).strip() + "\n"

    if md_path:
        Path(md_path).write_text(markdown_text, encoding="utf-8")

    return markdown_text