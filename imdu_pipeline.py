# imdu_pipeline.py
from pathlib import Path
import sys
from typing import Tuple, Dict, Any, Optional

# Import your modules
from pdf_sender import pdf_sender
from image_sender import image_sender
from text_to_json import text_to_json
from json_to_markdown import json_to_markdown

# -----------------------------
# CONFIGURATION
# -----------------------------
# Supported extensions
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.webp', '.gif'}
PDF_EXTS = {'.pdf'}
DOCX_EXTS = {'.docx'}

# Output directories
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


# -----------------------------
# MAIN FUNCTION
# -----------------------------
def process_document(
    file_path: str,
    json_path: Optional[str] = None,
    md_path: Optional[str] = None
) -> Tuple[Optional[Dict[Any, Any]], Optional[str]]:
    """
    Unified pipeline to process PDF, image, or DOCX → JSON + Markdown.

    Args:
        file_path (str): Path to input file.
        json_path (str, optional): Output path for JSON. Defaults to output/{name}.json
        md_path (str, optional): Output path for Markdown. Defaults to output/{name}.md

    Returns:
        (json_data, markdown_text)
    """
    input_path = Path(file_path).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    ext = input_path.suffix.lower()
    name = input_path.stem

    # Set default output paths
    json_path = json_path or OUTPUT_DIR / f"{name}.json"
    md_path = md_path or OUTPUT_DIR / f"{name}.md"

    print(f"📄 Processing: {input_path.name} ({ext})")

    # Step 1: Get raw text from Gemini
    try:
        if ext in IMAGE_EXTS:
            print("🖼️ Sending image to IMDU...")
            raw_text = image_sender(str(input_path))

        elif ext in PDF_EXTS:
            print("📑 Sending PDF to IMDU...")
            raw_text = pdf_sender(str(input_path))

        elif ext in DOCX_EXTS:
            print("📄 Converting DOCX to PDF...")
            try:
                from docx2pdf import convert
                pdf_path = input_path.with_suffix(".pdf")
                convert(str(input_path), str(pdf_path))
                print(f"✅ Converted to: {pdf_path.name}")
                raw_text = pdf_sender(str(pdf_path))
                # Optional: clean up temp PDF
                # pdf_path.unlink()
            except ImportError:
                raise ImportError(
                    "Missing 'docx2pdf'. Install with: pip install docx2pdf"
                )
        else:
            raise ValueError(f"Unsupported file type: {ext}. "
                           f"Supported: {IMAGE_EXTS | PDF_EXTS | DOCX_EXTS}")

        print("✅ IMDU response received.")

    except Exception as e:
        print(f"❌ Error during IMDU processing: {e}")
        raise

    # Step 2: Convert raw text → structured JSON
    try:
        json_data = text_to_json(raw_text, json_path=json_path)
        print(f"✅ JSON saved to: {json_path}")
    except Exception as e:
        print(f"❌ Failed to parse JSON: {e}")
        raise

    # Step 3: Convert JSON → Markdown
    try:
        markdown_text = json_to_markdown(json_data, md_path=md_path)
        print(f"✅ Markdown saved to: {md_path}")
    except Exception as e:
        print(f"❌ Failed to generate Markdown: {e}")
        raise

    return json_data, markdown_text


# -----------------------------
# CLI Usage
# -----------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python imdu_pipeline.py <file_path>")
        print("Example: python imdu_pipeline.py uploads/report.pdf")
        sys.exit(1)

    file_path = sys.argv[1]

    try:
        json_data, markdown = process_document(file_path)
        print("\n🎉 Processing complete!")
        print(f"📄 JSON: {OUTPUT_DIR / Path(file_path).stem}.json")
        print(f"📝 Markdown: {OUTPUT_DIR / Path(file_path).stem}.md")
    except Exception as e:
        print(f"\n💥 Failed to process {file_path}: {e}")
        sys.exit(1)