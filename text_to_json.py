# text_to_json.py
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

def text_to_json(text: str, json_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Convert Gemini's raw text response to structured JSON.
    
    Handles:
      - Markdown code blocks (```json ... ```)
      - Extra text before/after JSON
      - Malformed JSON fallback
    
    Args:
        text (str): Raw output from Gemini.
        json_path (str, optional): Path to save JSON file.
    
    Returns:
        dict: Parsed JSON data. If parsing fails, returns {"raw_text": "..."}.
    """
    if not text or not text.strip():
        raise ValueError("Empty or None text input")

    clean_text = text.strip()

    # Remove Markdown code fences: ```json ... ``` or ```json\n{...}\n```
    code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean_text, re.IGNORECASE)
    if code_block_match:
        clean_text = code_block_match.group(1).strip()
    else:
        # Try to extract the first JSON object
        json_match = re.search(r"\{[\s\S]*\}", clean_text)
        if json_match:
            clean_text = json_match.group().strip()

    try:
        data = json.loads(clean_text)
        # Ensure top-level key is "document"
        if "document" not in data:
            data = {"document": {"doc-name": "unknown", "pages": [], "raw_extract": True}}
        if json_path:
            Path(json_path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return data
    except json.JSONDecodeError as e:
        print(f"[⚠️] JSON parsing failed: {e}")
        fallback = {"raw_text": clean_text, "parsing_error": str(e)}
        if json_path:
            Path(json_path).write_text(json.dumps(fallback, indent=2, ensure_ascii=False), encoding="utf-8")
        return fallback