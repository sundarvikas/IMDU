# image_sender.py
from pathlib import Path
import google.generativeai as genai  # ✅ Correct import

API_KEY = "AIzaSyDzCbh41NS04x2kYdJXOshO296fVxII9yY"  # Replace with your actual API key
MODEL_NAME = "gemini-1.5-flash"  # or gemini-1.5-pro-latest

# ✅ Configure globally (only once per script)
genai.configure(api_key=API_KEY)

def image_sender(image_path: str) -> str:
    """
    Sends an image to Gemini and returns the structured OCR JSON text.
    """
    image_file = Path(image_path)
    
    if not image_file.exists():
        raise FileNotFoundError(f"Image not found: {image_file}")

    # ✅ Upload the file using genai.upload_file
    print("📤 Uploading image...")
    uploaded_file = genai.upload_file(path=image_file, mime_type="image/jpeg")
    print(f"✅ Uploaded: {uploaded_file.display_name} (URI: {uploaded_file.uri})")

    # Wait for processing
    import time
    while uploaded_file.state.name == "PROCESSING":
        print("⏳ Processing image...")
        time.sleep(5)
        uploaded_file = genai.get_file(uploaded_file.name)

    if uploaded_file.state.name == "FAILED":
        raise ValueError("File upload failed.")

    # Prompt
    prompt = f"""
You are an intelligent OCR post-processor.
Given OCR + layout extraction results, generate a clean structured JSON in the following format:

{{
  "document": {{
    "doc-name": "{uploaded_file.display_name}",
    "pages": [
      {{
        "page_index": 1,
        "blocks": [
          {{
            "id": "b1",
            "type": "section_header",
            "content": "Example section",
            "bbox": [0.1, 0.2, 0.8, 0.3],
            "languages": ["en"],
            "overall_confidence": 0.95
          }}
        ]
      }}
    ]
  }}
}}

Rules:
- doc-name must be the original file name.
- Each page must have page_index.
- Each block has id, type, content, bbox, languages, overall_confidence.
- Tables must be represented as {{ "headers": [...], "rows": [...] }}.
- If a block has no text (e.g. picture), content = "" and languages = [].
- Do not include any extra commentary, only valid JSON output.
"""

    # Generate content
    print("🧠 Asking Gemini to extract structured data...")
    model = genai.GenerativeModel(model_name=MODEL_NAME)
    response = model.generate_content(
        contents=[prompt, uploaded_file],
        request_options={"timeout": 600}
    )

    if response.candidates:
        return response.text
    else:
        return f"❌ No response. Blocked? {response.prompt_feedback}"

# Example usage
if __name__ == "__main__":
    result = image_sender("page_1.jpg")  # Replace with your image path
    print(result)