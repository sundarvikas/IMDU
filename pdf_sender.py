from pathlib import Path
import google.generativeai as genai

# -----------------------------
# CONFIG
# -----------------------------
API_KEY = "AIzaSyDqoS0L2DJ2IaKmJdC-BopHUDK-srerWzo"   # Replace with your actual API key
MODEL_NAME = "gemini-1.5-flash"  # or gemini-1.5-pro-latest

# Initialize the API (don't assign the result!)
genai.configure(api_key=API_KEY)

# -----------------------------
# PDF Sender Function
# -----------------------------
def pdf_sender(pdf_path: str) -> str:
    """
    Uploads a PDF to Gemini and returns the structured JSON response.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Gemini response text (structured JSON string).
    """
    pdf_file = Path(pdf_path)

    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_file}")

    # ✅ Use genai directly: Upload the file
    print("📤 Uploading file...")
    uploaded_file = genai.upload_file(path=pdf_file, mime_type="application/pdf")
    print(f"✅ Uploaded: {uploaded_file.display_name} (URI: {uploaded_file.uri})")

    # Wait for processing (optional but recommended)
    import time
    while uploaded_file.state.name == "PROCESSING":
        print("⏳ Processing file...")
        time.sleep(5)
        uploaded_file = genai.get_file(uploaded_file.name)

    if uploaded_file.state.name == "FAILED":
        raise ValueError(f"File processing failed: {uploaded_file.state.name}")

    # ✅ Build the prompt
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

    # ✅ Generate content using the model
    print("🧠 Generating response...")
    model = genai.GenerativeModel(model_name=MODEL_NAME)
    response = model.generate_content(
        contents=[prompt, uploaded_file],
        request_options={"timeout": 600}
    )

    if response.candidates:
        return response.text
    else:
        return f"❌ No response generated. Safety reasons: {response.prompt_feedback}"

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    try:
        # ❌ This path is from Colab: "/content/..."
        # On Windows, you need a real local path like:
        # "C:/Users/sunda/Desktop/Hackathon_Team_Roles_and_Responsibilities.pdf"
        pdf_path = "Hackathon_Team_Roles_and_Responsibilities.pdf"  # Assumes file is in same folder

        # Or full path:
        # pdf_path = r"C:\Users\sunda\OneDrive\Desktop\gemini OCR\Hackathon_Team_Roles_and_Responsibilities.pdf"

        text_output = pdf_sender(pdf_path)
        print("💡 Response:")
        print(text_output)

    except Exception as e:
        print(f"Error: {e}")