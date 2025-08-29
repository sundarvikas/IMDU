from pathlib import Path
import google.generativeai as genai

# -----------------------------
# CONFIG
# -----------------------------
API_KEY = "AIzaSyCPksKa-eqaDHK6dMRVIeJMxUwhcbalw3U"   # Replace with your actual API key
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
You are a highly advanced document intelligence engine. Your task is to analyze any given document—including multi-lingual text, tables, charts, flowcharts, and handwriting—and convert it into a single, clean, structured JSON output that preserves its content, layout, and semantic meaning.

The JSON output must follow this exact format:

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
            "content": "Example Section",
            "bbox": [0.1, 0.2, 0.8, 0.3],
            "languages": ["en"],
            "overall_confidence": 0.95
          }}
        ]
      }}
    ]
  }}
}}

**Rules for generating the JSON (in order of priority):**

1.  **General Structure:** The root must be a `document` object. Each content element is a `block` with a unique `id`, `type`, `content`, `bbox`, `languages`, and `overall_confidence`.

2.  **Recursive Image Analysis:** If a block on the page is an image, chart, map, or a photo of a whiteboard/slide, its `type` must be **`image_container`**. The `content` for this type must be a JSON object with two keys:
    * `"description"`: A short, natural language summary of the image itself (e.g., "A photograph of a whiteboard with notes").
    * `"blocks"`: An array where you will recursively analyze the content **within** that image. You must apply all the other rules below (for tables, flowcharts, handwriting, etc.) to the content you find inside the image. The `bbox` for these nested blocks should be relative to the parent image.

3.  **Medical Prescriptions:** If the document (or a nested image) is a medical prescription, a block's `type` must be "prescription". The `content` must be a **nested JSON object** structured for easy patient understanding: 
    `{{"patient_name": "...", "doctor_name": "...", "date": "...", "medications": [{{"name": "...", "dosage": "...", "frequency": "..."}}]}}`

4.  **Tables:** If a block is a table, its `type` must be "table" and its `content` must be an object: `{{"headers": [...], "rows": [...]}}`.

5.  **Flowcharts:** If a block is a flowchart or diagram, its `type` must be "flowchart". The `content` must be a nested Markdown list that accurately represents the flowchart's structure.

6. **Data Charts (Pie, Bar, Line):** If a block (or a nested block inside an image_container) is a data chart, its `type` MUST be its specific kind (e.g., `pie_chart`, `bar_chart`, `line_graph`). The `content` MUST be a JSON object containing the chart's `title`, a list of `labels` for the categories, and a list of `data` for the numerical values. **Do not just describe the chart.**`{{"title": "Sales by Region", "labels": ["North America", "Europe", "APAC"], "data": [4.2, 3.5, 2.1]}}`

7.  **Handwritten Text:** If a block is an image containing significant handwritten text, its `type` must be "handwritten_text". The `content` must be a string containing the exact transcribed text.

8.  **Standard Text:** Use standard types like `title`, `section_header`, `paragraph`, and `list_item` for all other text elements.

9.  **Final Output:** Your entire response must be **only the valid JSON content**. Do not include any commentary, explanations, or enclosing ```json``` tags.
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