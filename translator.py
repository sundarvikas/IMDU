from pathlib import Path
import google.generativeai as genai

# -----------------------------
# CONFIG
# -----------------------------
API_KEY = "AIzaSyBfdXelTH__x7ZsVEp9D31p5Dm_mz1kADE"
MODEL_NAME = "gemini-2.5-flash"  # or gemini-1.5-pro-latest

# Initialize the API
genai.configure(api_key=API_KEY)

# -----------------------------
# MARKDOWN TRANSLATOR FUNCTION
# -----------------------------
def translate_markdown(text: str, target_language: str, lang_code: str) -> str:
    """
    Translates Markdown text to the target language using Gemini.
    Returns only clean, structured Markdown — no explanations.

    Args:
        text (str): Input Markdown text.
        target_language (str): Full name of target language (e.g., "Hindi", "French")
        lang_code (str): ISO code (e.g., "hi", "fr")

    Returns:
        str: Translated Markdown string (no extra text).
    """
    if not text.strip():
        return ""

    print(f"🌍 Translating to {target_language} ({lang_code})...")

    prompt = f"""
You are a professional document translator.
Translate the following Markdown content into {target_language} ({lang_code}).
Preserve ALL formatting and structure exactly:
- Headings (#, ##, ###)
- Lists (- or 1.)
- Tables (format with |)
- Code blocks (```)
- Bold/italic (**text**, *text*)
- Links and images

Return ONLY the translated Markdown.
Do NOT add any comments, explanations, or notes.
Do NOT say 'Here is the translation:' or similar.

Content to translate: {text}
"""
    print("📜 Preparing translation prompt...")

    try:
        model = genai.GenerativeModel(model_name=MODEL_NAME)
        response = model.generate_content(
            prompt,
            request_options={"timeout": 600}
        )

        if response.candidates:
            translated = response.text.strip()
            if not translated:
                raise ValueError("Empty response from Gemini")
            return translated
        else:
            feedback = response.prompt_feedback
            return f"❌ Translation failed. Blocked: {feedback}"

    except Exception as e:
        return f"❌ Error: {str(e)}"


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Example Markdown input
    sample_markdown = """
# MEFA Student Loan Options

The Massachusetts Educational Financing Authority (MEFA) offers low-interest student loans.

| Loan Type       | Interest Rate |
|-----------------|---------------|
| College Loan    | 4.5%          |
| PLUS Loan       | 5.0%          |

## Application Process
1. Visit the MEFA website
2. Fill out the online form
3. Upload financial documents
"""

    # Translate to Hindi
    translated_md = translate_markdown(sample_markdown, target_language="Hindi", lang_code="hi")

    print("\n💡 Translated Markdown:")
    print(translated_md)

    # Optionally save to file
    output_path = Path("translated_output.md")
    output_path.write_text(translated_md, encoding="utf-8")
    print(f"\n✅ Saved to: {output_path}")