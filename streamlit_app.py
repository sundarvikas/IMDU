# streamlit_app.py
import streamlit as st
from pathlib import Path
import json
from datetime import datetime
import google.generativeai as genai
# --- NEW IMPORTS for Table Processing ---
import pandas as pd
from io import BytesIO

# -----------------------------
# GEMINI SETUP - Direct API Key
# -----------------------------
# ⚠️ WARNING: Only for demo/hackathon use. Never expose API keys in production.
GEMINI_API_KEY = "AIzaSyCsFipgzT4QDCUu7qd1bN-QX7mTcTtJ_Ys"  # Directly embedded

try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"❌ Failed to configure Gemini: {str(e)}")
    st.stop()

# Import your pipeline
try:
    from imdu_pipeline import process_document
except Exception as e:
    st.error(f"❌ Error loading pipeline: {e}")
    st.info("Make sure `imdu_pipeline.py` is in the same directory and has `process_document()`.")
    st.stop()

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="IMDU Document Parser",
    page_icon="📄",
    layout="wide"
)

# -----------------------------
# CUSTOM CSS (Dark Mode Friendly)
# -----------------------------
st.markdown("""
<style>
    /* Your existing CSS goes here */
    .main .block-container { padding-top: 1rem; }
    .scroll-box {
        max-height: 700px;
        overflow-y: auto;
        border: 1px solid #ddd;
        border-radius: 12px;
        padding: 16px;
        background-color: #f8f9fa;
        font-family: 'Courier New', monospace;
        line-height: 1.6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    @media (prefers-color-scheme: dark) {
        .scroll-box {
            background-color: #2d2d2d;
            color: #ddd;
            border-color: #444;
        }
    }
    .markdown-body {
        line-height: 1.8;
        font-size: 1.1em;
        color: #222;
    }
    @media (prefers-color-scheme: dark) {
        .markdown-body {
            color: #eee;
        }
    }
    .upload-area-modern {
        border: 2px dashed var(--primary-color, #1f77b4);
        border-radius: 16px;
        padding: 3rem 1.5rem;
        text-align: center;
        background-color: var(--background-color-upload, #f8f9fa);
        color: var(--text-color-upload, #333);
        transition: all 0.2s ease;
        cursor: pointer;
        margin-bottom: 1.5rem;
    }
    .upload-icon {
        font-size: 3rem;
        margin-bottom: 0.5rem;
        color: var(--primary-color, #1f77b4);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def get_summary_stats(json_data):
    """
    Extracts content type counts from the parsed JSON structure.
    """
    type_counts = {}
    try:
        pages = json_data.get("document", {}).get("pages", [])
        for page in pages:
            for block in page.get("blocks", []):
                block_type = block.get("type", "unknown").title()
                type_counts[block_type] = type_counts.get(block_type, 0) + 1
    except Exception as e:
        st.warning(f"Error parsing JSON structure: {str(e)}")
        return {"Error": 1}
    return type_counts

# --- NEW: Helper function to convert DataFrame to Excel in memory ---
def to_excel(df: pd.DataFrame):
    output = BytesIO()
    with pd.ExcelWriter(output) as writer:  # Removed engine='xlsxwriter'
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    processed_data = output.getvalue()
    return processed_data

# --- NEW: Helper function to find all tables within the parsed JSON ---
def find_tables_in_json(json_data):
    """
    Parses the JSON output and extracts all blocks with type 'table'.
    Returns a list of Pandas DataFrames.
    """
    table_dfs = []
    try:
        pages = json_data.get("document", {}).get("pages", [])
        for page in pages:
            for block in page.get("blocks", []):
                if block.get("type") == "table":
                    headers = block.get("content", {}).get("headers", [])
                    rows = block.get("content", {}).get("rows", [])
                    if headers and rows:
                        df = pd.DataFrame(rows, columns=headers)
                        table_dfs.append(df)
    except Exception as e:
        st.warning(f"Error extracting tables from JSON: {e}")
    return table_dfs

# -----------------------------
# SESSION STATE INITIALIZATION
# -----------------------------
if "document_history" not in st.session_state:
    st.session_state.document_history = {}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "ai_summary" not in st.session_state:
    st.session_state.ai_summary = None

# --- NEW: Add session state for JSON-derived tables ---
if "json_tables" not in st.session_state:
    st.session_state.json_tables = []

# -----------------------------
# HEADER & FILE UPLOADER
# -----------------------------
st.markdown("<h1 style='text-align: center;'>📄 INTELLIGENT MULTI-LINGUAL DOCUMENT UNDERSTANDING</h1>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; font-size: 1.1rem; color: #555; margin-bottom: 1.5rem;">
Upload a PDF, image, or Word document to extract structured content.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="upload-area-modern">
    <div class="upload-icon">📁</div>
    <div class="upload-instruction">Drag & drop your document here</div>
    <div class="upload-note">Supports: PDF, PNG, JPG, DOCX</div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload your document",
    type=["pdf", "png", "jpg", "jpeg", "docx"],
    label_visibility="collapsed"
)

if uploaded_file is None:
    st.info("📤 Upload a file to get started.")
else:
    uploads_dir = Path("uploads")
    output_dir = Path("output")
    uploads_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    file_path = uploads_dir / uploaded_file.name
    stem = file_path.stem

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"✅ **{uploaded_file.name}** uploaded.")

    # Preview Section
    st.markdown("### 🔍 Document Preview")
    if uploaded_file.type.startswith("image"):
        st.image(file_path, caption="📷 Image Preview", use_container_width=True)
    elif uploaded_file.type == "application/pdf":
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(file_path)
            page = doc.load_page(0)
            pix = page.get_pixmap(dpi=96)
            img_bytes = pix.tobytes("png")
            st.image(img_bytes, caption="📄 PDF Page 1 Preview", use_column_width=True)
        except Exception as e:
            st.info(f"Preview not available for this PDF. Error: {str(e)}")
    elif uploaded_file.name.endswith(".docx"):
        st.info("📄 Word document uploaded. Preview not available.")

    if st.button("✨ Parse Document", key="parse", type="primary", use_container_width=True):
        with st.spinner("🧠 Analyzing document... This may take a few moments."):
            try:
                # Reset previous results
                st.session_state.ai_summary = None
                st.session_state.json_tables = []
                
                json_data, markdown_text = process_document(
                    file_path=str(file_path),
                    json_path=str(output_dir / f"{stem}.json"),
                    md_path=str(output_dir / f"{stem}.md")
                )

                # --- NEW: Extract tables directly from the new JSON data ---
                st.session_state.json_tables = find_tables_in_json(json_data)
                
                st.session_state.json_data = json_data
                st.session_state.markdown_text = markdown_text
                st.session_state.filename = stem
                st.session_state.processed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                st.session_state.document_history[stem] = {
                    "json_data": json_data,
                    "markdown_text": markdown_text,
                    "timestamp": st.session_state.processed_at,
                    "translated": {}
                }
                st.toast("✅ Parsing complete!", icon="🎉")
            except Exception as e:
                st.error(f"❌ Processing failed: {str(e)}")
                st.exception(e)

# -----------------------------
# DISPLAY RESULTS (if processed)
# -----------------------------
if "json_data" in st.session_state:
    json_data = st.session_state.json_data
    markdown_text = st.session_state.markdown_text
    filename = st.session_state.filename

    st.markdown("---")
    st.markdown(f"<div class='result-header'>Parsed Output</div>", unsafe_allow_html=True)

    # --- UPDATED: Added a new tab for Table Data ---
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📑 Formatted Content",
        "🤖 AI Summary",
        "📊 Extracted Tables",
        "📦 Structured Data",
        "⬇️ Export",
        "💬 Chat with Document"
    ])

    with tab1:
        st.markdown("### 📝 Extracted Document (Markdown)")
        with st.expander("View Full Output", expanded=True):
            st.markdown('<div class="scroll-box markdown-body">', unsafe_allow_html=True)
            st.markdown(markdown_text)
            st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown("### 🧠 AI-Generated Summary")
        if st.session_state.ai_summary:
            st.markdown(st.session_state.ai_summary)
            if st.button("🔄 Regenerate Summary", key="regen_summary"):
                st.session_state.ai_summary = None
                st.rerun()
        else:
            st.info("Click the button below to generate a summary of the document.")
            if st.button("✨ Generate Summary", use_container_width=True, type="primary"):
                with st.spinner("✍️ Generating summary..."):
                    try:
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        prompt = f"""
                        You are an expert summarizer. Your task is to provide a concise, professional summary of the following document content.

                        Instructions:
                        1. Read the entire document content carefully.
                        2. Identify the main topic and purpose of the document.
                        3. Extract the 3-5 most important key points, findings, or conclusions.
                        4. Present the summary in a clean, easy-to-read format. Start with a brief overview paragraph, followed by a bulleted list of the key points.
                        5. Ensure the summary is objective and accurately reflects the source material.

                        Document Content:
                        ```
                        {markdown_text}
                        ```
                        """
                        response = model.generate_content(prompt)
                        summary_text = response.text.strip()
                        st.session_state.ai_summary = summary_text
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Could not generate summary: {str(e)}")

    # --- NEW: EXTRACTED TABLES TAB (from JSON) ---
    with tab3:
        st.markdown("### 📊 Extracted Tables")
        if st.session_state.json_tables:
            tables = st.session_state.json_tables
            st.success(f"Displaying **{len(tables)}** table(s) found in the document.")
            for i, df in enumerate(tables):
                st.markdown(f"---")
                st.markdown(f"#### Table {i+1}")
                st.dataframe(df)
                col1, col2 = st.columns(2)
                with col1:
                    csv_data = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label=f"⬇️ Download Table {i+1} as CSV",
                        data=csv_data,
                        file_name=f"{filename}_table_{i+1}.csv",
                        mime='text/csv',
                        use_container_width=True,
                        key=f"csv_{i}"
                    )
                with col2:
                    excel_data = to_excel(df)
                    st.download_button(
                        label=f"⬇️ Download Table {i+1} as Excel",
                        data=excel_data,
                        file_name=f"{filename}_table_{i+1}.xlsx",
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        use_container_width=True,
                        key=f"excel_{i}"
                    )
        else:
            st.info("No tables were identified in the document's JSON structure.")

    with tab4:
        st.markdown("### 🔍 JSON Structure")
        with st.expander("Show Full JSON", expanded=False):
            st.markdown('<div class="scroll-box">', unsafe_allow_html=True)
            st.json(json_data)
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("##### 📊 Content Summary")
        stats = get_summary_stats(json_data)
        items = list(stats.items())
        for i in range(0, len(items), 4):
            cols = st.columns(4)
            for j in range(4):
                idx = i + j
                if idx < len(items):
                    label, value = items[idx]
                    cols[j].metric(label=label, value=value)
        total_elements = sum(stats.values())
        st.caption(f"📄 Total content blocks detected: {total_elements}")

    with tab5:
        st.markdown("### 💾 Download Results")
        json_str = json.dumps(json_data, indent=2, ensure_ascii=False)
        markdown_str = markdown_text
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "⬇️ Download JSON",
                json_str,
                f"{filename}.json",
                "application/json",
                use_container_width=True
            )
        with col2:
            st.download_button(
                "⬇️ Download Markdown",
                markdown_str,
                f"{filename}.md",
                "text/markdown",
                use_container_width=True
            )
        st.markdown("### 🌍 Translate & Export")
        target_lang = st.selectbox(
            "Select Language",
            ["Hindi", "Tamil", "Telugu", "French", "Spanish", "German", "Arabic"],
            key="trans_lang"
        )
        lang_code_map = {
            "Hindi": "hi", "Tamil": "ta", "Telugu": "te",
            "French": "fr", "Spanish": "es", "German": "de", "Arabic": "ar"
        }
        lang_code = lang_code_map[target_lang]
        if st.button(f"🔁 Translate to {target_lang}", key="trans_btn"):
            with st.spinner(f"Translating to {target_lang}..."):
                try:
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    prompt = f"""
                    You are a professional document translator.
                    Translate the following Markdown content into {target_lang} ({lang_code}).
                    Preserve ALL structure exactly:
                    - Headings (#, ##, ###)
                    - Lists (- or 1.)
                    - Tables (format with |)
                    - Code blocks
                    - Emphasis (**bold**, *italic*)

                    Return ONLY the translated Markdown.
                    Do NOT add explanations, comments, or formatting notes.

                    Content to translate:
                    ```
                    {markdown_text}
                    ```
                    """
                    response = model.generate_content(prompt)
                    translated_md = response.text.strip()
                    if not translated_md or len(translated_md) < 5:
                        raise Exception("Empty or invalid translation received")
                    st.session_state.document_history[filename]["translated"][lang_code] = translated_md
                    st.success("✅ Translation complete!")
                    st.download_button(
                        f"⬇️ Download {target_lang} Markdown",
                        translated_md,
                        f"{filename}_{lang_code}.md",
                        "text/markdown",
                        use_container_width=True,
                        key=f"download_trans_{lang_code}_{filename}"
                    )
                    with st.expander("📄 Preview Translated Document"):
                        st.markdown('<div class="scroll-box markdown-body">', unsafe_allow_html=True)
                        st.markdown(translated_md)
                        st.markdown('</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"❌ Translation failed: {str(e)}")

    with tab6:
        st.markdown("### 💬 Ask About This Document")
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        user_input = st.chat_input("Ask a question about this document...")
        if user_input:
            with st.chat_message("user"):
                st.write(user_input)
            with st.chat_message("assistant"):
                with st.spinner("🧠 Thinking..."):
                    try:
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        response = model.generate_content([
                            f"{user_input}, Answer based only on this document. Be concise.\n\nDocument:\n{markdown_text}",
                            user_input
                        ])
                        answer = response.text
                    except Exception as e:
                        answer = f"❌ Error: {str(e)}"
                st.write(answer)
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

# -----------------------------
# SIDEBAR: Document History
# -----------------------------
if st.session_state.document_history:
    with st.sidebar:
        st.header("📁 Recent Documents")
        for name, data in reversed(list(st.session_state.document_history.items())):
            with st.expander(f"📄 {name}"):
                st.caption(f"📅 {data['timestamp']}")
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "📥 JSON",
                        json.dumps(data["json_data"], indent=2),
                        f"{name}.json",
                        "application/json",
                        use_container_width=True,
                        key=f"hist_json_{name}"
                    )
                with col2:
                    st.download_button(
                        "📝 MD",
                        data["markdown_text"],
                        f"{name}.md",
                        "text/markdown",
                        use_container_width=True,
                        key=f"hist_md_{name}"
                    )
                if data["translated"]:
                    st.markdown("**Translations:**")
                    for lang_code, trans_text in data["translated"].items():
                        st.download_button(
                            f"🌍 {lang_code.upper()}",
                            trans_text,
                            f"{name}_{lang_code}.md",
                            "text/markdown",
                            use_container_width=True,
                            key=f"trans_{name}_{lang_code}"
                        )

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown(
    "<p class='footer'>"
    "✨ Document parsing with IMDU | "
    f"Processed on {datetime.now().strftime('%Y-%m-%d')} | "
    "TEAM CODEX"
    "</p>",
    unsafe_allow_html=True
)