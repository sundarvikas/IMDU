# streamlit_app.py
import streamlit as st
from pathlib import Path
import json
from datetime import datetime
import google.generativeai as genai
# --- NEW IMPORTS for Table Processing ---
import pandas as pd
from io import BytesIO
import tempfile

# --- NEW: Import our email sending function ---
from email_report import send_report_email

# -----------------------------
# GEMINI SETUP - Direct API Key
# -----------------------------
# ⚠️ WARNING: Only for demo/hackathon use. Never expose API keys in production.
GEMINI_API_KEY = "AIzaSyBVFn8ZtXlPJNaVblV0aolqeyhVx6mW-zA"  # Directly embedded

try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    st.info(f"Failed to configure Gemini: {str(e)}")  # was st.error
    st.stop()

# Import your pipeline
try:
    from imdu_pipeline import process_document
except Exception as e:
    st.info(f"Error loading pipeline: {e}")  # was st.error
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
# HELPER FUNCTIONS
# -----------------------------
def to_excel(df_list: list) -> bytes:
    """
    Writes a list of DataFrames to separate sheets in an in-memory Excel file.
    Uses openpyxl engine (comes with pandas) for better compatibility.
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for i, df in enumerate(df_list):
            df.to_excel(writer, index=False, sheet_name=f'Table_{i+1}')
    output.seek(0)
    return output.read()

def find_tables_in_json(json_data):
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
# HEADER & FILE UPLOADER (normalized)
# -----------------------------
st.markdown("<h1 style='text-align: center; font-size:2rem; font-weight:600; margin-bottom:0.5rem;'>IMDU Document Parser</h1>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; font-size:1rem; color:#444; margin-bottom:1rem;'>Upload a PDF, image, or Word document to extract structured content.</div>", unsafe_allow_html=True)
st.markdown("""
<div>
    <div style="font-size:2rem; margin-bottom:0.5rem;">&#128194;</div>
    <div>Drag & drop your document here</div>
    <div>Supported formats: PDF, PNG, JPG, DOCX</div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload your document",
    type=["pdf", "png", "jpg", "jpeg", "docx"],
    label_visibility="collapsed"
)

if uploaded_file is None:
    st.info("Upload a file to get started.")
else:
    uploads_dir = Path("uploads")
    output_dir = Path("output")
    uploads_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    file_path = uploads_dir / uploaded_file.name
    stem = file_path.stem

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"{uploaded_file.name} uploaded.")

    # Preview Section
    st.markdown("### Document Preview")
    if uploaded_file.type.startswith("image"):
        st.image(file_path, caption="Image Preview", use_container_width=True)
    elif uploaded_file.type == "application/pdf":
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(file_path)
            page = doc.load_page(0)
            pix = page.get_pixmap(dpi=96)
            img_bytes = pix.tobytes("png")
            st.image(img_bytes, caption="PDF Page 1 Preview", use_column_width=True)
        except Exception as e:
            st.info(f"Preview not available for this PDF. Error: {str(e)}")
    elif uploaded_file.name.endswith(".docx"):
        st.info("Word document uploaded. Preview not available.")

    if st.button("Parse Document", key="parse", type="primary", use_container_width=True):
        with st.spinner("Analyzing document... This may take a few moments."):
            try:
                # Reset previous results
                st.session_state.ai_summary = None
                st.session_state.json_tables = []
                
                json_data, markdown_text = process_document(
                    file_path=str(file_path),
                    json_path=str(output_dir / f"{stem}.json"),
                    md_path=str(output_dir / f"{stem}.md")
                )

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

                # --- AUTOMATIC SUMMARY GENERATION ---
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
                except Exception as e:
                    st.session_state.ai_summary = None
                    st.warning(f"Automatic summary generation failed: {str(e)}")  # already blue/yellow

                st.toast("Parsing complete!")
            except Exception as e:
                st.info(f"Processing failed: {str(e)}")  # was st.error
                st.exception(e)

def get_summary_stats(json_data: dict) -> dict:
    """
    Extracts content type counts from the parsed JSON structure.
    
    This function navigates through the nested document -> pages -> blocks
    structure, counts each block by its 'type' attribute, and returns
    a dictionary of the counts.

    Args:
        json_data (dict): The JSON data loaded as a Python dictionary.

    Returns:
        dict: A dictionary where keys are block types (e.g., 'Table')
              and values are their counts (e.g., 2).
    """
    type_counts = {}
    try:
        # Safely access the list of pages, defaulting to an empty list
        pages = json_data.get("document", {}).get("pages", [])
        
        # Loop through each page in the document
        for page in pages:
            # Loop through each content block on the page
            for block in page.get("blocks", []):
                # Get the block type, default to "Unknown", and capitalize it
                block_type = block.get("type", "unknown").title()
                
                # Increment the counter for this block type
                type_counts[block_type] = type_counts.get(block_type, 0) + 1
                
    except Exception as e:
        # If any error occurs during parsing, show a warning in the UI
        st.warning(f"Error parsing JSON structure for stats: {str(e)}")
        return {"Error": 1}
        
    return type_counts


# -----------------------------
# DISPLAY RESULTS (if processed)
# -----------------------------
if "json_data" in st.session_state:
    json_data = st.session_state.json_data
    markdown_text = st.session_state.markdown_text
    filename = st.session_state.filename

    st.markdown("---")
    st.markdown(f"<div class='result-header'>Parsed Output</div>", unsafe_allow_html=True)

    if st.session_state.json_data:
        tab_titles = [
            "Formatted Content",
            "AI Summary",
            "Extracted Tables",
            "Structured Data",
            "Export",
            "Chat with Document",
            "Email Report"
        ]
        tabs = st.tabs(tab_titles)

        with tabs[0]:
            st.markdown("### Extracted Document (Markdown)")
            with st.expander("View Full Output", expanded=True):
                st.markdown('<div class="scroll-box markdown-body">', unsafe_allow_html=True)
                st.markdown(markdown_text)
                st.markdown('</div>', unsafe_allow_html=True)

        with tabs[1]:
            st.markdown("### AI-Generated Summary")
            if st.session_state.ai_summary:
                st.markdown(st.session_state.ai_summary)
                if st.button("Regenerate Summary", key="regen_summary"):
                    st.session_state.ai_summary = None
                    st.rerun()
            else:
                st.info("Click the button below to generate a summary of the document.")
                if st.button("Generate Summary", use_container_width=True, type="primary"):
                    with st.spinner("Generating summary..."):
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
                            st.info(f"Could not generate summary: {str(e)}")  # was st.error

        with tabs[2]:
            st.markdown("### Extracted Tables")
            if st.session_state.json_tables:
                tables = st.session_state.json_tables
                st.success(f"Displaying {len(tables)} table(s) found in the document.")
                for i, df in enumerate(tables):
                    st.markdown(f"---")
                    st.markdown(f"#### Table {i+1}")
                    st.dataframe(df)
                    col1, col2 = st.columns(2)
                    with col1:
                        csv_data = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label=f"Download Table {i+1} as CSV",
                            data=csv_data,
                            file_name=f"{filename}_table_{i+1}.csv",
                            mime='text/csv',
                            use_container_width=True,
                            key=f"csv_{i}"
                        )
                    with col2:
                        excel_data = to_excel([df])
                        st.download_button(
                            label=f"Download Table {i+1} as Excel",
                            data=excel_data,
                            file_name=f"{filename}_table_{i+1}.xlsx",
                            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                            use_container_width=True,
                            key=f"excel_{i}"
                        )
            else:
                st.info("No tables were identified in the document's JSON structure.")

        with tabs[3]:
            st.markdown("### JSON Structure")
            with st.expander("Show Full JSON", expanded=False):
                st.markdown('<div class="scroll-box">', unsafe_allow_html=True)
                st.json(json_data)
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("##### Content Summary")
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
            st.caption(f"Total content blocks detected: {total_elements}")

        with tabs[4]:
            st.markdown("### Download Results")
            json_str = json.dumps(json_data, indent=2, ensure_ascii=False)
            markdown_str = markdown_text
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "Download JSON",
                    json_str,
                    f"{filename}.json",
                    "application/json",
                    use_container_width=True
                )
            with col2:
                st.download_button(
                    "Download Markdown",
                    markdown_str,
                    f"{filename}.md",
                    "text/markdown",
                    use_container_width=True
                )
            st.markdown("### Translate & Export")
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
            if st.button(f"Translate to {target_lang}", key="trans_btn"):
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
                        st.success("Translation complete!")
                        st.download_button(
                            f"Download {target_lang} Markdown",
                            translated_md,
                            f"{filename}_{lang_code}.md",
                            "text/markdown",
                            use_container_width=True,
                            key=f"download_trans_{lang_code}_{filename}"
                        )
                        with st.expander("Preview Translated Document"):
                            st.markdown('<div class="scroll-box markdown-body">', unsafe_allow_html=True)
                            st.markdown(translated_md)
                            st.markdown('</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.info(f"Translation failed: {str(e)}")  # was st.error

        with tabs[5]:
            st.markdown("### Ask About This Document")
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
            user_input = st.chat_input("Ask a question about this document...")
            if user_input:
                with st.chat_message("user"):
                    st.write(user_input)
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            model = genai.GenerativeModel('gemini-1.5-flash')
                            response = model.generate_content([
                                f"{user_input}, Answer based only on this document. Be concise.\n\nDocument:\n{markdown_text}",
                                user_input
                            ])
                            answer = response.text
                        except Exception as e:
                            answer = f"Error: {str(e)}"
                st.write(answer)
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
            if st.button("Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()

        with tabs[6]:
            st.markdown("### Email Report")
            st.info("Select the content you wish to send and enter the recipient's details below.")

            with st.form("email_form"):
                recipient = st.text_input("Recipient's Email", placeholder="name@example.com")
                subject = st.text_input("Subject", value=f"Document Analysis: {st.session_state.filename}")

                st.markdown("Select content to include:")
                include_summary = st.checkbox("Include AI Summary (in email body)", value=True)
                attach_md = st.checkbox("Attach Markdown File (.md)", value=True)
                attach_json = st.checkbox("Attach JSON File (.json)")
                attach_tables = False
                if st.session_state.json_tables:
                    attach_tables = st.checkbox("Attach Extracted Tables (.xlsx)")

                custom_message = st.text_area("Custom Message (Optional)", placeholder="Add a personal note here...")

                submitted = st.form_submit_button("Send Email")

                if submitted:
                    if not recipient:
                        st.info("Please enter a recipient's email address.")  # was st.error
                    else:
                        with st.spinner("Preparing and sending your email..."):
                            attachment_paths = []
                            summary_text = st.session_state.ai_summary if include_summary else ""

                            with tempfile.TemporaryDirectory() as temp_dir:
                                temp_path = Path(temp_dir)

                                if attach_md:
                                    md_path = Path(f"output/{st.session_state.filename}.md")
                                    if md_path.is_file():
                                        attachment_paths.append(str(md_path))

                                if attach_json:
                                    json_path = Path(f"output/{st.session_state.filename}.json")
                                    if json_path.is_file():
                                        attachment_paths.append(str(json_path))

                                if attach_tables and st.session_state.json_tables:
                                    excel_path = temp_path / f"{st.session_state.filename}_tables.xlsx"
                                    excel_data = to_excel(st.session_state.json_tables)
                                    excel_path.write_bytes(excel_data)
                                    attachment_paths.append(str(excel_path))

                                # Send the email
                                success, message = send_report_email(
                                    recipient=recipient,
                                    subject=subject,
                                    custom_message=custom_message,
                                    summary_text=summary_text,
                                    attachment_paths=attachment_paths
                                )

                                if success:
                                    st.success(message)
                                else:
                                    st.info(message)  # was st.error

# -----------------------------
# SESSION STATE INITIALIZATION
# -----------------------------
if "json_data" not in st.session_state:
    st.session_state.json_data = None
if "markdown_text" not in st.session_state:
    st.session_state.markdown_text = None
if "filename" not in st.session_state:
    st.session_state.filename = None
if "ai_summary" not in st.session_state:
    st.session_state.ai_summary = None
if "json_tables" not in st.session_state:
    st.session_state.json_tables = []
if "document_history" not in st.session_state:
    st.session_state.document_history = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------
# SIDEBAR: Document History
# -----------------------------
if "document_history" in st.session_state and st.session_state.document_history:
    with st.sidebar:
        st.header("Recent Documents")
        for name, data in reversed(list(st.session_state.document_history.items())):
            with st.expander(f"{name}"):
                st.caption(f"{data['timestamp']}")
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "Download JSON",
                        json.dumps(data["json_data"], indent=2),
                        f"{name}.json",
                        "application/json",
                        use_container_width=True,
                        key=f"hist_json_{name}"
                    )
                with col2:
                    st.download_button(
                        "Download Markdown",
                        data["markdown_text"],
                        f"{name}.md",
                        "text/markdown",
                        use_container_width=True,
                        key=f"hist_md_{name}"
                    )
                if data["translated"]:
                    st.markdown("Translations:")
                    for lang_code, trans_text in data["translated"].items():
                        st.download_button(
                            f"{lang_code.upper()}",
                            trans_text,
                            f"{name}_{lang_code}.md",
                            "text/markdown",
                            use_container_width=True,
                            key=f"trans_{name}_{lang_code}"
                        )

# -----------------------------
# BUTTONS AND HOVER: RED ONLY (Streamlit default)
# -----------------------------
st.markdown("""
<style>
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
    background: #d33a2c !important;
    color: #fff !important;
    border-radius: 4px !important;
    border: none !important;
    font-weight: 500;
    box-shadow: none !important;
    transition: background 0.2s;
}
.stButton > button:hover, .stDownloadButton > button:hover, .stFormSubmitButton > button:hover {
    background: #a82312 !important;
    color: #fff !important;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown(
    "<p class='footer'>"
    "IMDU Document Parser &copy; TEAM CODEX"
    "</p>",
    unsafe_allow_html=True
)

