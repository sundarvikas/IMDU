# streamlit_app.py
import streamlit as st
from pathlib import Path
import json
from datetime import datetime

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
    page_title="Document Parser",
    page_icon="📄",
    layout="wide"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>
    .main .block-container { padding-top: 1rem; }
    .stButton button {
        border-radius: 8px;
        height: 3rem;
        font-weight: 600;
    }
    .scroll-box {
        max-height: 700px;
        overflow-y: auto;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 16px;
        background-color: #f8f9fa;
        font-family: 'Courier New', monospace;
        line-height: 1.6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .markdown-body {
        line-height: 1.8;
        font-size: 1.1em;
        color: #222;
    }
    .upload-area {
        border: 2px dashed #4CAF50;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        background-color: #f1f8e9;
        margin-bottom: 1.5rem;
    }
    .result-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .footer {
        font-size: 0.9rem;
        color: #777;
        text-align: center;
        margin-top: 3rem;
    }
    [data-testid="stFileUploader"] {
        padding: 0;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------
# HELPER FUNCTION: Extract Summary Stats
# -----------------------------
def get_summary_stats(json_data):
    """
    Extracts content type counts from the parsed JSON structure.
    Handles: document → pages → blocks
    Returns a dict of {block_type: count}
    """
    type_counts = {}
    try:
        pages = json_data.get("document", {}).get("pages", [])
        for page in pages:
            for block in page.get("blocks", []):
                block_type = block.get("type", "unknown").title()  # Capitalize for display
                type_counts[block_type] = type_counts.get(block_type, 0) + 1
    except Exception as e:
        st.warning(f"Error parsing JSON structure: {str(e)}")
        return {"Error": 1}
    return type_counts


# -----------------------------
# HEADER
# -----------------------------
st.markdown("<h1 style='text-align: center;'>📄 Document Parser</h1>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; font-size: 1.1rem; color: #555; margin-bottom: 1.5rem;">
Upload a PDF, image, or Word document to extract structured content.
</div>
""", unsafe_allow_html=True)


# -----------------------------
# FILE UPLOADER & PREVIEW
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload your document",
    type=["pdf", "png", "jpg", "jpeg", "docx"],
    label_visibility="collapsed"
)

if uploaded_file is None:
    st.markdown('<div class="upload-area">', unsafe_allow_html=True)
    st.markdown("📁 Drag & drop a PDF, image, or DOCX file here")
    st.markdown("Supported: PDF, PNG, JPG, DOCX")
    st.markdown('</div>', unsafe_allow_html=True)
else:
    # Setup directories
    uploads_dir = Path("uploads")
    output_dir = Path("output")
    uploads_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    file_path = uploads_dir / uploaded_file.name
    stem = file_path.stem

    # Save uploaded file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"✅ **{uploaded_file.name}** uploaded.")

    # Preview Section
    st.markdown("### 🔍 Document Preview")
    if uploaded_file.type.startswith("image"):
        st.image(file_path, caption="📷 Image Preview", use_column_width=True)
    elif uploaded_file.type == "application/pdf":
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(file_path)
            page = doc.load_page(0)
            pix = page.get_pixmap(dpi=96)
            img_bytes = pix.tobytes("png")
            st.image(img_bytes, caption="📄 PDF Page 1 Preview", use_column_width=True)
        except Exception:
            st.info("Preview not available for this PDF.")
    elif uploaded_file.name.endswith(".docx"):
        st.info("📄 Word document uploaded. Preview not available.")

    # Parse Button
    if st.button("✨ Parse Document", key="parse", type="primary", use_container_width=True):
        with st.spinner("🧠 Analyzing document... This may take a few seconds."):
            try:
                # Run the pipeline
                json_data, markdown_text = process_document(
                    file_path=str(file_path),
                    json_path=str(output_dir / f"{stem}.json"),
                    md_path=str(output_dir / f"{stem}.md")
                )

                # Save to session state
                st.session_state.json_data = json_data
                st.session_state.markdown_text = markdown_text
                st.session_state.filename = stem
                st.session_state.processed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                st.toast("✅ Parsing complete!", icon="🎉")
            except Exception as e:
                st.error(f"❌ Processing failed: {str(e)}")
                st.exception(e)  # Optional: show full error (remove in production)


# -----------------------------
# DISPLAY RESULTS (if processed)
# -----------------------------
if "json_data" in st.session_state:
    json_data = st.session_state.json_data
    markdown_text = st.session_state.markdown_text
    filename = st.session_state.filename

    st.markdown("---")
    st.markdown(f"<div class='result-header'>Parsed Output</div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📑 Formatted Content", "📦 Structured Data", "⬇️ Export"])

    with tab1:
        st.markdown("### 📝 Extracted Document (Markdown)")
        with st.expander("View Full Output", expanded=True):
            st.markdown('<div class="scroll-box markdown-body">', unsafe_allow_html=True)
            st.markdown(markdown_text)
            st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown("### 🔍 JSON Structure")
        with st.expander("Show Full JSON", expanded=False):
            st.markdown('<div class="scroll-box">', unsafe_allow_html=True)
            st.json(json_data)
            st.markdown('</div>', unsafe_allow_html=True)

        # 📊 Summary Stats using helper function
        st.markdown("##### 📊 Content Summary")
        stats = get_summary_stats(json_data)

        # Display metrics in rows of up to 4
        items = list(stats.items())
        for i in range(0, len(items), 4):
            cols = st.columns(4)
            for j in range(4):
                idx = i + j
                if idx < len(items):
                    label, value = items[idx]
                    cols[j].metric(label=label, value=value)

        # Total elements
        total_elements = sum(stats.values())
        st.caption(f"📄 Total content blocks detected: {total_elements}")

    with tab3:
        st.markdown("### 💾 Download Results")
        json_str = json.dumps(json_data, indent=2, ensure_ascii=False)
        markdown_str = markdown_text

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="⬇️ Download JSON",
                data=json_str,
                file_name=f"{filename}.json",
                mime="application/json",
                use_container_width=True
            )
        with col2:
            st.download_button(
                label="⬇️ Download Markdown",
                data=markdown_str,
                file_name=f"{filename}.md",
                mime="text/markdown",
                use_container_width=True
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