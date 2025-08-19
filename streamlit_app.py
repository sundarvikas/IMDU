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
    page_title="INTELLIGENT MULTI-LINGUAL DOCUMENT UNDERSTANDING",
    page_icon="📄",
    layout="wide"
)

# -----------------------------
# CUSTOM CSS (Dark Mode Friendly)
# -----------------------------
st.markdown("""
<style>
    /* General */
    .main .block-container { padding-top: 1rem; }

    /* Button styling */
    .stButton button {
        border-radius: 8px;
        height: 3rem;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton button:hover {
        transform: scale(1.02);
    }

    /* Scrollable boxes */
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

    /* Markdown output */
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

    /* Upload Area - Modern & Theme-Aware */
    .upload-instruction {
        font-size: 1.2rem;
        color: #555;
        margin-bottom: 0.5rem;
    }
    .upload-note {
        font-size: 0.95rem;
        color: #777;
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
    .upload-area-modern:hover {
        background-color: var(--background-color-upload-hover, #e9ecef);
        border-color: var(--primary-color, #1f77b4);
        transform: translateY(-2px);
    }
    .upload-icon {
        font-size: 3rem;
        margin-bottom: 0.5rem;
        color: var(--primary-color, #1f77b4);
    }
    @media (prefers-color-scheme: dark) {
        .upload-area-modern {
            background-color: #2d2d2d;
            color: #ddd;
            border-color: #555;
        }
        .upload-note, .upload-instruction {
            color: #bbb;
        }
        .scroll-box, .stJson, .stDownloadButton, .stMetric {
            color: #ddd !important;
        }
    }

    /* Result Header */
    .result-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--primary-color, #1f77b4);
        margin-bottom: 1rem;
    }

    /* Footer */
    .footer {
        font-size: 0.9rem;
        color: #777;
        text-align: center;
        margin-top: 3rem;
    }
    @media (prefers-color-scheme: dark) {
        .footer {
            color: #aaa;
        }
    }

    /* Hide default uploader filename */
    [data-testid="stFileUploader"] {
        padding: 0;
    }
    .uploadedFileName > span {
        display: none;
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
                block_type = block.get("type", "unknown").title()
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
# Custom upload area with theme support
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
        except Exception as e:
            st.info(f"Preview not available for this PDF. Error: {str(e)}")
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

        # 📊 Summary Stats
        st.markdown("##### 📊 Content Summary")
        stats = get_summary_stats(json_data)

        # Display metrics in rows of 4
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