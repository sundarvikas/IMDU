import streamlit as st
from pathlib import Path
import json
from datetime import datetime
import google.generativeai as genai
import pandas as pd
from io import BytesIO
import tempfile
import fitz  # PyMuPDF
from supabase import create_client, Client
import subprocess
import os

# --- Import Custom Modules ---
try:
    from email_report import send_report_email
except ImportError:
    def send_report_email(*args, **kwargs):
        st.error("email_report.py not found. Email functionality is disabled.")
        return False, "Email function not found."
try:
    from imdu_pipeline import process_document
except Exception as e:
    st.error(f"Error loading pipeline: {e}")
    st.stop()

# --- Message Handler (shows messages after a rerun) ---
if "message" in st.session_state and st.session_state.message:
    st.toast(st.session_state.message, icon="✅")
    st.session_state.message = None

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="IMDU Document Parser",
    page_icon="📄",
    initial_sidebar_state="expanded"  # This line makes the sidebar open on load
)

# -----------------------------
# SUPABASE CONNECTION
# -----------------------------
@st.cache_resource
def init_connection():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_connection()

# -----------------------------
# APP FUNCTIONS
# -----------------------------
def auth_ui():
    """Handles session restoration and the user authentication UI."""
    if 'user' not in st.session_state:
        st.session_state.user = None
    if st.session_state.user is None:
        session = supabase.auth.get_session()
        if session:
            st.session_state.user = session.user

    if st.session_state.user is None:
        st.title("IMDU Document Parser")
        st.sidebar.subheader("Login / Sign Up")
        auth_option = st.sidebar.radio("Choose an action", ("Login", "Sign Up", "Forgot Password"), label_visibility="collapsed")

        if auth_option == "Login":
            with st.sidebar.form("login_form"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                if st.form_submit_button("Log In", type="primary", use_container_width=True):
                    try:
                        session = supabase.auth.sign_in_with_password({"email": email, "password": password})
                        st.session_state.user = session.user
                        st.rerun()
                    except Exception:
                        st.sidebar.error("Login failed: Incorrect email or password.")
        elif auth_option == "Sign Up":
            with st.sidebar.form("signup_form"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                if st.form_submit_button("Sign Up", type="primary", use_container_width=True):
                    try:
                        supabase.auth.sign_up({"email": email, "password": password})
                        st.sidebar.success("Account created! Please check your email to verify.")
                    except Exception as e:
                        st.sidebar.error(f"Sign up failed: {e}")
        elif auth_option == "Forgot Password":
            with st.sidebar.form("reset_password_form"):
                email = st.text_input("Enter your email address")
                if st.form_submit_button("Send Reset Link", use_container_width=True):
                    try:
                        supabase.auth.reset_password_for_email(email=email)
                        st.sidebar.success("Password reset link sent. Please check your email.")
                    except Exception as e:
                        st.sidebar.error(f"Failed to send reset link: {e}")
        return None
    else:
        user = st.session_state.user
        st.sidebar.subheader("Welcome")
        st.sidebar.markdown(f"**{user.email}**")
        if st.sidebar.button("Logout", use_container_width=True):
            supabase.auth.sign_out()
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()
        return user

def load_user_documents(user_id):
    """Loads all documents for a user in an optimized way (2 queries)."""
    history_dict = {}
    try:
        docs_response = supabase.table('documents').select('*').eq('user_id', user_id).order('created_at', desc=True).execute()
        if not docs_response.data: return {}
        doc_ids = [doc['id'] for doc in docs_response.data]
        for doc in docs_response.data:
            history_dict[doc['doc_name']] = {
                "db_id": doc['id'], "json_data": doc['original_json'], "markdown_text": doc['original_md'],
                "ai_summary": doc['ai_summary'], "timestamp": doc['created_at'], "translations": {}
            }
        translations_response = supabase.table('translations').select('*').in_('document_id', doc_ids).execute()
        if translations_response.data:
            doc_id_to_name_map = {data['db_id']: doc_name for doc_name, data in history_dict.items()}
            for trans in translations_response.data:
                doc_id = trans['document_id']
                if doc_id in doc_id_to_name_map:
                    doc_name = doc_id_to_name_map[doc_id]
                    lang_code = trans['language_code']
                    history_dict[doc_name]["translations"][lang_code] = trans['translated_md']
    except Exception as e:
        st.error(f"Error loading document history: {e}")
    return history_dict

def update_document_summary(doc_id, summary_text):
    """Updates an existing document record with the AI-generated summary."""
    try:
        supabase.table('documents').update({'ai_summary': summary_text}).eq('id', doc_id).execute()
    except Exception as e:
        st.error(f"Error updating summary: {e}")

def save_new_document(user_id, doc_name, json_data, markdown_text, ai_summary):
    """Saves a new document, including its summary, to the database."""
    try:
        response = supabase.table('documents').insert({
            'user_id': user_id, 'doc_name': doc_name, 'original_json': json.loads(json.dumps(json_data)),
            'original_md': markdown_text, 'ai_summary': ai_summary
        }).execute()
        return response.data[0]
    except Exception as e:
        st.error(f"Error saving document: {e}")
        return None

def load_document_to_state(doc_key, doc_data):
    """Callback function to load a selected historical document into the main app state."""
    st.session_state.json_data = doc_data.get('json_data')
    st.session_state.markdown_text = doc_data.get('markdown_text')
    st.session_state.ai_summary = doc_data.get('ai_summary')
    st.session_state.filename = Path(doc_key).stem
    st.session_state.current_doc_key = doc_key
    st.session_state.json_tables = find_tables_in_json(st.session_state.json_data)
    st.session_state.chat_history = []

def save_translation(document_id, lang_code, translated_text):
    """Saves a new translation for a specific document."""
    try:
        supabase.table('translations').upsert({
            'document_id': document_id, 'language_code': lang_code, 'translated_md': translated_text
        }, on_conflict='document_id, language_code').execute()
    except Exception as e:
        st.error(f"Error saving translation: {e}")

def delete_document(doc_id):
    """Deletes a document and its associated translations from the database."""
    try:
        supabase.table('documents').delete().eq('id', doc_id).execute()
        return True, "Document deleted successfully."
    except Exception as e:
        return False, f"Error deleting document: {e}"

def clean_dataframe_columns(df):
    """Finds and renames duplicate column names in a DataFrame by appending a suffix."""
    new_columns, seen_columns = [], {}
    for col in df.columns:
        if col in seen_columns:
            seen_columns[col] += 1
            new_columns.append(f"{col}_{seen_columns[col]}")
        else:
            seen_columns[col] = 0
            new_columns.append(col)
    df.columns = new_columns
    return df

@st.cache_data
def to_excel(df_list):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for i, df in enumerate(df_list):
            df.to_excel(writer, index=False, sheet_name=f'Table_{i+1}')
    return output.getvalue()

def find_tables_in_json(json_data):
    tables = []
    if not isinstance(json_data, dict): return []
    try:
        for page in json_data.get("document", {}).get("pages", []):
            for block in page.get("blocks", []):
                if block.get("type") == "table":
                    headers = block.get("content", {}).get("headers", [])
                    rows = block.get("content", {}).get("rows", [])
                    if headers and rows:
                        tables.append(pd.DataFrame(rows, columns=headers))
    except Exception as e:
        st.warning(f"Error extracting tables: {e}")
    return tables

def get_summary_stats(json_data: dict) -> dict:
    type_counts = {}
    if not isinstance(json_data, dict): return {}
    try:
        pages = json_data.get("document", {}).get("pages", [])
        for page in pages:
            for block in page.get("blocks", []):
                block_type = block.get("type", "unknown").title()
                type_counts[block_type] = type_counts.get(block_type, 0) + 1
    except Exception as e:
        st.warning(f"Error parsing JSON structure for stats: {str(e)}")
    return type_counts

def convert_to_pdf(input_path: Path, output_dir: Path):
    """Converts a file to PDF using a direct path to LibreOffice on Windows."""
    if os.name == 'nt':
        libreoffice_path = "C:/Program Files/LibreOffice/program/soffice.exe"
    else:
        libreoffice_path = "libreoffice"
    try:
        subprocess.run(
            [libreoffice_path, "--headless", "--convert-to", "pdf", "--outdir", str(output_dir), str(input_path)],
            check=True, timeout=30.0
        )
        pdf_path = output_dir / f"{input_path.stem}.pdf"
        return pdf_path if pdf_path.exists() else None
    except FileNotFoundError:
        st.error(f"Error: Could not find LibreOffice. Please ensure it's installed at the path: '{libreoffice_path}'")
        return None
    except Exception as e:
        st.error(f"Error during DOCX to PDF conversion: {e}")
        return None

# -----------------------------
# SESSION STATE INITIALIZATION
# -----------------------------
keys_to_init = ["user", "json_data", "markdown_text", "filename", "current_doc_key", "ai_summary"]
for key in keys_to_init:
    if key not in st.session_state:
        st.session_state[key] = None
if "json_tables" not in st.session_state: st.session_state.json_tables = []
if "document_history" not in st.session_state: st.session_state.document_history = {}
if "chat_history" not in st.session_state: st.session_state.chat_history = []

# -----------------------------
# GEMINI SETUP
# -----------------------------
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
except KeyError:
    st.error("GEMINI_API_KEY not found in secrets.toml. Please add it to proceed.")
    st.stop()

# =================================================================================================
# --- MAIN APP ---
# =================================================================================================
user = auth_ui()

if not user:
    st.info("Please log in or sign up to use the app.")
    st.stop()

if 'history_loaded' not in st.session_state:
    st.session_state.document_history = load_user_documents(user.id)
    st.session_state.history_loaded = True

st.markdown("<h1 style='text-align: center;'>IMDU Document Parser</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #aaa;'>Upload a document to extract structured content.</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload PDF, Image, or DOCX", type=["pdf", "png", "jpg", "jpeg", "docx"], label_visibility="collapsed")

if uploaded_file:
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(exist_ok=True)
    current_file_id = f"{uploaded_file.name}-{uploaded_file.size}"
    if st.session_state.get('processed_file_id') != current_file_id:
        st.session_state.file_path_for_processing = None
        st.session_state.processed_file_id = current_file_id

    original_file_path = uploads_dir / uploaded_file.name
    with open(original_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    if not st.session_state.get('file_path_for_processing'):
        st.success(f"`{uploaded_file.name}` uploaded successfully.")
        if original_file_path.suffix.lower() == ".docx":
            with st.spinner("Converting DOCX to PDF for analysis..."):
                pdf_path = convert_to_pdf(original_file_path, uploads_dir)
                if pdf_path:
                    st.session_state.file_path_for_processing = pdf_path
                    st.info("DOCX file converted successfully.")
                else:
                    st.error("Failed to convert DOCX file.")
                    st.stop()
        else:
            st.session_state.file_path_for_processing = original_file_path

    file_path_for_processing = st.session_state.get('file_path_for_processing')

    if file_path_for_processing:
        stem = file_path_for_processing.stem
        with st.expander("Show Document Preview"):
            try:
                doc = fitz.open(file_path_for_processing)
                page = doc.load_page(0)
                pix = page.get_pixmap(dpi=96)
                st.image(pix.tobytes("png"), caption="Page 1 Preview", use_column_width=True)
            except Exception as e:
                st.warning(f"Could not generate document preview: {e}")

        if st.button("Parse Document", type="primary", use_container_width=True):
            with st.spinner("Processing document and generating summary..."):
                try:
                    json_data, markdown_text = process_document(file_path=str(file_path_for_processing))
                    summary = ""
                    try:
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        response = model.generate_content(f"Provide a concise summary of the document:\n\n{markdown_text}")
                        summary = response.text
                        st.toast("AI summary generated!", icon="🧠")
                    except Exception as e:
                        st.warning(f"Could not generate AI summary: {e}")

                    doc_key = f"{original_file_path.stem}{original_file_path.suffix}"
                    new_doc_record = save_new_document(user.id, doc_key, json_data, markdown_text, summary)

                    if new_doc_record:
                        st.session_state.update(
                            json_data=json_data, markdown_text=markdown_text, filename=stem,
                            json_tables=find_tables_in_json(json_data), ai_summary=summary,
                            chat_history=[], current_doc_key=doc_key
                        )
                        st.session_state.document_history[doc_key] = {
                            "db_id": new_doc_record['id'], "json_data": new_doc_record['original_json'],
                            "markdown_text": new_doc_record['original_md'], "ai_summary": new_doc_record['ai_summary'],
                            "timestamp": new_doc_record['created_at'], "translations": {}
                        }
                        st.toast("Document processed and saved!", icon="✅")
                except Exception as e:
                    st.error(f"A critical error occurred: {str(e)}")
                    st.stop()
            st.rerun()

if st.session_state.get("json_data"):
    st.divider()
    tabs = st.tabs(["Formatted Content", "AI Summary", "Extracted Tables", "Structured Data", "Export & Translate", "Chat with Document", "Email Report"])

    with tabs[0]:
        st.markdown(st.session_state.markdown_text, unsafe_allow_html=True)

    with tabs[1]:
        st.subheader("AI-Generated Summary")
        if st.session_state.ai_summary:
            st.markdown(st.session_state.ai_summary)
        else:
            st.info("No AI summary available for this document.")
        if st.button("Regenerate Summary", use_container_width=True, type="primary"):
            with st.spinner("Regenerating summary..."):
                try:
                    doc_key = st.session_state.current_doc_key
                    doc_db_id = st.session_state.document_history[doc_key].get('db_id')
                    if not doc_db_id:
                        st.error("Could not find the document ID to save the summary.")
                    else:
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        response = model.generate_content(f"Provide a concise, professional summary...\n\n{st.session_state.markdown_text}")
                        new_summary = response.text
                        update_document_summary(doc_db_id, new_summary)
                        st.session_state.ai_summary = new_summary
                        st.session_state.document_history[doc_key]['ai_summary'] = new_summary
                        st.toast("Summary updated and saved!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Summary generation failed: {e}")

    with tabs[2]:
        st.subheader("Extracted Tables")
        if st.session_state.json_tables:
            for i, df in enumerate(st.session_state.json_tables):
                st.markdown(f"**Table {i+1}**")
                try:
                    df_display = df.copy()
                    df_display = clean_dataframe_columns(df_display)
                    df_display = df_display.astype(str)
                    st.dataframe(df_display, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not display Table {i+1} even after cleaning. Error: {e}")
                    st.write("Original DataFrame with potential issues:", df)

                c1, c2 = st.columns(2)
                c1.download_button(f"Download Table {i+1} as CSV", df.to_csv(index=False).encode('utf-8'), f"{st.session_state.filename}_table_{i+1}.csv", "text/csv", use_container_width=True, key=f"csv_{i}")
                c2.download_button(f"Download Table {i+1} as Excel", to_excel([df]), f"{st.session_state.filename}_table_{i+1}.xlsx", use_container_width=True, key=f"excel_{i}")
        else:
            st.info("No tables were found in the document.")

    with tabs[3]:
        st.subheader("Document Structure Overview")
        stats = get_summary_stats(st.session_state.json_data)
        items = list(stats.items())
        for i in range(0, len(items), 4):
            cols = st.columns(4)
            for j, col in enumerate(cols):
                if (i + j) < len(items):
                    label, value = items[i+j]
                    col.metric(label=label, value=value)
        with st.expander("Show Full JSON Structure"):
            st.json(st.session_state.json_data)

    with tabs[4]:
        st.subheader("Download & Translate")
        c1, c2 = st.columns(2)
        c1.download_button("Download Markdown (.md)", st.session_state.markdown_text, f"{st.session_state.filename}.md", "text/markdown", use_container_width=True, type="primary")
        c2.download_button("Download JSON (.json)", json.dumps(st.session_state.json_data, indent=2), f"{st.session_state.filename}.json", "application/json", use_container_width=True, type="primary")
        st.divider()
        st.subheader("Translate Document")
        lang_map = {"Hindi": "hi", "Tamil": "ta", "Telugu": "te", "French": "fr", "Spanish": "es", "German": "de"}
        target_lang_name = st.selectbox("Select language", lang_map.keys())
        if st.button(f"Translate to {target_lang_name}", use_container_width=True):
            with st.spinner(f"Translating..."):
                try:
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    lang_code = lang_map[target_lang_name]
                    prompt = f"Translate the following text into {target_lang_name}. Preserve markdown.\n\n{st.session_state.markdown_text}"
                    response = model.generate_content(prompt)
                    translated_text = response.text
                    doc_key = st.session_state.current_doc_key
                    if doc_key and doc_key in st.session_state.document_history:
                        document_db_id = st.session_state.document_history[doc_key].get('db_id')
                        if document_db_id:
                            save_translation(document_db_id, lang_code, translated_text)
                            st.session_state.document_history[doc_key]["translations"][lang_code] = translated_text
                    st.success("Translation complete!")
                    st.markdown(translated_text)
                    st.download_button("Download Translated Version", translated_text, f"{st.session_state.filename}_{lang_code}.md", use_container_width=True)
                except Exception as e:
                    st.error(f"Translation failed: {e}")

    with tabs[5]:
        st.subheader("Chat with Document")
        for msg in st.session_state.chat_history:
            st.chat_message(msg["role"]).write(msg["content"])
        if prompt := st.chat_input("Ask a question about the document..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        response = model.generate_content(f"Based on this document:\n{st.session_state.markdown_text}\n\nAnswer this question: {prompt}")
                        answer = response.text
                        st.write(answer)
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"Could not get answer: {e}")

    with tabs[6]:
        st.markdown("### Email Report")
        st.info("Compose your email and select attachments to send.")
        with st.form("email_form"):
            recipient = st.text_input("Recipient's Email", placeholder="name@example.com")
            subject = st.text_input("Subject", value=f"Document Analysis: {st.session_state.filename}")
            st.markdown("**Select content to include:**")
            include_summary = st.checkbox("Include AI Summary (in email body)", value=False)
            attach_md = st.checkbox("Attach Markdown File (.md)", value=False)
            attach_json = st.checkbox("Attach JSON File (.json)", value=False)
            attach_tables = st.checkbox("Attach Extracted Tables (.xlsx)") if st.session_state.json_tables else False
            custom_message = st.text_area("Custom Message (Optional)", placeholder="Add a personal note here...")
            submitted = st.form_submit_button("✉️ Send Email", use_container_width=True)
            if submitted:
                if not recipient:
                    st.warning("Please enter a recipient's email address.")
                else:
                    with st.spinner("Preparing and sending your email..."):
                        attachment_paths, summary_text = [], ""
                        if include_summary: summary_text = st.session_state.ai_summary
                        with tempfile.TemporaryDirectory() as temp_dir:
                            temp_path = Path(temp_dir)
                            if attach_md:
                                md_path = temp_path / f"{st.session_state.filename}.md"
                                md_path.write_text(st.session_state.markdown_text, encoding='utf-8')
                                attachment_paths.append(str(md_path))
                            if attach_json:
                                json_path = temp_path / f"{st.session_state.filename}.json"
                                json_path.write_text(json.dumps(st.session_state.json_data, indent=2), encoding='utf-8')
                                attachment_paths.append(str(json_path))
                            if attach_tables and st.session_state.json_tables:
                                excel_path = temp_path / f"{st.session_state.filename}_tables.xlsx"
                                excel_data = to_excel(st.session_state.json_tables)
                                excel_path.write_bytes(excel_data)
                                attachment_paths.append(str(excel_path))

                            success, message = send_report_email(
                                recipient=recipient, subject=subject, custom_message=custom_message,
                                summary_text=summary_text, attachment_paths=attachment_paths
                            )
                            if success: st.success(message)
                            else: st.error(message)

# -----------------------------
# SIDEBAR: Document History & DELETE LOGIC
# -----------------------------
if st.session_state.document_history:
    st.sidebar.divider()
    st.sidebar.subheader("Document History")
    
    # Use the in-sidebar delete confirmation logic
    sorted_history = sorted(st.session_state.document_history.items(), key=lambda item: item[1]['timestamp'], reverse=True)
    
    for doc_name, data in sorted_history:
        doc_to_delete = st.session_state.get('doc_to_delete')
        is_pending_delete = (doc_to_delete is not None) and (doc_to_delete['name'] == doc_name)

        with st.sidebar.expander(doc_name, expanded=is_pending_delete):
            if is_pending_delete:
                st.warning("Are you sure?")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Confirm", key=f"confirm_del_{doc_name}", use_container_width=True, type="primary"):
                        doc_db_id = doc_to_delete['data'].get('db_id')
                        success, message = delete_document(doc_db_id)
                        if success:
                            if st.session_state.get('current_doc_key') == doc_name:
                                keys_to_clear = ['json_data', 'markdown_text', 'ai_summary', 'filename', 'json_tables', 'current_doc_key', 'chat_history']
                                for key in keys_to_clear: st.session_state[key] = None if key not in ['json_tables', 'chat_history'] else []
                            del st.session_state.document_history[doc_name]
                            st.session_state.doc_to_delete = None
                            st.session_state.message = message
                            st.rerun()
                        else:
                            st.error(message)
                with col2:
                    if st.button("Cancel", key=f"cancel_del_{doc_name}", use_container_width=True):
                        st.session_state.doc_to_delete = None
                        st.rerun()
            else:
                st.button("Load Session", key=f"load_btn_{doc_name}", on_click=load_document_to_state, args=(doc_name, data), use_container_width=True)
                if st.button("Delete", key=f"del_btn_{doc_name}", use_container_width=True):
                    st.session_state.doc_to_delete = {'name': doc_name, 'data': data}
                    st.rerun()

# -----------------------------
# FOOTER
# -----------------------------
st.divider()
st.caption("<p style='text-align:center;'>© 2025 IMDU Document Parser by TEAM CODEX</p>", unsafe_allow_html=True)