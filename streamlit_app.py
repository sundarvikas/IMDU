import streamlit as st
from pathlib import Path
import json
from datetime import datetime
import google.generativeai as genai
import pandas as pd
from io import BytesIO
import tempfile
import fitz  # PyMuPDF
from supabase import create_client, Client, ClientOptions # Added for Supabase

# --- Import email function ---
try:
    from email_report import send_report_email
except ImportError:
    def send_report_email(*args, **kwargs):
        st.error("email_report.py not found. Email functionality is disabled.")
        return False, "Email function not found."

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="IMDU Document Parser",
    page_icon="📄",
    initial_sidebar_state="expanded"
)

# -----------------------------
# SUPABASE CONNECTION
# -----------------------------
@st.cache_resource
def init_connection():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    
    # Revert to the simplest and correct connection method
    return create_client(url, key)

supabase = init_connection()

# -----------------------------
# SUPABASE AUTHENTICATION UI
# -----------------------------
def auth_ui():
    """Handles session restoration and the user authentication UI."""
    if 'user' not in st.session_state:
        st.session_state.user = None

    # ✅ If session_state.user is None, try to recover the session on each rerun
    if st.session_state.user is None:
        session = supabase.auth.get_session()
        if session:
            st.session_state.user = session.user

    # --- UI Rendering ---
    if st.session_state.user is None:
        st.title("IMDU Document Parser")
        st.sidebar.subheader("Login / Sign Up")
        
        auth_option = st.sidebar.radio(
            "Choose an action", 
            ("Login", "Sign Up", "Forgot Password"), 
            label_visibility="collapsed"
        )

        if auth_option == "Login":
            with st.sidebar.form("login_form"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                if st.form_submit_button("Log In", type="primary", use_container_width=True):
                    try:
                        session = supabase.auth.sign_in_with_password({"email": email, "password": password})
                        st.session_state.user = session.user
                        st.rerun()
                    except Exception as e:
                        st.sidebar.error("Login failed: Incorrect email or password.")

        elif auth_option == "Sign Up":
            with st.sidebar.form("signup_form"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                if st.form_submit_button("Sign Up", type="primary", use_container_width=True):
                    try:
                        session = supabase.auth.sign_up({"email": email, "password": password})
                        st.sidebar.success("Account created! Please check your email to verify.")
                    except Exception as e:
                        st.sidebar.error(f"Sign up failed: {e}")
        
        # ✅ "Forgot Password" UI
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

    else: # If user is logged in
        user = st.session_state.user
        st.sidebar.subheader("Welcome")
        st.sidebar.markdown(f"**{user.email}**")
        if st.sidebar.button("Logout", use_container_width=True):
            supabase.auth.sign_out()
            # Clear all session state keys for a clean logout
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()
        return user

# -----------------------------
# SUPABASE DATABASE FUNCTIONS
# -----------------------------
def load_user_documents(user_id):
    """
    Loads all documents and their available translations/summaries for a user
    and reconstructs the document_history dictionary.
    """
    history_dict = {}
    try:
        docs_response = supabase.table('documents').select('*').eq('user_id', user_id).order('created_at', desc=True).execute()
        if not docs_response.data:
            return {}

        for doc in docs_response.data:
            doc_id = doc['id']
            doc_key = doc['doc_name']
            
            history_dict[doc_key] = {
                "db_id": doc_id,
                "json_data": doc['original_json'],
                "markdown_text": doc['original_md'],
                "ai_summary": doc['ai_summary'], # ✅ Load the summary
                "timestamp": doc['created_at'],
                "translations": {}
            }
            
            translations_response = supabase.table('translations').select('*').eq('document_id', doc_id).execute()
            if translations_response.data:
                for trans in translations_response.data:
                    lang_code = trans['language_code']
                    history_dict[doc_key]["translations"][lang_code] = trans['translated_md']

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
            'user_id': user_id,
            'doc_name': doc_name,
            'original_json': json_data,
            'original_md': markdown_text,
            'ai_summary': ai_summary  # ✅ Save the summary at creation
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
    # Reset the chat history when loading a new document
    st.session_state.chat_history = []

def save_translation(document_id, lang_code, translated_text):
    """Saves a new translation for a specific document."""
    try:
        # Use upsert to either insert a new translation or update an existing one
        supabase.table('translations').upsert({
            'document_id': document_id,
            'language_code': lang_code,
            'translated_md': translated_text
        }, on_conflict='document_id, language_code').execute()
    except Exception as e:
        st.error(f"Error saving translation: {e}")

# -----------------------------
# SESSION STATE INITIALIZATION
# -----------------------------
# This is now handled after the user logs in.
if "json_data" not in st.session_state:
    st.session_state.json_data = None
if "markdown_text" not in st.session_state:
    st.session_state.markdown_text = None
if "filename" not in st.session_state:
    st.session_state.filename = None
if "current_doc_key" not in st.session_state:
    st.session_state.current_doc_key = None
if "ai_summary" not in st.session_state:
    st.session_state.ai_summary = None
if "json_tables" not in st.session_state:
    st.session_state.json_tables = []
if "document_history" not in st.session_state:
    st.session_state.document_history = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------
# GEMINI SETUP
# -----------------------------
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error("GEMINI_API_KEY not found in secrets.toml. Please add it to proceed.")
    st.stop()

# -----------------------------
# PIPELINE AND HELPERS
# -----------------------------
try:
    from imdu_pipeline import process_document
except Exception as e:
    st.error(f"Error loading pipeline: {e}")
    st.stop()

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

# =================================================================================================
# --- MAIN APP ---
# =================================================================================================
# =================================================================================================
# --- MAIN APP ---
# =================================================================================================
user = auth_ui()

if not user:
    st.info("Please log in or sign up to use the app.")
    st.stop()

# --- If user is logged in, the rest of the app runs ---

# Load user's history from Supabase at the start of their session
# --- At the start of your logged-in app section ---
if 'history_loaded' not in st.session_state:
    st.session_state.document_history = load_user_documents(user.id)
    st.session_state.history_loaded = True

# (The rest of your main app UI code follows here)

# -----------------------------
# MAIN APP UI
# -----------------------------
st.markdown("<h1 style='text-align: center;'>IMDU Document Parser</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #aaa;'>Upload a document to extract structured content.</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload PDF, Image, or DOCX", type=["pdf", "png", "jpg", "jpeg", "docx"], label_visibility="collapsed")

if uploaded_file:
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(exist_ok=True)
    file_path = uploads_dir / uploaded_file.name
    stem = file_path.stem
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"`{uploaded_file.name}` uploaded successfully.")

    with st.expander("Show Document Preview"):
        if uploaded_file.type.startswith("image"):
            st.image(file_path, caption="Image Preview", use_container_width=True)
        elif uploaded_file.type == "application/pdf":
            try:
                doc = fitz.open(file_path)
                page = doc.load_page(0)
                pix = page.get_pixmap(dpi=96)
                img_bytes = pix.tobytes("png")
                st.image(img_bytes, caption="PDF Page 1 Preview", use_column_width=True)
            except Exception as e:
                st.warning(f"Could not generate PDF preview: {e}")
        else:
            st.info("Preview is not available for this file type.")

    if st.button("Parse Document", type="primary", use_container_width=True):
        with st.spinner("Processing document and generating summary..."):
            try:
                # Step 1: Process the document to get JSON and Markdown
                json_data, markdown_text = process_document(file_path=str(file_path))
                
                # Step 2: Generate the AI summary immediately
                summary = "" # Default to empty string
                try:
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    response = model.generate_content(f"Provide a concise summary of the document:\n\n{markdown_text}")
                    summary = response.text
                    st.toast("AI summary generated!", icon="🧠")
                except Exception as e:
                    st.warning(f"Could not generate AI summary: {e}")

                # Step 3: Save the complete record to the database in one call
                doc_key = f"{stem}{file_path.suffix}"
                new_doc_record = save_new_document(user.id, doc_key, json_data, markdown_text, summary)

                # Step 4: If save is successful, update the session state
                if new_doc_record:
                    st.session_state.update(
                        json_data=json_data,
                        markdown_text=markdown_text,
                        filename=stem,
                        json_tables=find_tables_in_json(json_data),
                        ai_summary=summary,
                        chat_history=[],
                        current_doc_key=doc_key
                    )
                    
                    # Update the history dictionary with the complete, correct data
                    st.session_state.document_history[doc_key] = {
                        "db_id": new_doc_record['id'],
                        "json_data": new_doc_record['original_json'],
                        "markdown_text": new_doc_record['original_md'],
                        "ai_summary": new_doc_record['ai_summary'],
                        "timestamp": new_doc_record['created_at'],
                        "translations": {}
                    }
                    st.toast("Document processed and saved!", icon="✅")

            except Exception as e:
                st.error(f"A critical error occurred: {str(e)}")
                st.stop()
                
        st.rerun()

        with st.spinner("Generating AI summary..."):
            try:
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(f"Provide a concise summary of the document:\n\n{markdown_text}")
                st.session_state.ai_summary = response.text
                st.toast("Parsing and summary complete!", icon="✅")
            except Exception as e:
                st.warning(f"Automatic summary generation failed: {e}")
                st.toast("Parsing complete, but summary failed.", icon="⚠️")
        st.rerun()

# -----------------------------
# DISPLAY RESULTS TABS
# -----------------------------
if st.session_state.get("json_data"):
    st.divider()
    tabs = st.tabs(["Formatted Content", "AI Summary", "Extracted Tables", "Structured Data", "Export & Translate", "Chat with Document", "Email Report"])

    # ... (The code for all the tabs remains the same as your original) ...
    # ... I am including it here for completeness ...

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
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    response = model.generate_content(f"Provide a concise, professional summary of the document:\n\n{st.session_state.markdown_text}")
                    st.session_state.ai_summary = response.text
                    st.rerun()
                except Exception as e:
                    st.error(f"Summary generation failed: {e}")

    with tabs[2]:
        st.subheader("Extracted Tables")
        if st.session_state.json_tables:
            for i, df in enumerate(st.session_state.json_tables):
                st.markdown(f"**Table {i+1}**")
                st.dataframe(df, use_container_width=True)
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
        st.subheader("Download Full Results")
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
                    # --- Inside the "Translate" button logic ---
                    doc_key = st.session_state.current_doc_key
                    if doc_key and doc_key in st.session_state.document_history:
                        # Get the document's database ID from session state
                        document_db_id = st.session_state.document_history[doc_key].get('db_id')
                        
                        if document_db_id:
                            # Save the translation to the new table
                            save_translation(document_db_id, lang_code, translated_text)
                            # Update the local session state as well
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
            subject = st.text_input("Subject", value=f"Analysis of: {st.session_state.filename}", placeholder="Email Subject")
            
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
                        # --- ✅ THIS IS THE MISSING LOGIC ---
                        attachment_paths = []
                        summary_text = st.session_state.ai_summary if include_summary else ""

                        # Use a temporary directory to safely create attachment files
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

                            # Call the email function from email_report.py
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
                                st.error(message)
# -----------------------------
# SIDEBAR: Document History
# -----------------------------
if st.session_state.document_history:
    st.sidebar.divider()
    st.sidebar.subheader("Document History")
    
    # Sort history to show the most recent documents first
    sorted_history = sorted(st.session_state.document_history.items(), key=lambda item: item[1]['timestamp'], reverse=True)
    
    for doc_name, data in sorted_history:
        # ✅ Create a button for each document. Clicking it loads the state.
        st.sidebar.button(
            doc_name,
            key=f"hist_btn_{doc_name}",
            on_click=load_document_to_state,
            args=(doc_name, data),
            use_container_width=True
        )
# -----------------------------
# FOOTER
# -----------------------------
st.divider()
st.caption("<p style='text-align:center;'>© 2025 IMDU Document Parser by TEAM CODEX</p>", unsafe_allow_html=True)