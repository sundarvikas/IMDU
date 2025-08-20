import streamlit as st
from pathlib import Path
import json
from datetime import datetime
import google.generativeai as genai
import pyrebase
import pandas as pd
from io import BytesIO
import tempfile

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
    layout="wide"
)

# -----------------------------
# SESSION STATE INITIALIZATION
# -----------------------------
if 'user' not in st.session_state:
    st.session_state.user = None
if 'auth_view' not in st.session_state:
    st.session_state.auth_view = 'login'
if 'username' not in st.session_state:
    st.session_state.username = None
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
# FIREBASE INITIALIZATION (Cache it)
# -----------------------------
@st.cache_resource
def get_firebase():
    try:
        firebase_config = st.secrets["firebase_credentials"]
        return pyrebase.initialize_app(firebase_config)
    except Exception as e:
        st.error(f"Firebase config error: {e}")
        return None

firebase = get_firebase()
if not firebase:
    st.stop()

auth = firebase.auth()

# -----------------------------
# AUTO-LOGIN FUNCTION
# -----------------------------
def auto_login():
    if st.session_state.user and isinstance(st.session_state.user, dict):
        refresh_token = st.session_state.user.get("refreshToken")
        if not refresh_token:
            return False
        try:
            refreshed = auth.refresh(refresh_token)
            st.session_state.user.update({
                "idToken": refreshed["idToken"],
                "refreshToken": refreshed["refreshToken"]
            })
            user_info = auth.get_account_info(refreshed["idToken"])
            email = st.session_state.user["email"]
            display_name = user_info['users'][0].get('displayName')
            st.session_state.username = display_name or email.split('@')[0]
            return True
        except Exception:
            st.session_state.user = None
            st.session_state.username = None
            return False
    return False

# -----------------------------
# USER AUTHENTICATION
# -----------------------------
def app_auth():
    # Try auto-login first
    if st.session_state.user and auto_login():
        st.sidebar.subheader(f"Welcome, {st.session_state.username}!")
        if st.sidebar.button("Logout", use_container_width=True):
            st.session_state.update(user=None, username=None, auth_view='login')
            st.rerun()
        return True

    # --- AUTH LOGIC ---
    def login_callback():
        try:
            email, password = st.session_state['login_email'], st.session_state['login_password']
            user = auth.sign_in_with_email_and_password(email, password)
            st.session_state.user = {
                "email": user["email"],
                "idToken": user["idToken"],
                "refreshToken": user["refreshToken"]
            }
            info = auth.get_account_info(user["idToken"])["users"][0]
            st.session_state.username = info.get("displayName", email.split('@')[0])
        except Exception:
            st.error("Login Failed: Please check your credentials.")

    def signup_callback():
        try:
            email, password, username = st.session_state['signup_email'], st.session_state['signup_password'], st.session_state['signup_username']
            user = auth.create_user_with_email_and_password(email, password)
            auth.update_profile(user['idToken'], display_name=username)
            st.session_state.user = {
                "email": user["email"],
                "idToken": user["idToken"],
                "refreshToken": user["refreshToken"]
            }
            st.session_state.username = username
            st.success("Account created successfully!")
        except Exception as e:
            st.error(f"Sign-up Failed: {e}")

    def reset_password_callback():
        try:
            auth.send_password_reset_email(st.session_state['reset_email'])
            st.success("Password reset link sent! Check your email.")
            st.session_state.auth_view = 'login'
        except Exception as e:
            st.error(f"Failed to send reset link: {e}")

    def logout_user():
        st.session_state.update(user=None, username=None, auth_view='login')

    # --- AUTH UI ---
    if not st.session_state.user:
        st.title("IMDU Document Parser")
        if st.session_state.auth_view == 'login':
            st.subheader("Log In")
            with st.form("login_form"):
                st.text_input("Email", key="login_email")
                st.text_input("Password", type="password", key="login_password")
                st.form_submit_button("Log In", on_click=login_callback, type="primary")
            c1, c2 = st.columns(2)
            if c1.button("Sign Up", use_container_width=True):
                st.session_state.auth_view = 'signup'
                st.rerun()
            if c2.button("Forgot Password?", use_container_width=True):
                st.session_state.auth_view = 'forgot_password'
                st.rerun()

        elif st.session_state.auth_view == 'signup':
            st.subheader("Create an Account")
            with st.form("signup_form"):
                st.text_input("User Name", key="signup_username")
                st.text_input("Email", key="signup_email")
                pw1 = st.text_input("Password", type="password", key="signup_password")
                pw2 = st.text_input("Confirm Password", type="password")
                if st.form_submit_button("Sign Up", type="primary"):
                    if not all([st.session_state['signup_username'], st.session_state['signup_email'], pw1, pw2]):
                        st.error("Please fill in all fields.")
                    elif pw1 != pw2:
                        st.error("Passwords do not match.")
                    else:
                        signup_callback()
            if st.button("Already have an account? Log In", use_container_width=True):
                st.session_state.auth_view = 'login'
                st.rerun()

        elif st.session_state.auth_view == 'forgot_password':
            st.subheader("Reset Password")
            with st.form("reset_form"):
                st.text_input("Enter your email address", key="reset_email")
                st.form_submit_button("Send Reset Link", on_click=reset_password_callback, type="primary")
            if st.button("Back to Log In", use_container_width=True):
                st.session_state.auth_view = 'login'
                st.rerun()

        return False
    else:
        st.sidebar.subheader(f"Welcome, {st.session_state.username}!")
        if st.sidebar.button("Logout", use_container_width=True):
            logout_user()
            st.rerun()
        return True

# --- Main app starts here ---
if not app_auth():
    st.stop()

# -----------------------------
# GEMINI SETUP
# -----------------------------
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error("GEMINI_API_KEY not found in secrets.toml. Please add it to proceed.")
    st.stop()

# Load pipeline
try:
    from imdu_pipeline import process_document
except Exception as e:
    st.error(f"Error loading pipeline: {e}")
    st.stop()

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
@st.cache_data
def to_excel(df_list):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for i, df in enumerate(df_list):
            df.to_excel(writer, index=False, sheet_name=f'Table_{i+1}')
    return output.getvalue()

def find_tables_in_json(json_data):
    tables = []
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
    try:
        pages = json_data.get("document", {}).get("pages", [])
        for page in pages:
            for block in page.get("blocks", []):
                block_type = block.get("type", "unknown").title()
                type_counts[block_type] = type_counts.get(block_type, 0) + 1
    except Exception as e:
        st.warning(f"Error parsing JSON structure for stats: {str(e)}")
    return type_counts

# -----------------------------
# MAIN APP UI
# -----------------------------
st.markdown("<h1 style='text-align: center;'>IMDU Document Parser</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #aaa;'>Upload a document to extract structured content.</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload PDF, Image, or DOCX", type=["pdf", "png", "jpg", "jpeg", "docx"], label_visibility="collapsed")

if uploaded_file:
    uploads_dir, output_dir = Path("uploads"), Path("output")
    uploads_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
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
                import fitz
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
        with st.spinner("Analyzing document..."):
            try:
                json_data, markdown_text = process_document(
                    file_path=str(file_path),
                    json_path=str(output_dir / f"{stem}.json"),
                    md_path=str(output_dir / f"{stem}.md")
                )
                st.session_state.update(
                    json_data=json_data,
                    markdown_text=markdown_text,
                    filename=stem,
                    json_tables=find_tables_in_json(json_data),
                    ai_summary=None,
                    chat_history=[]
                )
                doc_key = f"{stem}{file_path.suffix}"
                st.session_state.current_doc_key = doc_key
                st.session_state.document_history[doc_key] = {
                    "json_data": json_data,
                    "markdown_text": markdown_text,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "translations": {}
                }
            except Exception as e:
                st.error(f"Processing failed: {str(e)}")
                st.stop()

        # --- Auto Summary ---
        with st.spinner("Generating AI summary..."):
            try:
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(f"Provide a concise, professional summary of the document:\n\n{markdown_text}")
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
                    prompt = f"Translate the following text into {target_lang_name} ({lang_code}). Preserve all markdown formatting. Only return the translated text.\n\n{st.session_state.markdown_text}"
                    response = model.generate_content(prompt)
                    translated_text = response.text
                    if st.session_state.current_doc_key:
                        st.session_state.document_history[st.session_state.current_doc_key]["translations"][lang_code] = translated_text
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
            st.info("Select the content you wish to send and enter the recipient's details below.")

            with st.form("email_form"):
                recipient = st.text_input("Recipient's Email", placeholder="name@example.com")
                subject = st.text_input("Subject", value=f"Document Analysis: {st.session_state.filename}")

                st.markdown("**Select content to include:**")
                include_summary = st.checkbox("Include AI Summary (in email body)", value=True)
                attach_md = st.checkbox("Attach Markdown File (.md)", value=True)
                attach_json = st.checkbox("Attach JSON File (.json)")
                
                # Only show the option to attach tables if tables were found
                attach_tables = False
                if st.session_state.json_tables:
                    attach_tables = st.checkbox("Attach Extracted Tables (.xlsx)")

                custom_message = st.text_area("Custom Message (Optional)", placeholder="Add a personal note here...")
                
                # A single, final submit button
                submitted = st.form_submit_button("✉️ Send Email")

                if submitted:
                    if not recipient:
                        st.warning("Please enter a recipient's email address.")
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

                                # Call the updated email function
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
    st.sidebar.subheader("Recent Documents")
    sorted_history = sorted(st.session_state.document_history.items(), key=lambda item: item[1]['timestamp'], reverse=True)
    for doc_name, data in sorted_history:
        with st.sidebar.expander(f"{doc_name} ({data['timestamp']})"):
            st.download_button("JSON", json.dumps(data["json_data"], indent=2), f"{Path(doc_name).stem}.json", use_container_width=True, key=f"hist_json_{doc_name}")
            st.download_button("Markdown", data["markdown_text"], f"{Path(doc_name).stem}.md", use_container_width=True, key=f"hist_md_{doc_name}")
            if data.get("translations"):
                st.markdown("**Translations:**")
                lang_map_rev = {v: k for k, v in {"Hindi": "hi", "Tamil": "ta", "Telugu": "te", "French": "fr", "Spanish": "es", "German": "de"}.items()}
                for lang_code, translated_text in data["translations"].items():
                    lang_name = lang_map_rev.get(lang_code, lang_code.upper())
                    st.download_button(f"Download ({lang_name})", translated_text, f"{Path(doc_name).stem}_{lang_code}.md", use_container_width=True, key=f"hist_trans_{doc_name}_{lang_code}")

# -----------------------------
# FOOTER
# -----------------------------
st.divider()
st.caption("<p style='text-align:center;'>© 2025 IMDU Document Parser by TEAM CODEX</p>", unsafe_allow_html=True)