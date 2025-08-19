# email_report.py
import os
import resend
import base64
from pathlib import Path
from typing import List

def send_report_email(recipient: str, subject: str, custom_message: str, summary_text: str, attachment_paths: List[str]):
    """
    Sends a highly customizable email with multiple optional attachments and text sections.
    """
    api_key = "re_5vZhvTzV_PfSsskdNMyZif5jHWjQn64YY"
    if not api_key:
        print("Error: RESEND_API_KEY environment variable not set.")
        # Return a failure status
        return False, "Error: RESEND_API_KEY environment variable is not set."

    resend.api_key = api_key
    
    # --- 1. Build the HTML content for the email body ---
    html_body = "<html><body>"
    if custom_message:
        html_body += f"<p>{custom_message.replace(os.linesep, '<br>')}</p><hr>"
    
    if summary_text:
        html_body += "<h3>AI-Generated Summary</h3>"
        html_body += f"<pre style='font-family: monospace; white-space: pre-wrap; padding: 10px; background-color: #f4f4f4; border-radius: 5px;'>{summary_text}</pre><hr>"
        
    html_body += "<p>This email was sent from the IMDU Application.</p>"
    html_body += "</body></html>"

    # --- 2. Prepare attachments ---
    attachments = []
    if attachment_paths:
        for file_path_str in attachment_paths:
            file_path = Path(file_path_str)
            if file_path.is_file():
                with open(file_path, "rb") as f:
                    content = f.read()
                encoded_content = base64.b64encode(content).decode("utf-8")
                attachments.append({
                    "filename": file_path.name,
                    "content": encoded_content
                })
            else:
                print(f"Warning: Attachment file not found at '{file_path_str}'. Skipping.")

    # --- 3. Send the email via Resend API ---
    try:
        params = {
            "from": "IMDU Report <onboarding@resend.dev>",
            "to": [recipient],
            "subject": subject or "Your Document Analysis Report",
            "html": html_body,
            "attachments": attachments,
        }
        
        email = resend.Emails.send(params)
        print(f"Email sent successfully. Message ID: {email['id']}")
        # Return a success status
        return True, f"Email sent successfully! (ID: {email['id']})"

    except Exception as e:
        print(f"An error occurred while sending the email: {e}")
        # Return a failure status
        return False, f"An error occurred: {e}"