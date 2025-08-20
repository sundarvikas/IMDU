# email_report.py
import os
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path
from typing import List

def send_report_email(recipient: str, subject: str, custom_message: str, summary_text: str, attachment_paths: List[str]):
    """
    Sends a customizable email with attachments using a Gmail account.
    This version uses a more robust MIME structure for mixed content.
    """
    sender_email = "sundarvikas01@gmail.com"
    app_password = "rxkkennktixothdy"

    if not sender_email or not app_password:
        return False, "Error: GMAIL_ADDRESS or GMAIL_APP_PASSWORD not set."

    # Use the default MIMEMultipart, which is best for mixed text and attachments.
    message = MIMEMultipart()
    message["Subject"] = subject or "Your Document Analysis Report"
    message["From"] = f"IMDU Report <{sender_email}>"
    message["To"] = recipient

    # Build the HTML body
    html_body = "<html><body>"
    if custom_message:
        html_body += f"<p>{custom_message.replace(os.linesep, '<br>')}</p><hr>"
    if summary_text:
        html_body += "<h3>AI-Generated Summary</h3>"
        html_body += f"<pre style='font-family: monospace; white-space: pre-wrap; padding: 10px; background-color: #f4f4f4; border-radius: 5px;'>{summary_text}</pre><hr>"
    html_body += "<p>This email was sent from the IMDU Application.</p>"
    html_body += "</body></html>"
    message.attach(MIMEText(html_body, "html"))

    # Attach files
    for file_path_str in attachment_paths:
        file_path = Path(file_path_str)
        if file_path.is_file():
            with open(file_path, "rb") as attachment:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename= {file_path.name}")
            message.attach(part)

    # Send the email
    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, app_password)
            server.sendmail(sender_email, recipient, message.as_string())
        return True, "Email sent successfully!"
    except Exception as e:
        return False, f"An error occurred: {e}"