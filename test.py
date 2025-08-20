# test_gmail.py
import os
import smtplib
import ssl

def run_gmail_login_test():
    """Attempts to log in to the Gmail SMTP server to verify credentials."""
    
    # 1. Load credentials from environment variables
    sender_email = "sundarvikas01@gmail.com"
    app_password = "rxkkennktixothdy"

    if not sender_email or not app_password:
        print("❌ FAILURE: GMAIL_ADDRESS or GMAIL_APP_PASSWORD not found.")
        print("-> Make sure you set both variables in your terminal before running.")
        return

    print(f"🔑 Credentials loaded for: {sender_email}")
    print("-> Attempting to connect to smtp.gmail.com on port 465...")

    try:
        # 2. Create a secure SSL context
        context = ssl.create_default_context()
        
        # 3. Connect and log in to the server
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, app_password)
        
        # 4. If login succeeds, print a success message
        print("\n✅ SUCCESS! Login to Gmail was successful.")
        print("-> Your credentials are correct and the connection works.")

    except smtplib.SMTPAuthenticationError:
        print("\n💥 FAILURE: SMTP Authentication Error (code: 535)")
        print("-> This means your App Password or email address is incorrect.")
        print("-> Please double-check the 16-character password (no spaces!) and try again.")
    
    except Exception as e:
        print(f"\n💥 FAILURE: An unexpected error occurred.")
        print(f"-> Error details: {e}")

if __name__ == "__main__":
    run_gmail_login_test()