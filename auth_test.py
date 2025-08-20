import os
from supabase import create_client
from dotenv import load_dotenv

print("Attempting to connect to Supabase...")

try:
    load_dotenv() # Load variables from .env file

    url = "https://pmnvfgmxufsabqdarphz.supabase.co"
    key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBtbnZmZ214dWZzYWJxZGFycGh6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTU3MTAzMzIsImV4cCI6MjA3MTI4NjMzMn0.VMqp7vJzsjQvi-bWyq-SdOKAwW8kF_iW6EAmV7mlfPI"
    test_email = "sundarvikas01@gmail.com"
    test_password = "123456"

    if not all([url, key, test_email, test_password]):
        print("❌ Error: Please make sure all variables are set in the .env file.")
    else:
        supabase = create_client(url, key)
        print("✅ Supabase client created.")
        
        print(f"Attempting to sign in with email: {test_email}...")
        session = supabase.auth.sign_in_with_password({
            "email": test_email, 
            "password": test_password
        })
        
        if session.user:
            print("\n🎉 SUCCESS! Logged in successfully.")
            print(f"User ID: {session.user.id}")
        else:
            print("\n❌ FAILED! Could not log in, but connected successfully.")

except Exception as e:
    print(f"\n❌ FAILED with an error: {e}")