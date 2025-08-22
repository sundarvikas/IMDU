# virustotal_service.py
import streamlit as st
import requests
import base64
import time

class VirusTotalService:
    """
    A service to interact with the VirusTotal API for URL analysis.
    It uses a list of API keys from Streamlit secrets for rate limiting.
    """
    def __init__(self):
        try:
            # ✅ Securely get credentials from st.secrets
            self.api_keys = ["d686ce1c961994b8361c0325bb827bc54f1072c0c703b5e56da0482728384246",
"267f0eea0f6a005db1f443516e6c8db1a290a541a88e200ced98efc0cce4e3cf",
"061ab8204850d0cf60ab58fc6d4167955ccd23e4f1a365877e392ecfae453d11"
]
            if not isinstance(self.api_keys, list) or not self.api_keys:
                raise KeyError
        except KeyError:
            st.error("VirusTotal API keys not found or misconfigured in secrets.toml.")
            st.info("Please add [virustotal] with api_keys = ['key1', 'key2'] to your secrets.")
            self.api_keys = []
        
        self.current_key_index = 0
        self.base_url = "https://www.virustotal.com/api/v3"

    def _get_next_key(self):
        """Rotates to the next available API key."""
        if not self.api_keys:
            return None
        key = self.api_keys[self.current_key_index]
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        return key

    def analyze_url(self, url: str):
        """
        Analyzes a single URL with VirusTotal and returns a summary.
        
        Args:
            url (str): The URL to analyze.

        Returns:
            dict: A dictionary containing the analysis results.
        """
        if not self.api_keys:
            return {"error": "API keys not configured."}

        # VirusTotal API requires URL-safe base64 without padding
        url_id = base64.urlsafe_b64encode(url.encode()).decode().strip("=")
        endpoint = f"{self.base_url}/urls/{url_id}"
        
        headers = {
            "accept": "application/json",
            "x-apikey": self._get_next_key()
        }

        try:
            response = requests.get(endpoint, headers=headers)
            
            # If the request is rate-limited (429), switch key and retry once.
            if response.status_code == 429:
                time.sleep(1) # Small delay before retrying
                headers["x-apikey"] = self._get_next_key()
                response = requests.get(endpoint, headers=headers)

            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
            
            data = response.json()
            stats = data.get("data", {}).get("attributes", {}).get("last_analysis_stats", {})
            
            return {
                "url": url,
                "status": "Completed",
                "stats": {
                    "harmless": stats.get("harmless", 0),
                    "malicious": stats.get("malicious", 0),
                    "suspicious": stats.get("suspicious", 0),
                    "undetected": stats.get("undetected", 0)
                },
                "report_link": f"https://www.virustotal.com/gui/url/{url_id}"
            }

        except requests.exceptions.HTTPError as http_err:
            if http_err.response.status_code == 404:
                return {"url": url, "status": "Not Found", "error": "This URL has not been analyzed by VirusTotal before."}
            return {"url": url, "status": "Error", "error": f"HTTP Error: {http_err}"}
        except Exception as e:
            return {"url": url, "status": "Error", "error": str(e)}