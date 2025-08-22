from futurehouse_client import FutureHouseClient, JobNames
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("API_KEY3")
if not api_key:
    raise ValueError("❌ API_KEY2 is not set or is empty")

print(f"API key loaded: {api_key[:10]}...")

try:
    client = FutureHouseClient(api_key=api_key)
    print("✅ Client initialized successfully")
except Exception as e:
    print(f"❌ Initialization failed: {e}")
    raise

# Remember to close gracefully
finally:
    client.close()
