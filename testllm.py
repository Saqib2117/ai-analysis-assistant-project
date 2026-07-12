"""
Test DeepSeek API Connection
Run: python test_deepseek.py
"""

import os
import requests
from dotenv import load_dotenv

# Load .env file
load_dotenv()

print("=" * 50)
print("🔍 Testing DeepSeek API Connection")
print("=" * 50)

# Get API key
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Check if key exists
if not DEEPSEEK_API_KEY:
    print("❌ DEEPSEEK_API_KEY not found in .env file")
    print("Please add: DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    exit(1)

print(f"✅ API Key found: {DEEPSEEK_API_KEY[:15]}...{DEEPSEEK_API_KEY[-5:]}")

# Test DeepSeek API
try:
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "user", "content": "Say 'Hello! DeepSeek API is working!'"}
        ],
        "temperature": 0.3,
        "max_tokens": 50
    }
    
    response = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers=headers,
        json=data,
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        answer = result["choices"][0]["message"]["content"]
        print("\n✅ DeepSeek API is working!")
        print(f"Response: {answer}")
        print("\n" + "=" * 50)
        print("✅ API Test PASSED! You can proceed with the project.")
        print("=" * 50)
    else:
        print(f"\n❌ API Error: {response.status_code}")
        print(f"Response: {response.text}")
        print("\nPossible issues:")
        print("1. API key is invalid or expired")
        print("2. API key format is wrong (should start with 'sk-')")
        print("3. No internet connection")
        print("4. DeepSeek API may have changed")

except requests.exceptions.Timeout:
    print("❌ Connection timeout: DeepSeek API took too long to respond")
except requests.exceptions.ConnectionError:
    print("❌ Connection Error: No internet connection")
except Exception as e:
    print(f"❌ Error: {e}")