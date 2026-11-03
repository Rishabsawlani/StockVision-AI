import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Gemini API key not found!")

client = genai.Client(api_key=api_key)

try:
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents="Explain stock market volatility in two simple sentences."
    )

    print("\nGemini API connection successful!")
    print(response.text)

except Exception as error:
    print("\nAPI request failed:")
    print(type(error).__name__)
    print(error)