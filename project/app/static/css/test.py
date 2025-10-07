import google.generativeai as genai
import os

# ✅ Load API key from environment variable (recommended)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

try:
    models = genai.list_models()

    print("✅ Available Gemini Models:\n")
    for model in models:
        print(f"Model Name: {model.name}")
        print(f"Display Name: {model.display_name}")
        print(f"Supported Methods: {model.supported_generation_methods}")
        print("-" * 40)

except Exception as e:
    print("❌ Error fetching models:", e)
