import openai
import os

# Retrieve API key from environment variables
OPENAI_API_KEY = os.environ.get("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")

if not OPENAI_API_KEY:
    print("Error: OpenAI API key is not set. Please set the environment variable OPENAI_API_KEY.")
else:
    print("OpenAI API key is set. Testing connection...")

    # Set the API key
    openai.api_key = OPENAI_API_KEY
    openai.api_base = "https://api.openai.com/v1"

    # Test API by making a request
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello, can you confirm my API key is working?"}]
        )
        print("API Key is valid! Response from OpenAI:")
        print(response["choices"][0]["message"]["content"])
    except Exception as e:
        print(f"Error: {e}")
