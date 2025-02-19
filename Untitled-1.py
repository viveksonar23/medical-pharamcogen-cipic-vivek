import os
import openai
import streamlit as st  



def load_llm():
    DEEPSEEK_API_KEY = os.environ.get("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")
    
    if not DEEPSEEK_API_KEY:
        st.error("DeepSeek API key is not set.")
        return None
    
    # Configure DeepSeek API
    openai.api_key = DEEPSEEK_API_KEY
    openai.api_base = "https://api.deepseek.com/v1"  # Replace with DeepSeek API endpoint

    return openai