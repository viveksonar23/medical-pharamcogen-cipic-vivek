import os
import openai
import streamlit as st  



def load_llm():
    if not DEEPSEEK_API_KEY:
        st.error("DeepSeek API key is not set.")
        return None
    
    # Configure DeepSeek API
    openai.api_key = DEEPSEEK_API_KEY
    openai.api_base = "https://api.deepseek.com/v1"  # Replace with DeepSeek API endpoint

    return openai
