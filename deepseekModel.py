import os
import requests
from bs4 import BeautifulSoup
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import re
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
import urllib.parse
import openai

# Constants
DB_FAISS_PATH = "vectorstore/db_faiss"
SAVE_FOLDER = "downloaded_files"
DATA_PATH = r"D:\new_gait\medical-chatbot-main\medical-chatbot-main\data"

OPENAI_API_KEY = os.environ.get("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")

@st.cache_resource
def get_vectorstore():
    try:
        embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        st.error(f"Error loading FAISS vectorstore: {e}")
        return None

def apply_custom_styles():
    st.markdown("""
        <style>
        /* Set full-page dark purple background */
        html, body, [data-testid="stAppViewContainer"] {
            background-color: #2C1A47 !important;
            color: white !important;
        }
        /* Customizing Sidebar */
        [data-testid="stSidebar"] {
            background-color: #3A1D5F !important;
        }
        /* Button Styling */
        div.stButton > button {
            background-color: #5E3B8A;
            color: white;
            border-radius: 8px;
            padding: 12px;
            font-size: 16px;
            transition: 0.3s;
            border: none;
        }
        div.stButton > button:hover {
            background-color: #8A5EDA;
            transform: scale(1.05);
            box-shadow: 0px 0px 10px rgba(255, 255, 255, 0.5);
        }
        /* Textbox Styling */
        textarea, input {
            background-color: #3A1D5F !important;
            color: white !important;
            border-radius: 8px;
            border: 1px solid #8A5EDA !important;
        }
        /* Centering Elements */
        .centered {
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)

def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920x1080")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

def extract_pdf_links(url):
    try:
        st.info("Opening webpage in headless mode...")
        driver = setup_driver()
        driver.get(url)
        time.sleep(5)  # Allow time for JavaScript to load

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        guideline_links = [link['href'] for link in soup.find_all('a', href=True) if "guideline" in link['href'].lower()]

        pdf_links = []
        for guideline_link in guideline_links:
            if not guideline_link.startswith("http"):
                guideline_link = requests.compat.urljoin(url, guideline_link)
            
            driver.get(guideline_link)
            time.sleep(3)
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            pdfs = [link['href'] for link in soup.find_all('a', href=True) if link['href'].endswith('.pdf')]
            for pdf in pdfs:
                if not pdf.startswith("http"):
                    pdf = requests.compat.urljoin(guideline_link, pdf)
                pdf_links.append(pdf)

        driver.quit()
        return pdf_links
    except Exception as e:
        st.error(f"An error occurred while extracting PDF links: {str(e)}")
        return []

def download_pdfs(url):
    try:
        if not os.path.exists(SAVE_FOLDER):
            os.makedirs(SAVE_FOLDER)

        pdf_links = extract_pdf_links(url)
        if not pdf_links:
            st.warning("No PDF files found on the webpage.")
            return

        st.info(f"Found {len(pdf_links)} PDF files. Starting download...")

        for pdf_link in pdf_links:
            file_name = os.path.join(SAVE_FOLDER, os.path.basename(pdf_link))
            with requests.get(pdf_link, stream=True) as pdf_response:
                if pdf_response.status_code == 200:
                    with open(file_name, 'wb') as file:
                        for chunk in pdf_response.iter_content(chunk_size=1024):
                            file.write(chunk)
                    st.success(f"Downloaded: {file_name}")
                else:
                    st.error(f"Failed to download {pdf_link}")

        st.success(f"All PDF files have been downloaded to '{SAVE_FOLDER}' folder.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

def extract_zip_links(url):
    try:
        st.info("Opening webpage in headless mode...")
        driver = setup_driver()
        driver.get(url)
        time.sleep(5)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        driver.quit()

        zip_links = [link['href'] for link in soup.find_all('a', href=True) if link['href'].endswith('.zip')]
        return zip_links
    except Exception as e:
        st.error(f"An error occurred while extracting ZIP links: {str(e)}")
        return []

def download_zip_files(url):
    try:
        if not os.path.exists(SAVE_FOLDER):
            os.makedirs(SAVE_FOLDER)

        zip_links = extract_zip_links(url)
        if not zip_links:
            st.warning("No ZIP files found on the webpage.")
            return

        st.info(f"Found {len(zip_links)} ZIP files. Starting download...")

        for zip_link in zip_links:
            if not zip_link.startswith("http"):
                zip_link = requests.compat.urljoin(url, zip_link)
            file_name = os.path.join(SAVE_FOLDER, os.path.basename(zip_link))
            with requests.get(zip_link, stream=True) as zip_response:
                if zip_response.status_code == 200:
                    with open(file_name, 'wb') as file:
                        for chunk in zip_response.iter_content(chunk_size=1024):
                            file.write(chunk)
                    st.success(f"Downloaded: {file_name}")
                else:
                    st.error(f"Failed to download {zip_link}")

        st.success(f"All ZIP files have been downloaded to '{SAVE_FOLDER}' folder.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

def set_custom_prompt():
    return PromptTemplate(template="""
        Use the pieces of information provided in the context to answer user's question.
        If you don't know the answer, just say you don't know. Do not make up an answer.
        Do not provide anything outside the given context.

        Context: {context}
        Question: {question}

        Start the answer directly.
    """, input_variables=["context", "question"])

def load_llm():
    openai_api_key = os.environ.get("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")  # Fetching from environment

    if not openai_api_key:
        st.error("⚠️ OpenAI API key is not set. Please check your environment variables.")
        return None

    print(f"✅ OpenAI API Key Loaded: {openai_api_key}")  # Debugging step

    openai.api_key = openai_api_key
    openai.api_base = "https://api.openai.com/v1"

    return openai


def show_history():
    """Show chat history in the sidebar and allow clearing it."""
    st.sidebar.header("📜 History")

    if "offline_chat_history" not in st.session_state or len(st.session_state["offline_chat_history"]) == 0:
        st.sidebar.write("No conversation history yet.")
        return

    for i, item in enumerate(st.session_state["offline_chat_history"]):
        with st.sidebar.expander(f"Entry #{i+1}"):
            st.write(f"**Medications**: {item['medications']}")
            st.write(f"**Pharmacogenomics**: {item['pharmacogenomics']}")
            st.write(f"**Response**: {item['response']}")

    if st.sidebar.button("Clear History"):
        st.session_state["offline_chat_history"] = []
        st.rerun()

def chatbot_offline_interface():
    apply_custom_styles()
    show_history()

    st.title("💊 Offline Pharmacogen & CIPIC Chatbot")
    st.markdown("### Provide the required inputs below:")

    with st.container():
        st.subheader("🧠 What are the patient’s medications?")
        medications = st.text_area("Enter medications here:", placeholder="Tegretol, Lipitor, ciprofloxacin, ibuprofen")

        st.markdown(
            "<span style='color: lightblue; cursor: pointer; text-decoration: underline;'>Import Excel Sheet</span>",
            unsafe_allow_html=True
        )
        uploaded_file = st.file_uploader("", type=["xlsx", "xls"], key="file_upload")

        if uploaded_file is not None:
            df = pd.read_excel(uploaded_file)
            st.write("### Preview of Uploaded Excel File:")

            selected_data = st.data_editor(df, num_rows="dynamic", key="selected_data")
            if selected_data is not None:
                selected_indices = selected_data.index[selected_data.notna().any(axis=1)].tolist()
                if selected_indices:
                    selected_medications = ", ".join(
                        df.iloc[selected_indices, 0].dropna().astype(str).tolist()
                    )
                    selected_pharmacogenomics = ", ".join(
                        df.iloc[selected_indices, 1].dropna().astype(str).tolist()
                    )
                    st.session_state["medications_val"] = selected_medications
                    st.session_state["pharmacogenomics_val"] = selected_pharmacogenomics

    with st.container():
        st.subheader("🧬 What is the patient’s pharmacogenomic information?")
        pharmacogenomics = st.text_area(
            "Enter pharmacogenomic data here:",
            placeholder="A/A, E3/E4, G/G, *1A/*1A, etc.",
            height=200
        )

    if st.button("Submit ✔️"):
        if medications and pharmacogenomics:
            st.success("Data received. Processing...")

            try:
                vectorstore = get_vectorstore()
                if vectorstore is None:
                    st.error("Failed to load the vector store.")
                    return

                llm = load_llm()
                if llm is None:
                    return

                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                    return_source_documents=True,
                    chain_type_kwargs={'prompt': set_custom_prompt()}
                )

                response = qa_chain.invoke({
                    'query': f"Patient Medications: {medications}\nPharmacogenomic Data: {pharmacogenomics}"
                })

                result = response["result"]
                source_docs = response["source_documents"]

                st.subheader("Answer:")
                st.write(result)

                if source_docs:
                    st.markdown("**Source Documents:**")
                    for i, doc in enumerate(source_docs, start=1):
                        source_obj = doc.metadata.get("source", "Unknown PDF")
                        page_num = doc.metadata.get("page", 1)

                        # Convert sets or other types to a string
                        if isinstance(source_obj, set):
                            if len(source_obj) > 0:
                                source_path = list(source_obj)[0]
                            else:
                                source_path = "Unknown PDF"
                        elif isinstance(source_obj, str):
                            source_path = source_obj
                        else:
                            source_path = str(source_obj)

                        st.write(f"Document {i}: {source_path}, Page: {page_num}")

                        # Safely handle the path
                        abs_path = os.path.abspath(source_path)
                        normalized_path = abs_path.replace("\\", "/")
                        quoted_path = urllib.parse.quote(normalized_path)
                        pdf_link = f"file:///{quoted_path}#page={page_num}"

                        st.markdown(
                            f'<a href="{pdf_link}" target="_blank">Open PDF at Page {page_num}</a>',
                            unsafe_allow_html=True
                        )
                else:
                    st.write("No source documents returned.")

                # Append to history
                if "offline_chat_history" not in st.session_state:
                    st.session_state["offline_chat_history"] = []

                st.session_state["offline_chat_history"].append({
                    "medications": medications,
                    "pharmacogenomics": pharmacogenomics,
                    "response": result
                })

            except Exception as e:
                st.error(f"Error during chatbot interaction: {e}")
        else:
            st.error("Please fill out both fields before submitting.")

    if st.button("☚ Back"):
        st.session_state.page = "chatbot_mode"
        st.rerun()

def chatbot_mode_selection():
    apply_custom_styles()
    st.markdown("<h1 class='centered'>🚑 Select PharmacoGen-CIPIC Chatbot Mode</h1>", unsafe_allow_html=True)

    if st.button("Offline Mode 💉"):
        st.session_state.page = "chatbot_offline"
        st.rerun()

    if st.button("Online Mode 💉"):
        st.session_state.page = "chatbot_online"
        st.rerun()

    if st.button("☚ Back"):
        st.session_state.page = "main"
        st.rerun()

def main():
    apply_custom_styles()
    st.markdown("<h1 class='centered'>💊 PharmacoGen & CIPIC Auto-Downloader</h1>", unsafe_allow_html=True)
    st.markdown("<h3 class='centered'>Download ZIP & PDF Files Automatically</h3>", unsafe_allow_html=True)

    url = st.text_input("Enter the webpage URL", placeholder="https://www.pharmgkb.org/downloads")

    if st.button("Download ZIP & PDF Files"):
        if url:
            download_zip_files(url)
            download_pdfs(url)
        else:
            st.error("Please enter a valid URL.")

    st.markdown("<h1 class='centered'>🤖 PharmacoGen-CIPIC Chatbot</h1>", unsafe_allow_html=True)
    if st.button("Start Chatbot"):
        st.session_state.page = "chatbot_mode"
        st.rerun()

def app():
    if "page" not in st.session_state:
        st.session_state.page = "main"

    if st.session_state.page == "main":
        main()
    elif st.session_state.page == "chatbot_mode":
        chatbot_mode_selection()
    elif st.session_state.page == "chatbot_offline":
        chatbot_offline_interface()

if __name__ == "__main__":
    app()