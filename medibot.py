import os
import requests
from bs4 import BeautifulSoup
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
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



# HF_TOKEN = os.getenv("HF_TOKEN")



if "HF_TOKEN" not in st.secrets:
    st.error("❌ Hugging Face token is missing. Add it in Streamlit Secrets Manager!")
    HF_TOKEN = None
else:
    HF_TOKEN = st.secrets["HF_TOKEN"]

# For DOCX support
try:
    import docx
except ImportError:
    st.error("Please install python-docx for Word file support: pip install python-docx")

# Constants
DB_FAISS_PATH = "vectorstore/db_faiss"
SAVE_FOLDER = "downloaded_files"
DATA_PATH = r"D:\new_gait\medical-chatbot-main\medical-chatbot-main\data"

@st.cache_resource
def get_vectorstore():
    try:
        embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        if not os.path.exists(DB_FAISS_PATH):
            st.warning("FAISS vector store not found. Creating a new one...")
            db = FAISS(embedding_model)  # Initialize FAISS
            db.save_local(DB_FAISS_PATH)
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
        Using the following context, please provide a detailed, comprehensive, and well-supported answer to the question below.
        Include step-by-step reasoning and reference relevant parts of the context where applicable.
        If there is not enough information, clearly state that you do not have sufficient data to answer, rather than guessing.

        Context: {context}
        Question: {question}

        Answer:
    """, input_variables=["context", "question"])

def load_llm2():
    # Replace with the model repo ID for Falcon-7B-Instruct
    HUGGINGFACE_REPO_ID = "tiiuae/falcon-7b-instruct"
    HF_TOKEN="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"
    if not HF_TOKEN:
        st.error("Hugging Face token is not set.")
        return None
    
    return HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": 1024}
    )

def load_llm():
    """
    Loads the Hugging Face LLM with the API token and explicitly sets a task.
    """
    HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"  # Replace with your model

    # Retrieve the Hugging Face token from Streamlit secrets or environment variables
    HF_TOKEN = os.getenv("HF_TOKEN", st.secrets.get("HF_TOKEN"))

    if not HF_TOKEN:
        st.error("❌ Hugging Face token is missing. Please add it to Streamlit secrets or as an environment variable.")
        return None

    try:
        # ✅ Explicitly specify the task
        return HuggingFaceEndpoint(
            repo_id=HUGGINGFACE_REPO_ID,
            temperature=0.5,
            task="text-generation",  # Make sure task matches the model type
            model_kwargs={"token": HF_TOKEN, "max_length": 1024}
        )
    except Exception as e:
        st.error(f"🚨 Error initializing LLM: {str(e)}")
        return None

def load_l2m():
    # Replace with the correct repo ID for your DeepSeek model
    HUGGINGFACE_REPO_ID = "deepseek-ai/DeepSeek-R1"  # Update to your model's repo id
    HF_TOKEN = os.environ.get("HF_TOKEN")
    if not HF_TOKEN:
        st.error("Hugging Face token is not set.")
        return None
    
    return HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": 1024}
    )

def load_llm2():
    # Replace with the correct repo ID for your DeepSeek model
    HUGGINGFACE_REPO_ID = "deepseek-ai/DeepSeek-Coder-V2-Base"  # Update to your model's repo id
    HF_TOKEN = os.environ.get("HF_TOKEN")
    if not HF_TOKEN:
        st.error("Hugging Face token is not set.")
        return None
    
    return HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": 1024}
    )


def process_uploaded_file(uploaded_file):
    """Process uploaded file and extract text based on file type."""
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    text = ""
    try:
        if file_extension == ".pdf":
            # Save temporary file and load using PyPDFLoader
            tmp_path = f"temp_{uploaded_file.name}"
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            text = "\n".join([doc.page_content for doc in docs])
            os.remove(tmp_path)
        elif file_extension == ".docx":
            doc = docx.Document(uploaded_file)
            fullText = []
            for para in doc.paragraphs:
                fullText.append(para.text)
            text = "\n".join(fullText)
        elif file_extension in [".xlsx", ".xls"]:
            df = pd.read_excel(uploaded_file)
            text = df.to_string(index=False)
        else:
            st.warning(f"Unsupported file type: {file_extension}")
    except Exception as e:
        st.error(f"Error processing file {uploaded_file.name}: {e}")
    return text

def show_history():
    """Show chat history for the offline chatbot (if needed)."""
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

#########################################
# ChatGPT‑Style Chat Interface
#########################################
def chatgpt_style_interface():
    apply_custom_styles()

    # Initialize conversation store if it doesn't exist
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}  # {conv_id: {"name": <str>, "messages": [ {role, content}, ... ] } }
        st.session_state.current_conv_id = None

    # Sidebar: Conversation list and new conversation button
    st.sidebar.title("Conversations")
    if st.sidebar.button("New Conversation"):
        new_conv_id = str(len(st.session_state.conversations) + 1)
        st.session_state.conversations[new_conv_id] = {
            "name": f"Conversation {new_conv_id}",
            "messages": []
        }
        st.session_state.current_conv_id = new_conv_id
        st.rerun()

    conv_ids = list(st.session_state.conversations.keys())
    if conv_ids:
        selected_conv = st.sidebar.radio("Select Conversation", conv_ids,
                                         index=conv_ids.index(st.session_state.current_conv_id)
                                         if st.session_state.current_conv_id in conv_ids else 0)
        st.session_state.current_conv_id = selected_conv
    else:
        st.session_state.current_conv_id = None

    # Main chat window
    if st.session_state.current_conv_id is None:
        st.write("No conversation selected. Please start a new conversation.")
        return

    conv = st.session_state.conversations[st.session_state.current_conv_id]
    st.header(conv["name"])

    # Display conversation messages
    for msg in conv["messages"]:
        if msg["role"] == "user":
            st.markdown(f"<div style='text-align: right; background-color: #5E3B8A; padding: 10px; border-radius: 10px; margin: 10px 0; display: inline-block;'>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='text-align: left; background-color: #3A1D5F; padding: 10px; border-radius: 10px; margin: 10px 0; display: inline-block;'>{msg['content']}</div>", unsafe_allow_html=True)

    # Optional file upload for additional context
    uploaded_files = st.file_uploader("Upload Files for Additional Context (Optional)",
                                      type=["pdf", "docx", "xlsx", "xls"],
                                      accept_multiple_files=True, key="chatgpt_file_upload")
    file_context = ""
    if uploaded_files:
        file_texts = []
        for uploaded_file in uploaded_files:
            extracted_text = process_uploaded_file(uploaded_file)
            if extracted_text:
                file_texts.append(f"Content from {uploaded_file.name}:\n{extracted_text}")
        file_context = "\n\n".join(file_texts)
        st.info("File context added.")

    # User message input
    user_input = st.text_area("Your Message", key="chatgpt_input", height=100)

    if st.button("Send"):
        if user_input.strip():
            # Append user's message
            conv["messages"].append({"role": "user", "content": user_input})
            
            # Load vectorstore and LLM (common to both query types)
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store.")
                return
            llm = load_llm()
            if llm is None:
                return

            # Check if the user input is a follow-up question that references the previous answer.
            # For example: "Give me the second point of the above paragraph"
            followup_match = re.search(r"(\d+)(?:st|nd|rd|th)\s+point", user_input.lower())
            if followup_match:
                point_number = followup_match.group(1)
                # Find the most recent assistant message
                last_assistant_message = ""
                for msg in reversed(conv["messages"]):
                    if msg["role"] == "assistant":
                        last_assistant_message = msg["content"]
                        break

                if last_assistant_message:
                    # Create a custom follow-up prompt template
                    followup_prompt = PromptTemplate(template="""
                        Given the following text:
                        {context}
                        Extract the {point} point from it.
                    """, input_variables=["context", "point"])

                    qa_chain_followup = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                        return_source_documents=True,
                        chain_type_kwargs={'prompt': followup_prompt}
                    )

                    response = qa_chain_followup.invoke({
                        "context": last_assistant_message,
                        "point": point_number
                    })
                    result = response["result"]
                    conv["messages"].append({"role": "assistant", "content": result})
                    st.rerun()
                else:
                    st.error("No previous answer available to reference.")
            else:
                # Standard query: include file context if available.
                query_input = f"{'File Content:\n' + file_context + '\n' if file_context else ''}{user_input}"
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                    return_source_documents=True,
                    chain_type_kwargs={'prompt': set_custom_prompt()}
                )
                response = qa_chain.invoke({'query': query_input})
                result = response["result"]
                conv["messages"].append({"role": "assistant", "content": result})
                st.rerun()

    # Sidebar options to clear or delete the current conversation
    if st.sidebar.button("Clear Current Conversation"):
        st.session_state.conversations[st.session_state.current_conv_id]["messages"] = []
        st.rerun()

    if st.sidebar.button("Delete Current Conversation"):
        del st.session_state.conversations[st.session_state.current_conv_id]
        st.session_state.current_conv_id = list(st.session_state.conversations.keys())[0] if st.session_state.conversations else None
        st.rerun()

#########################################
# Other Chatbot Modes (Offline & Online)
#########################################
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
        uploaded_excel = st.file_uploader("Upload Excel file", type=["xlsx", "xls"], key="file_upload_excel")
        if uploaded_excel is not None:
            df = pd.read_excel(uploaded_excel)
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

    with st.container():
        st.subheader("📎 Upload Files for Additional Context (Optional)")
        uploaded_files = st.file_uploader(
            "Choose file(s) [PDF, DOCX, Excel]",
            type=["pdf", "docx", "xlsx", "xls"],
            accept_multiple_files=True,
            key="file_upload_context"
        )
        file_context = ""
        if uploaded_files:
            file_texts = []
            for uploaded_file in uploaded_files:
                extracted_text = process_uploaded_file(uploaded_file)
                if extracted_text:
                    file_texts.append(f"Content from {uploaded_file.name}:\n{extracted_text}")
            file_context = "\n\n".join(file_texts)
            st.info("Extracted file content for context.")
        else:
            st.info("No additional files uploaded.")

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
                query_input = f"{'File Content:\n' + file_context + '\n' if file_context else ''}Patient Medications: {medications}\nPharmacogenomic Data: {pharmacogenomics}"
                response = qa_chain.invoke({'query': query_input})
                result = response["result"]
                source_docs = response["source_documents"]

                st.subheader("Answer:")
                st.write(result)

                if source_docs:
                    st.markdown("**Source Documents:**")
                    for i, doc in enumerate(source_docs, start=1):
                        source_obj = doc.metadata.get("source", "Unknown PDF")
                        page_num = doc.metadata.get("page", 1)
                        if isinstance(source_obj, set):
                            source_path = list(source_obj)[0] if source_obj else "Unknown PDF"
                        elif isinstance(source_obj, str):
                            source_path = source_obj
                        else:
                            source_path = str(source_obj)
                        st.write(f"Document {i}: {source_path}, Page: {page_num}")
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
    st.markdown("<h1 class='centered'>🚑 Select Chat Mode</h1>", unsafe_allow_html=True)

    if st.button(" Chat"):
        st.session_state.page = "Healthcare"
        st.rerun()

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
    elif st.session_state.page == "chatgpt_style":
        chatgpt_style_interface()

if __name__ == "__main__":
    app()
