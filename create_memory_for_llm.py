# embedding_code.py

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Step 1: Load raw PDF(s)
DATA_PATH = "data/"

def load_pdf_files(data):
    """
    Loads PDFs using DirectoryLoader + PyPDFLoader,
    each page is a separate Document. We then ensure each Document 
    has doc.metadata["page"] for page numbering.
    """
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()

    # If PyPDFLoader doesn't already set doc.metadata["page"],
    # we'll explicitly add it here:
    for i, doc in enumerate(documents):
        # If "page" isn't present, set it to i+1:
        if "page" not in doc.metadata:
            doc.metadata["page"] = i + 1

        # 'doc.metadata["source"]' should contain the PDF file path
        # but you can confirm or replace if needed.

    return documents

documents = load_pdf_files(data=DATA_PATH)

# Step 2: Create Chunks
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks = create_chunks(extracted_data=documents)

# Step 3: Create Vector Embeddings
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embedding_model = get_embedding_model()

# Step 4: Store embeddings in FAISS
DB_FAISS_PATH = "vectorstore/db_faiss"
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)

print("✅ FAISS index saved locally with page numbers in doc.metadata['page']!")
