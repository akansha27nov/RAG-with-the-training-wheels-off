"""
PDF loading and chunking utilities for the RAG demo.
Turns the source document into overlapping LangChain chunks.
"""

from pypdf import PdfReader
from config import PDF_PATH
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdf(path=PDF_PATH):
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def get_chunks(chunk_size=1200, chunk_overlap=200):
    print(f"Loading PDF from {PDF_PATH}...")
    pdf_text = load_pdf()
    
    pdf_document = Document(page_content=pdf_text, metadata={"source": PDF_PATH})
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap, 
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunks = splitter.split_documents([pdf_document])
    print(f"Successfully created {len(chunks)} chunks.")
    return chunks
