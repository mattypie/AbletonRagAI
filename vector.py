from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

#Variables
DB_LOCATION = "./chrome_langchain_db"
ADD_DOCUMENTS = not os.path.exists(DB_LOCATION)
CHUNK_SIZE = 600
CHUNK_OVERLAP = 80
K_VECTOR = 5
SCORE_TRESHOLD_VECTOR = 0.4
MAX_BATCH_SIZE = 500

#Clean Function
def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)  # collapse multiple spaces/newlines
    text = re.sub(r'Page\s*\d+\s*/\s*\d+', '', text, flags=re.IGNORECASE)  # remove page indicators
    return text.strip()

#Batch documents to not exceed embedding limit
def batch_documents(docs, batch_size):
    for i in range(0, len(docs), batch_size):
        yield docs[i:i + batch_size]

#Load embedding model
embeddings = OllamaEmbeddings(model = "nomic-embed-text")

#Add different manuals
pdf_files = ["Documents/live12-manual-en.pdf",
            "Documents/101_Ableton_Tips.pdf",
            "Documents/MakingMusic_DennisDeSantis.pdf",
            "Documents/Ableton12.pdf"
            #"Documents/.pdf", #Use when adding more pdf's
]

#Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size = CHUNK_SIZE, 
    chunk_overlap = CHUNK_OVERLAP
)

docs = [] #Where the documents end up

if ADD_DOCUMENTS:
#Split into chunks
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        #pages = [page.extract_text() for page in reader.pages]
        for page_number, page in enumerate(reader.pages, start = 1):
            raw_text = page.extract_text()
            if not raw_text:
                continue
            text = clean_text(raw_text)
            if not text:
                continue
            chunks = splitter.split_text(text)
            for chunk in chunks:
                docs.append(Document
                       (page_content=chunk, 
                        metadata={
                            "source": os.path.basename(pdf), 
                            "page": page_number}))

#Vector storage
vector_store = Chroma(
    collection_name="ableton_manual",
    persist_directory=DB_LOCATION,
    embedding_function=embeddings
)

if ADD_DOCUMENTS and docs:
    for batch in batch_documents(docs, MAX_BATCH_SIZE):  # Safe batch size
        vector_store.add_documents(batch)
    print(f"Added {len(docs)} chunks to the vector database in batches.")
else:
    print("Using existing database.")

#creates retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": K_VECTOR}
)
    
print("PDF embeddngs are in:", DB_LOCATION)