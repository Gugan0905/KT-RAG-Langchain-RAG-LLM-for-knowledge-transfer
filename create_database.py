from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from PyPDF2 import PdfFileReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.vectorstores.chroma import Chroma
import os
import shutil
import glob

CHROMA_PATH = "chroma"
DATA_PATH = "data/Image-Manipulation-and-Enhancement-Java-UI-UX"
inference_api_key = "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"


def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

def pdf_to_md(pdf_file_path, md_file_path):
    with open(pdf_file_path, 'rb') as pdf_file:
        pdf_reader = PdfFileReader(pdf_file)
        with open(md_file_path, 'w') as md_file:
            for page_num in range(pdf_reader.numPages):
                page = pdf_reader.getPage(page_num)
                md_file.write(page.extractText())
                md_file.write('\n\n')

def other_to_md(input_file_path, md_file_path):
    with open(input_file_path, 'r') as input_file:
        content = input_file.read()
        with open(md_file_path, 'w') as md_file:
            md_file.write(content)

def load_documents():
    documents = []
    md_directory = './data/MDfiles/'
    os.makedirs(md_directory, exist_ok=True)
    for root, _, files in os.walk(DATA_PATH):
        for f in files:
            file_path = os.path.join(root, f)
            if f.endswith('.pdf'):
                print("Reading PDF - ", f)
                pdf_to_md(file_path, md_directory+str(f)+'.md')
            
            elif f.endswith('.java') or f.endswith('.txt'):
                print("Reading - ", f)
                other_to_md(file_path, md_directory+str(f)+'.md')
    
    loader = DirectoryLoader(md_directory, glob="*.md")
    documents = loader.load()
    return documents


def split_text(documents):
    print('Splitting Text')
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    # chunks = []
    # for doc in documents:
    #     chunked_doc = text_splitter.split_documents(doc)
    #     chunks.extend(chunked_doc)
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks


def save_to_chroma(chunks):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    
    print('Embedding...')
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=inference_api_key, 
        model_name="sentence-transformers/all-MiniLM-l6-v2"
    )
    
    # Create a new DB from the documents.
    db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    main()
