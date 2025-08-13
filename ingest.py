#!/usr/bin/env python3
"""
PDF Q&A Tool
------------

Reads all PDFs in a folder, builds a local vector DB, 
and lets you ask questions about them using a local LLM.

If no PDFs are found, it'll grab a sample one for you 
so you can test it right away.

Author: Your Name
GitHub: https://github.com/yourusername
"""

import os
import glob
import time
import argparse
from urllib.request import urlretrieve
from typing import List

# LangChain bits
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Updated embeddings and LLM packages
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM


# ------------------------------------------------------------
# Helper: download a public domain PDF so first-time users
# can try the tool without hunting for files.
# ------------------------------------------------------------
def download_sample_pdf(folder: str) -> str:
    os.makedirs(folder, exist_ok=True)
    sample_path = os.path.join(folder, "sample.pdf")
    sample_url = "https://www.archives.gov/files/founding-docs/declaration_transcript.pdf"

    print("📥 No PDFs found — grabbing a sample one...")
    try:
        urlretrieve(sample_url, sample_path)
        print(f"✅ Downloaded sample PDF: {sample_path}")
        return sample_path
    except Exception as e:
        print(f"❌ Could not download sample PDF: {e}")
        return ""


# ------------------------------------------------------------
# Load PDFs from folder into a list of LangChain Documents
# ------------------------------------------------------------
def load_all_pdfs(folder: str) -> List:
    docs = []
    pdf_files = glob.glob(os.path.join(folder, "*.pdf"))

    # Auto-download if empty
    if not pdf_files:
        sample = download_sample_pdf(folder)
        if sample:
            pdf_files.append(sample)
        else:
            return docs

    print(f"📄 Found {len(pdf_files)} PDF(s) in '{folder}'")

    for pdf in pdf_files:
        print(f"🔹 Loading: {pdf}")
        loader = PyPDFLoader(pdf)
        file_docs = loader.load()

        # Tag each doc with the filename it came from
        for d in file_docs:
            d.metadata["source"] = os.path.basename(pdf)
        docs.extend(file_docs)

    return docs


# ------------------------------------------------------------
# Build a persistent vector DB from the documents
# ------------------------------------------------------------
def build_vector_db(docs: List, persist_dir: str, model_name: str) -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    print("⚙️ Building vector database (this might take a sec)...")
    db = Chroma.from_documents(docs, embedding=embeddings, persist_directory=persist_dir)
    # No need to call db.persist() — saving happens automatically
    print(f"✅ Vector database ready at '{persist_dir}'")
    return db


# ------------------------------------------------------------
# Start an interactive Q&A loop
# ------------------------------------------------------------
def start_qa(persist_dir: str, model_name: str, llm_model: str, chunks: int) -> None:
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": chunks})

    llm = OllamaLLM(model=llm_model, callbacks=[StreamingStdOutCallbackHandler()])
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",
                                     retriever=retriever, return_source_documents=True)

    while True:
        query = input("\n❓ Ask me something (or type 'exit'): ").strip()
        if query.lower() == "exit":
            print("👋 Goodbye!")
            break
        if not query:
            continue

        t0 = time.time()
        res = qa(query)
        t1 = time.time()

        print(f"\n💬 {res['result']}")
        print(f"⏱ Took {t1 - t0:.2f}s")

        # Show where the answer came from
        for doc in res["source_documents"]:
            print(f"📄 Source: {doc.metadata.get('source', 'Unknown')}")


# ------------------------------------------------------------
# CLI arguments
# ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Ask questions about your PDFs using a local LLM.")
    parser.add_argument("--pdf-folder", default="pdfs", help="Folder with your PDFs")
    parser.add_argument("--persist-dir", default="vector_db", help="Where to save the vector DB")
    parser.add_argument("--embeddings-model", default="all-MiniLM-L6-v2", help="HuggingFace embeddings model")
    parser.add_argument("--llm-model", default=os.environ.get("MODEL", "llama3"), help="Ollama model to use")
    parser.add_argument("--chunks", type=int, default=int(os.environ.get("TARGET_SOURCE_CHUNKS", 4)),
                        help="Number of document chunks to retrieve per query")
    return parser.parse_args()


# ------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()

    docs = load_all_pdfs(args.pdf_folder)
    if not docs:
        print("❌ No documents loaded. Exiting.")
        exit(1)

    build_vector_db(docs, args.persist_dir, args.embeddings_model)
    start_qa(args.persist_dir, args.embeddings_model, args.llm_model, args.chunks)
