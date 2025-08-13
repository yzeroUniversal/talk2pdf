# PDF Q&A Tool

A Python-based tool that ingests PDF files, stores their content in a vector database, and allows you to query them interactively using a local LLM via [Ollama](https://ollama.com/).

## ✨ Features
- 📄 Automatic PDF ingestion from a folder
- 💾 Persistent [Chroma](https://www.trychroma.com/) vector database
- 🔍 Retrieval-based Q&A using [LangChain](https://www.langchain.com/)
- 🧠 Configurable HuggingFace embeddings and LLM model
- ⚡ Fast, local, and offline-friendly

---

## 📦 Installation
Clone this repository and install dependencies:
```bash
git clone https://github.com/yourusername/pdf-qa-tool.git
cd pdf-qa-tool
pip install -r requirements.txt


🚀 Quick Start

1. Add PDFs
Place your PDF files in the pdfs/ folder.
Example:
    pdfs/
├── example1.pdf
├── example2.pdf

2. Run the Script
python pdf_qa.py --pdf-folder pdfs --persist-dir vector_db --embeddings-model all-MiniLM-L6-v2 --llm-model llama3

3. Ask Questions
When prompted: ❓ Enter a question (or 'exit' to quit): What is the main topic of example1.pdf?

4. Exit
Type exit to quit the interactive session.

# COMMAND LINE OPTIONS

| Option               | Default            | Description                           |
| -------------------- | ------------------ | ------------------------------------- |
| `--pdf-folder`       | `pdfs`             | Folder containing PDF files           |
| `--persist-dir`      | `vector_db`        | Directory to save the vector DB       |
| `--embeddings-model` | `all-MiniLM-L6-v2` | HuggingFace embeddings model          |
| `--llm-model`        | `llama3`           | Ollama model to use                   |
| `--chunks`           | `4`                | Number of document chunks to retrieve |
