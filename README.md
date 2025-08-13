# Overview
This is essentially a "local PDF question-answering" project.
It lets you:
Load PDFs from a folder.
Break them into chunks of text.
Convert chunks into vector embeddings (numerical meaning-representations).
Store those embeddings in a local ChromaDB database.
Ask natural-language questions, and the system retrieves relevant chunks and sends them to a Large Language Model (LLM) — here Ollama with LLaMA 3.1 — to answer based on your PDFs only.

# What this project does (in one sentence)
It builds a local Q&A over PDFs: PDFs → text chunks → embeddings in a Chroma vector DB → retrieve top chunks for a question → feed them to a local LLM via Ollama to answer

# Tech Stack & Why It’s Used
| Component                          | Purpose                                                                 | Why It’s Used                                                    |
| ---------------------------------- | ----------------------------------------------------------------------- | ---------------------------------------------------------------- |
| **LangChain**                      | Orchestration of document loading, splitting, embedding, and retrieval. | Saves you from writing boilerplate; integrates with many models. |
| **PyMuPDFLoader**                  | PDF reading.                                                            | Reliable and fast PDF text extraction.                           |
| **RecursiveCharacterTextSplitter** | Splits text into chunks for embeddings.                                 | Maintains sentence coherence while chunking.                     |
| **HuggingFaceEmbeddings**          | Turns text into numerical vectors.                                      | HuggingFace offers quality pre-trained sentence transformers.    |
| **Chroma**                         | Vector database.                                                        | Fast, local, persistent store for embeddings.                    |
| **Ollama**                         | Runs LLaMA and other models locally.                                    | Avoids sending data to external servers.                         |
| **Multiprocessing + tqdm**         | Loads PDFs faster & shows progress.                                     | Good for large document sets.                                    |

