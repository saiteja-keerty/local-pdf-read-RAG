 Local Multi-Modal RAG (PDF + Image Q&A)
📌 Repository Overview

This repository demonstrates a fully local Retrieval-Augmented Generation (RAG) system that allows users to:

📄 Upload a PDF and ask questions about it

🖼 Upload an image (e.g., electricity bill, invoice) and extract text using OCR

📄 + 🖼 Upload both PDF and image together and ask combined questions

🤖 Get answers using a local LLM (LLaMA 3 via Ollama)

Everything runs locally — no OpenAI API, no cloud calls, no data leaving your machine.

🚀 What This Project Does

This project provides two runnable Python applications:

1️⃣ single_pdf_input.py

Allows you to:

Upload one or multiple PDFs

Convert PDF text into embeddings

Store them in FAISS vector database

Ask questions grounded strictly in the uploaded PDF

Use case:

Upload a contract / research paper / policy document and ask questions about it.

2️⃣ image_pdf_input.py

Allows you to:

Upload PDFs

Upload Images (JPG, PNG)

Extract text from images using OCR (Tesseract)

Combine all text into one vector database

Ask cross-document questions

Use case:

Upload a solar quotation PDF + electricity bill image
Ask: “Is installing solar profitable based on my current bill?”

🧰 Tools & Technologies Used
Component	Tool
LLM	Ollama (LLaMA 3)
Framework	LangChain
Vector Database	FAISS
Embeddings	Sentence Transformers
UI	Gradio
OCR	Tesseract
Backend	Python
🧠 How It Works (Architecture)

📄 PDF → Extract text using PyPDF

🖼 Image → Extract text using Tesseract OCR

✂ Text → Split into chunks

🔢 Generate embeddings using sentence-transformers

💾 Store in FAISS vector DB

🔍 Retrieve relevant chunks for query

🤖 Send retrieved context to local LLM

💬 Generate grounded response

💻 Prerequisites

This project requires both Python dependencies and system-level tools.

🔹 Required Software
1️⃣ Python 3.10+

Check version:

python --version

2️⃣ Ollama (Local LLM)

Download:
https://ollama.com

After installation:

ollama pull llama3

3️⃣ Tesseract OCR (For Image Support)
Windows:

Download from:
https://github.com/UB-Mannheim/tesseract/wiki

During installation:
✔ Select “Add to PATH”

Restart terminal after installation.

Mac:
brew install tesseract

Linux:
sudo apt install tesseract-ocr

📦 Installation Steps
1️⃣ Clone Repository
git clone https://github.com/saiteja-keerty/local-pdf-read-RAG.git
cd local-pdf-read-RAG

2️⃣ Create Virtual Environment (Recommended)
python -m venv venv


Activate:

Windows:
venv\Scripts\activate

Mac/Linux:
source venv/bin/activate

3️⃣ Install Python Dependencies
pip install -r requirements.txt

▶️ How To Run
Run PDF-Only Version
python single_pdf_input.py

Run PDF + Image Version
python image_pdf_input.py


Then open browser:

http://127.0.0.1:7860

⚙️ Adjustments for Windows vs Mac
Feature	Windows	Mac
Activate venv	venv\Scripts\activate	source venv/bin/activate
Install Tesseract	Download installer	brew install tesseract
Ollama install	Download installer	brew install ollama
⚠ Known Limitations

First run downloads embedding model (~90MB)

Ollama model size may be several GB

OCR accuracy depends on image clarity

Retrieval is based on top-k similarity (may miss context)

LLM reasoning is approximate (not a financial calculator)

🔒 Privacy

No external APIs used

No OpenAI key required

All data processed locally

Documents remain on your system

📁 Repository Files
single_pdf_input.py   → PDF-based RAG
image_pdf_input.py    → Multi-modal RAG (PDF + Image)
requirements.txt      → Python dependencies
.gitignore            → Ignore virtual environment
README.md             → Project documentation

🎯 Example Questions

“Summarize this contract.”

“What is the total installation cost?”

“Is solar installation profitable based on my bill?”

“What are the warranty terms?”

🚀 Future Improvements

Structured financial ROI calculator

Persistent FAISS storage

Streaming token output

LangGraph agent workflow

Docker container support

Advanced similarity scoring

📌 Summary

This repository demonstrates:

Retrieval-Augmented Generation

Multi-document reasoning

OCR integration

Local LLM inference

Cross-platform compatibility

Production-style setup

It simulates a private ChatGPT-like system that runs entirely on your machine.