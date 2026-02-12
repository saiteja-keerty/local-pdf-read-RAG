 Local PDF (Fully Offline)
 Local PDF + Image RAG (Fully Offline)
📌 Overview

This project is a fully local Retrieval-Augmented Generation (RAG) system.

It allows you to:

📄 Upload a PDF and ask questions
🖼 Upload an image (bill, invoice, rulebook page) and extract text via OCR
📄 + 🖼 Upload both and ask combined questions
🤖 Get answers using a local LLM (LLaMA 3 via Ollama)

⚠ No OpenAI API.
⚠ No cloud calls.
⚠ Your data stays on your machine.

🗂 Files in This Repo
single_pdf_input.py

Upload a PDF

Ask questions about that PDF only

image_pdf_input.py

Upload PDF(s)

Upload Image(s)

Extract text using OCR

Ask cross-document questions

Example:
Upload solar quote PDF + electricity bill image
Ask: “Is installing solar profitable?”

🛠 Tools Used

LLM: Ollama (LLaMA 3)

Framework: LangChain

Vector DB: FAISS

Embeddings: Sentence Transformers

OCR: Tesseract

UI: Gradio

Language: Python

⚙️ How It Works (Simple)

Extract text from PDF

Extract text from image (OCR)

Split into chunks

Convert to embeddings

Store in FAISS

Retrieve relevant chunks

Send context to local LLM

Generate grounded answer

💻 Requirements

You must install:
Python 3.10+

Ollama

Tesseract (for image support)

🔹 Install Ollama

Download:
https://ollama.com

Then run:
ollama pull llama3

🔹 Install Tesseract
Windows:
Download installer and check Add to PATH

https://github.com/UB-Mannheim/tesseract/wiki

Mac:
brew install tesseract

Linux:
sudo apt install tesseract-ocr

📦 Setup
Clone repo:

git clone https://github.com/saiteja-keerty/local-pdf-read-RAG.git
cd local-pdf-read-RAG


Create virtual environment:
python -m venv venv


Activate:
Windows

venv\Scripts\activate


Mac/Linux

source venv/bin/activate


Install dependencies:

pip install -r requirements.txt

▶ Run
PDF Only:
python single_pdf_input.py

PDF + Image:
python image_pdf_input.py


Open:

http://127.0.0.1:7860

-  Quick Mac Setup (All-in-One)
brew install python
brew install ollama
brew install tesseract

ollama pull llama3

git clone https://github.com/saiteja-keerty/local-pdf-read-RAG.git
cd local-pdf-read-RAG

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python image_pdf_input.py

⚠ Known Limitations

First run downloads embedding model (~90MB)

Ollama model is large (GBs)

OCR depends on image clarity

Retrieval may miss relevant chunks

LLM reasoning is approximate (not a calculator)

🔒 Privacy

Fully offline

No API keys

No external services

Data never leaves your machine

🎯 Example Questions

“Summarize this document.”

“What is the total installation cost?”

“Is solar installation profitable?”

“Explain rule 4.2 from the rulebook.”

📌 Summary

This repo demonstrates:

Retrieval-Augmented Generation

Multi-document reasoning

OCR integration

Local LLM usage

Cross-platform setup

It simulates a private ChatGPT-like system running entirely on your machine.