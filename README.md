---
title: InsureChat
emoji: "\U0001F6E1\uFE0F"
colorFrom: blue
colorTo: green
sdk: gradio
app_file: insurechat/app.py
pinned: false
tags:
  - healthcare
  - local-ai
  - rag
  - privacy
  - multilingual
  - track-local-ai
  - track-healthcare
  - badge-open-source
  - badge-multilingual
  - badge-privacy-first
---

# InsureChat

InsureChat is a local Gradio assistant for U.S. medical insurance definitions, medical bills, EOBs, and claims. It combines a local knowledge base, OCR, FAISS retrieval, and optional GGUF language models through `llama-cpp-python`.

No OpenAI API or cloud inference service is used. After setup and model downloads, document processing and inference run locally.

## Hackathon Submission

- **Social media post:** [Link will be added after publishing](SOCIAL_MEDIA_POST_URL)
- **Demo video:** [Link will be added after recording](DEMO_VIDEO_URL)

### The Idea

Medical bills and U.S. insurance terms can be difficult to understand, especially for newcomers and people who prefer another language. InsureChat is a privacy-first assistant that explains common insurance concepts before a document is uploaded, then switches to answers grounded in the user's active bill, EOB, claim, image, or text file after it is processed.

### How It Was Built

The app uses Gradio for the interface and a local retrieval-augmented generation pipeline for question answering. It extracts text from PDFs and text files, applies Tesseract OCR to images and scanned PDFs, retrieves relevant context with FAISS or a lexical fallback, and parses important bill fields such as billed amount, discount, plan payment, service date, and patient responsibility. Uploaded documents remain session-specific so an earlier bill cannot influence a new session.

### Technology

- Python and Gradio
- `llama-cpp-python` with local GGUF models
- Aya Expanse 8B for multilingual translation
- Llama 3.1 8B for optional reasoning and structured extraction
- FAISS and Sentence Transformers for retrieval
- Tesseract, PyMuPDF, Pillow, and pypdf for OCR and document processing
- Local Markdown and text files for trusted insurance definitions

## Features

- Answers insurance definitions from local Markdown, text, and PDF knowledge files.
- OCRs JPG, PNG, TIFF, and scanned PDF medical bills.
- Extracts billed amount, discounts, patient balance, service date, and related fields.
- Uses Aya Expanse 8B for local multilingual translation.
- Uses Llama 3.1 8B for optional reasoning and structured extraction.
- Keeps only one GGUF model loaded at a time to reduce peak RAM and VRAM use.
- Continues with deterministic RAG and regex fallbacks when optional GGUF models are missing.

## Requirements

- Python 3.10 or newer
- Approximately 6 GB of free space for Aya alone
- Approximately 12-14 GB for all three Q4 GGUF models
- Tesseract OCR for JPG, PNG, TIFF, and scanned PDF processing
- Enough system RAM to load the selected model; models are loaded by role and are not all required for every request

### Install Tesseract

Windows:

1. Install Tesseract from [UB Mannheim's Windows builds](https://github.com/UB-Mannheim/tesseract/wiki).
2. Enable the option to add Tesseract to `PATH`, or install it in `C:\Program Files\Tesseract-OCR`.

macOS:

```bash
brew install tesseract
```

Ubuntu/Debian:

```bash
sudo apt install tesseract-ocr
```

## Setup

Create and activate a virtual environment:

```powershell
python -m venv venv
venv\Scripts\activate
```

On macOS/Linux, activate it with:

```bash
source venv/bin/activate
```

Install the InsureChat dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r insurechat/requirements.txt
```

The requirements install Python packages only. They do not install Tesseract or download multi-gigabyte GGUF model files.

## Download Models

Models are intentionally excluded from Git and are not downloaded automatically. Each user must download the models needed on their own computer.

Translation model:

```powershell
python insurechat/download_models.py --model aya-8b
```

Optional alternative reasoning model:

```powershell
python insurechat/download_models.py --model nemotron-4b
```

Recommended document extraction and reasoning model:

```powershell
python insurechat/download_models.py --model llama3-8b
```

The files are stored in `insurechat/models/`:

| Model | Role | Approximate Q4 size |
| --- | --- | --- |
| Aya Expanse 8B | Translation | 4.7 GB |
| Nemotron Mini 4B | Reasoning | 2-3 GB |
| Llama 3.1 8B | Extraction/reasoning fallback | 4.5-5 GB |

The app selects models by role and keeps only one GGUF loaded at a time with a 4,096-token context window. Switching between Llama reasoning and Aya translation unloads the current model before loading the next one. This lowers peak memory use, but the first request after a role switch can take longer while the next model loads.

Performance depends heavily on hardware. On integrated graphics or CPU-only systems, an 8B response or model switch may take from several seconds to a few minutes. A discrete GPU with sufficient VRAM will generally respond faster.

For the recommended setup, install Aya and Llama 3.1. Nemotron is an optional lighter English reasoning alternative and is not required when Llama is installed.

Without the models, the application still starts:

- Local glossary definitions continue to work.
- OCR and regex medical-bill extraction continue to work.
- Translation is unavailable without Aya.
- LLM reasoning/extraction is unavailable without Nemotron or Llama.

## Run

From the repository root:

```powershell
python insurechat/app.py
```

Open the URL printed by Gradio, normally <http://127.0.0.1:7860>.

## Troubleshooting

### Translation unavailable

Confirm that `insurechat/models/aya-expanse-8b-Q4_K_M.gguf` exists, then restart the app.

### No text extracted from an image

Confirm both components are installed:

```powershell
python -c "import pytesseract; print(pytesseract.get_tesseract_version())"
```

If that fails, install `pytesseract` from the requirements and install the separate Tesseract application.

### Scanned PDF has no text

Install `pymupdf` and `pytesseract`. InsureChat renders scanned PDF pages locally and sends the rendered images to Tesseract OCR.

### llama-cpp tries to compile on Windows

`insurechat/requirements.txt` includes the official prebuilt CPU wheel index. A native compiler should not be required for supported Python versions.

## Privacy

Uploaded documents, OCR text, retrieval, and GGUF inference remain on the local computer. Internet access is only needed during dependency and model installation, or when the app displays an external reference link for a definition not found locally.
