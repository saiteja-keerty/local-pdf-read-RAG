# InsureChat

InsureChat is a local Gradio assistant for U.S. medical insurance definitions, medical bills, EOBs, and claims. It combines a local knowledge base, OCR, FAISS retrieval, and optional GGUF language models through `llama-cpp-python`.

No OpenAI API or cloud inference service is used. After setup and model downloads, document processing and inference run locally.

## Features

- Answers insurance definitions from local Markdown, text, and PDF knowledge files.
- OCRs JPG, PNG, TIFF, and scanned PDF medical bills.
- Extracts billed amount, discounts, patient balance, service date, and related fields.
- Uses Aya Expanse 8B for local multilingual translation.
- Supports optional Nemotron Mini 4B and Llama 3.1 8B reasoning models.
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

Reasoning model:

```powershell
python insurechat/download_models.py --model nemotron-4b
```

Document extraction and reasoning fallback:

```powershell
python insurechat/download_models.py --model llama3-8b
```

The files are stored in `insurechat/models/`:

| Model | Role | Approximate Q4 size |
| --- | --- | --- |
| Aya Expanse 8B | Translation | 4.7 GB |
| Nemotron Mini 4B | Reasoning | 2-3 GB |
| Llama 3.1 8B | Extraction/reasoning fallback | 4.5-5 GB |

The combined parameter count is approximately 20B. The app selects models by role; it does not combine them into one 20B model or load all three for every answer.

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
