"""
Simple interactive downloader for GGUF models using huggingface_hub
This script is intentionally minimal: it downloads selected GGUF files into models/.
Run once during setup. Requires internet for downloads.
"""
import argparse
from huggingface_hub import hf_hub_download
from pathlib import Path

MODELS = {
    'llama3-8b': {'repo':'bartowski/Meta-Llama-3.1-8B-Instruct-GGUF','filename':'Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf'},
    'nemotron-4b': {'repo':'bartowski/Nemotron-Mini-4B-Instruct-GGUF','filename':'Nemotron-Mini-4B-Instruct-Q4_K_M.gguf'},
    'aya-8b': {'repo':'bartowski/aya-expanse-8b-GGUF','filename':'aya-expanse-8b-Q4_K_M.gguf'},
    'moondream2': {'repo':'vikhyatk/moondream2','filename':'moondream2-text-model-f16.gguf'},
    'meta-llama-8b-quant': {'repo':'bartowski/Meta-Llama-3.1-8B-Instruct-GGUF','filename':'Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf'},
}

OUT = Path(__file__).parent / 'models'
OUT.mkdir(exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=list(MODELS.keys()), help='Model to download')
args = parser.parse_args()

if args.model:
    m = MODELS[args.model]
    print('Downloading', args.model)
    p = hf_hub_download(repo_id=m['repo'], filename=m['filename'], local_dir=OUT)
    dest = Path(p)
    print('Saved to', dest)
else:
    print('Available models:')
    for k in MODELS:
        print('-', k)
    print('\nRun with --model <name> to download')
