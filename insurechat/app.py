"""
InsureChat - main Gradio app (minimal runnable scaffold)
- Local RAG with llama-cpp GGUF LLMs, FAISS persistence
- Lightweight fallbacks for OCR (pytesseract) and PDF (pymupdf/pypdf)

Run after setup:
python app.py
"""

import os
import json
import threading
from pathlib import Path

import gradio as gr
import io

# LLM + embeddings + vectorstore imports are optional until models are present
try:
    from llama_cpp import Llama
except Exception:
    Llama = None

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:
    SentenceTransformer = None

try:
    import faiss
except Exception:
    faiss = None

# PDF and OCR
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    import pytesseract
    from PIL import Image
except Exception:
    pytesseract = None
    Image = None

try:
    from transformers import pipeline
except Exception:
    pipeline = None

# Basic paths
ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
FAISS_DIR = ROOT / "faiss_index"
KB_DIR = ROOT / "knowledge_base"
FAISS_DIR.mkdir(exist_ok=True)
KB_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# System prompt
SYSTEM_PROMPT = (
    "You are InsureChat, a US medical insurance assistant specializing in UnitedHealthcare.\n"
    "Answer from DOCUMENT CONTEXT when available and cite the source. If no context, say so."
)

# RAG index: uses FAISS + sentence-transformers when available, otherwise TF-IDF fallback
class RAGIndex:
    def __init__(self, index_dir: Path, embed_model_name: str = 'all-MiniLM-L6-v2'):
        self.index_dir = Path(index_dir)
        self.embed_model_name = embed_model_name
        self.texts = []
        self.meta = []
        self.embedding_model = None
        self.faiss_index = None
        self.dimension = None
        self.use_faiss = faiss is not None

        if SentenceTransformer is not None:
            try:
                self.embedding_model = SentenceTransformer(self.embed_model_name)
                self.dimension = self.embedding_model.get_sentence_embedding_dimension()
            except Exception:
                self.embedding_model = None

        self._load()

    def _meta_path(self):
        return self.index_dir / 'meta.json'

    def _faiss_path(self):
        return self.index_dir / 'index.faiss'

    def _load(self):
        if self._meta_path().exists():
            with open(self._meta_path(), 'r', encoding='utf8') as f:
                data = json.load(f)
            self.texts = [d['text'] for d in data.get('docs', [])]
            self.meta = [d.get('meta', {}) for d in data.get('docs', [])]

        if self.use_faiss and self._faiss_path().exists() and self.dimension:
            try:
                self.faiss_index = faiss.read_index(str(self._faiss_path()))
            except Exception:
                self.faiss_index = None

    def save(self):
        self.index_dir.mkdir(parents=True, exist_ok=True)
        data = {'docs': [{'text': t, 'meta': m} for t, m in zip(self.texts, self.meta)]}
        with open(self._meta_path(), 'w', encoding='utf8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        if self.use_faiss and self.faiss_index is not None:
            faiss.write_index(self.faiss_index, str(self._faiss_path()))

    def _embed(self, texts):
        if self.embedding_model is not None:
            return self.embedding_model.encode(texts, convert_to_numpy=True)
        # fallback: TF-IDF vectors (dense) via scikit-learn
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vec = TfidfVectorizer().fit(self.texts + list(texts))
            return vec.transform(texts).toarray()
        except Exception:
            # last resort: random vectors (deterministic hash)
            import numpy as _np
            vecs = []
            for t in texts:
                h = abs(hash(t)) % (10 ** 8)
                _np.random.seed(h)
                vecs.append(_np.random.rand(256))
            return _np.vstack(vecs)

    def add_documents(self, texts, metas=None):
        metas = metas or [{}] * len(texts)
        self.texts.extend(texts)
        self.meta.extend(metas)

        # build or update FAISS index if possible
        if self.embedding_model is not None and self.use_faiss:
            vecs = self._embed(texts)
            if self.faiss_index is None:
                import numpy as _np
                self.dimension = vecs.shape[1]
                self.faiss_index = faiss.IndexFlatIP(self.dimension)
                self.faiss_index.add(vecs)
            else:
                self.faiss_index.add(vecs)

        self.save()

    def query(self, q, topk=6):
        if self.embedding_model is not None and self.use_faiss and self.faiss_index is not None:
            qv = self._embed([q])
            D, I = self.faiss_index.search(qv, topk)
            results = []
            for idx in I[0]:
                if idx < len(self.texts):
                    results.append(self.texts[idx])
            return results
        # fallback substring match
        qq = q.lower()
        hits = [t for t in self.texts if qq in t.lower()]
        return hits[:topk]

index = RAGIndex(FAISS_DIR)

# If index is empty, seed it from knowledge_base/*.md so chat works without uploads
def seed_from_kb():
    if index.texts:
        return
    files = list(KB_DIR.glob('*.md'))
    for f in files:
        try:
            txt = f.read_text(encoding='utf8')
            chunks = chunk_text(txt)
            metas = [{'source': str(f), 'chunk': i} for i in range(len(chunks))]
            index.add_documents(chunks, metas=metas)
        except Exception:
            pass

seed_from_kb()
print(f"[insurechat] RAG index loaded {len(index.texts)} chunks from knowledge_base or previous saves")

# Utilities

def extract_text_from_pdf(path):
    if PdfReader is None:
        return ""
    reader = PdfReader(path)
    pages = []
    for p in reader.pages:
        pages.append(p.extract_text() or "")
    return "\n\n".join(pages)


def ocr_image(path):
    if pytesseract is None or Image is None:
        return ""
    try:
        # path may be a file path, a file-like object, or raw bytes
        if hasattr(path, 'read'):
            data = path.read()
            img = Image.open(io.BytesIO(data))
        else:
            img = Image.open(path)
        return pytesseract.image_to_string(img)
    except Exception:
        return ""

# Gradio handlers

def chunk_text(text, chunk_size=900, overlap=100):
    out = []
    start = 0
    L = len(text)
    while start < L:
        end = min(L, start + chunk_size)
        out.append(text[start:end])
        start = end - overlap if end - overlap > start else end
    return out


def ingest_file(file):
    # Resolve input: can be Gradio dict, file-like, or path string
    path = None
    fileobj = None
    if isinstance(file, dict):
        path = file.get('tmp_path') or file.get('name') or file.get('file')
        fileobj = file.get('file')
    else:
        # file may be an UploadedFile object with .name and .file
        path = getattr(file, 'name', None) or file
        fileobj = getattr(file, 'file', None)

    if not path and not fileobj:
        return "Could not resolve file path"

    text = ''
    # determine by filename if available
    fname = str(path) if path else ''
    if fname.lower().endswith(('.pdf')):
        text = extract_text_from_pdf(path)
    elif fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')) or fileobj is not None:
        # prefer fileobj if available
        target = fileobj if fileobj is not None else path
        text = ocr_image(target)
    else:
        try:
            # try reading as text
            with open(path, 'r', encoding='utf8') as f:
                text = f.read()
        except Exception:
            text = ''

    if not text:
        return "No text extracted from the provided file/image"

    chunks = chunk_text(text)
    metas = [{'source': path, 'chunk': i} for i in range(len(chunks))]
    index.add_documents(chunks, metas=metas)
    return f"Ingested {path} (chunks={len(chunks)})"


def ask_question(q, lang='en'):
    # Retrieve
    # If user asked in non-English, translate question to English for retrieval
    q_for_search = q
    try:
        if lang and lang.lower() != 'en':
            # translate question FROM lang -> en using transformers fallback if available
            if pipeline is not None:
                model_map_to_en = {
                    'es': 'Helsinki-NLP/opus-mt-es-en', 'fr': 'Helsinki-NLP/opus-mt-fr-en',
                    'hi': 'Helsinki-NLP/opus-mt-hi-en', 'de': 'Helsinki-NLP/opus-mt-de-en',
                    'pt': 'Helsinki-NLP/opus-mt-pt-en', 'ru': 'Helsinki-NLP/opus-mt-ru-en',
                    'ja': 'Helsinki-NLP/opus-mt-ja-en', 'ko': 'Helsinki-NLP/opus-mt-ko-en',
                    'zh': 'Helsinki-NLP/opus-mt-zh-en', 'ar': 'Helsinki-NLP/opus-mt-ar-en',
                    'vi': 'Helsinki-NLP/opus-mt-vi-en',
                }
                m = model_map_to_en.get(lang.lower())
                if m:
                    try:
                        trans = pipeline('translation', model=m)
                        out = trans(q[:4000])
                        if isinstance(out, list) and out:
                            q_for_search = out[0].get('translation_text', q)
                    except Exception:
                        q_for_search = q
            else:
                q_for_search = q
    except Exception:
        q_for_search = q

    chunks = index.query(q_for_search, topk=6)
    context = "\n\n---\n\n".join(chunks)
    prompt = f"{SYSTEM_PROMPT}\n\nDOCUMENT CONTEXT:\n{context}\n\nQuestion: {q}\nAnswer:"

    if Llama is None:
        # fallback: return retrieved context as answer (translated back if needed)
        answer = context[:1500] or "(no context)"
        if lang and lang.lower() != 'en':
            try:
                return translate_text(answer, target_lang=lang)
            except Exception:
                return answer
        return answer

    # find a GGUF model in models dir
    ggufs = list(MODELS_DIR.glob('*.gguf'))
    if not ggufs:
        return "No GGUF Llama model found in models/. Run download_models.py"

    # Prefer the primary LLM (llama3-8b) if present
    ggufs_sorted = sorted(ggufs, key=lambda p: 'llama' not in p.name)
    llm_path = str(ggufs_sorted[0])
    llm = Llama(model=llm_path, n_threads=max(1, os.cpu_count() // 2))

    # Keep prompts compact by limiting context length
    resp = llm(prompt=prompt, max_tokens=512)
    text = resp.get('choices', [{}])[0].get('text') or resp.get('text') or ''
    # If multilingual output requested and translator model available, translate back to requested language
    if lang and lang.lower() != 'en':
        try:
            return translate_text(text, target_lang=lang)
        except Exception:
            return text
    return text


def _find_gguf_by_keyword(keyword):
    ggufs = list(MODELS_DIR.glob('*.gguf'))
    for p in ggufs:
        if keyword.lower() in p.name.lower():
            return str(p)
    return None


def translate_text(text, target_lang='en'):
    # Prefer aya-expanse GGUF via llama-cpp
    if Llama is not None:
        aya_path = _find_gguf_by_keyword('aya') or _find_gguf_by_keyword('expanse')
        if aya_path:
            llm = Llama(model=aya_path, n_threads=max(1, os.cpu_count() // 2))
            prompt = f"Translate the following text to {target_lang}:\n\n{text}"
            resp = llm(prompt=prompt, max_tokens=512)
            return resp.get('choices',[{}])[0].get('text') or resp.get('text') or text

    # Transformers Helsinki-NLP fallback (best-effort)
    if pipeline is not None:
        # mapping for en->XX and XX->en
        model_map_en_to = {
            'es': 'Helsinki-NLP/opus-mt-en-es', 'fr': 'Helsinki-NLP/opus-mt-en-fr',
            'hi': 'Helsinki-NLP/opus-mt-en-hi', 'de': 'Helsinki-NLP/opus-mt-en-de',
            'pt': 'Helsinki-NLP/opus-mt-en-pt', 'ru': 'Helsinki-NLP/opus-mt-en-ru',
            'ja': 'Helsinki-NLP/opus-mt-en-ja', 'ko': 'Helsinki-NLP/opus-mt-en-ko',
            'zh': 'Helsinki-NLP/opus-mt-en-zh', 'ar': 'Helsinki-NLP/opus-mt-en-ar',
            'vi': 'Helsinki-NLP/opus-mt-en-vi',
        }
        model_map_to_en = {k: v.replace('-en-', f'-{k}-en') for k, v in model_map_en_to.items()}

        # If target_lang == 'en' we need to translate FROM source->en (caller should pass source language)
        model_name = None
        if target_lang.lower() == 'en':
            # we expect caller to pass source text language via text content; try common models by guessing
            # Fallback: return text unchanged if no model
            return text
        else:
            # translate from en->target
            model_name = model_map_en_to.get(target_lang.lower())

        if model_name:
            try:
                trans = pipeline('translation', model=model_name)
                out = trans(text[:4000])
                if isinstance(out, list) and out:
                    return out[0].get('translation_text', text)
            except Exception:
                pass

    return text


def extract_structured(file):
    # Resolve path and extract text
    path = None
    if isinstance(file, dict):
        path = file.get('tmp_path') or file.get('name') or file.get('file')
    else:
        path = getattr(file, 'name', None) or file
    if not path:
        return json.dumps({'error': 'no file path'})

    text = ''
    if str(path).lower().endswith(('.pdf', '.PDF')):
        text = extract_text_from_pdf(path)
    else:
        text = ocr_image(path) or ''

    if not text:
        return json.dumps({'error': 'no text extracted'})

    # If Nemotron model available, ask it to produce structured JSON
    nemotron_path = _find_gguf_by_keyword('nemotron') or _find_gguf_by_keyword('nemotron-mini')
    schema = {
        'member_id': None,
        'patient_name': None,
        'provider_name': None,
        'service_date': None,
        'billed_amount': None,
        'allowed_amount': None,
        'patient_owed': None,
        'CPT_codes': [],
    }

    if Llama is not None and nemotron_path:
        llm = Llama(model=nemotron_path, n_threads=max(1, os.cpu_count() // 2))
        prompt = (
            "Extract the following fields from the document and return valid JSON only. "
            "Fields: member_id, patient_name, provider_name, service_date, billed_amount, allowed_amount, patient_owed, CPT_codes (list).\n\n"
            f"Document text:\n{text[:20000]}\n\nReturn only JSON."
        )
        resp = llm(prompt=prompt, max_tokens=512)
        out = resp.get('choices',[{}])[0].get('text') or resp.get('text') or ''
        # try to parse JSON from output
        try:
            parsed = json.loads(out)
            return json.dumps(parsed, ensure_ascii=False, indent=2)
        except Exception:
            # fallthrough to regex fallback
            pass

    # Regex fallback: best-effort extraction
    import re
    res = dict(schema)
    # amounts
    amounts = re.findall(r"\$?\s*(\d{1,3}(?:[,\d{3}]*)(?:\.\d{2})?)", text)
    if amounts:
        # heuristics: billed, allowed, owed -> pick up to first three
        def clean(a):
            return a.replace(',', '')
        cleaned = [clean(a) for a in amounts]
        if len(cleaned) >= 1:
            res['billed_amount'] = cleaned[0]
        if len(cleaned) >= 2:
            res['allowed_amount'] = cleaned[1]
        if len(cleaned) >= 3:
            res['patient_owed'] = cleaned[2]

    # CPT codes: 5-digit numbers
    cpts = re.findall(r"\b(\d{5})\b", text)
    res['CPT_codes'] = list(dict.fromkeys(cpts))

    # member id heuristic
    mem = re.search(r"Member(?: ID| #|:)?\s*([A-Z0-9\-]{5,})", text, re.I)
    if mem:
        res['member_id'] = mem.group(1)

    # dates
    dt = re.search(r"(\d{1,2}/\d{1,2}/\d{2,4})", text)
    if dt:
        res['service_date'] = dt.group(1)

    return json.dumps(res, ensure_ascii=False, indent=2)

with gr.Blocks() as demo:
    gr.Markdown("# InsureChat — Local RAG for US medical insurance (UHC)")
    with gr.Row():
        with gr.Column(scale=2):
            chat = gr.Chatbot()
            inp = gr.Textbox(placeholder='Ask a question about your bill or coverage')
            submit = gr.Button('Ask')
        with gr.Column(scale=1):
            file_input = gr.File(label='Upload PDF/JPG/PNG/TXT')
            ingest_btn = gr.Button('Ingest file')
            status = gr.Textbox(label='Status')

            extract_btn = gr.Button('Extract Bill (JSON)')
            extract_output = gr.Textbox(label='Structured JSON', lines=10)

            lang_dropdown = gr.Dropdown(choices=[
                ("English","en"),("Spanish","es"),("French","fr"),("Hindi","hi"),
                ("Chinese","zh"),("Arabic","ar"),("Portuguese","pt"),("German","de"),
                ("Japanese","ja"),("Korean","ko"),("Russian","ru"),("Vietnamese","vi")
            ], value='en', label='Language')
            translate_input = gr.Textbox(label='Text to translate')
            translate_btn = gr.Button('Translate')
            translate_output = gr.Textbox(label='Translation', lines=4)

    ingest_btn.click(ingest_file, inputs=file_input, outputs=status)
    extract_btn.click(extract_structured, inputs=file_input, outputs=extract_output)

    # Chat handler: accepts question and language, returns updated chat history
    def chat_submit(question, lang, chat_history=None):
        if chat_history is None:
            chat_history = []
        # Normalize chat_history into list of {'role':..., 'content':...}
        normalized = []
        for item in chat_history:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                role = str(item[0]).lower()
                # map display roles to 'user'/'assistant'
                if 'user' in role:
                    r = 'user'
                elif 'assistant' in role or 'bot' in role:
                    r = 'assistant'
                else:
                    r = 'user'
                normalized.append({'role': r, 'content': str(item[1])})
            elif isinstance(item, dict):
                if 'role' in item and 'content' in item:
                    normalized.append({'role': item['role'], 'content': item['content']})
                else:
                    # try common keys
                    if 'user' in item and 'assistant' in item:
                        normalized.append({'role': 'user', 'content': str(item.get('user'))})
                    else:
                        normalized.append({'role': 'user', 'content': str(item)})
            else:
                normalized.append({'role': 'user', 'content': str(item)})

        # Add user message
        normalized.append({'role': 'user', 'content': question})

        # Produce answer
        ans = ask_question(question, lang=lang)

        # Translate answer if needed (ask_question may already translate, but ensure)
        try:
            if lang and lang.lower() != 'en':
                ans = translate_text(ans, target_lang=lang)
        except Exception:
            pass

        normalized.append({'role': 'assistant', 'content': ans})
        return normalized

    submit.click(chat_submit, inputs=[inp, lang_dropdown, chat], outputs=chat)
    translate_btn.click(lambda txt, lg: translate_text(txt, lg), inputs=[translate_input, lang_dropdown], outputs=translate_output)

if __name__ == '__main__':
    demo.launch()
