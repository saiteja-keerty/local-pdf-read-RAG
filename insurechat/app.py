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
import re
import io

# LLM + embeddings + vectorstore imports are optional until models are present
try:
    from llama_cpp import Llama
    from llama_cpp import llama_cpp as _llama_backend
    # Initialize llama.cpp before Torch/FAISS load their native runtimes. On
    # Windows, doing this in the opposite order can crash backend_init().
    _llama_backend.llama_backend_init()
except Exception:
    Llama = None

import gradio as gr

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
    "You are InsureChat, a local medical insurance assistant for people who may be new to U.S. health insurance.\n"
    "Explain terms in plain language, define acronyms, and avoid assuming the user knows words like copay, deductible, EOB, claim, or allowed amount.\n"
    "Use DOCUMENT CONTEXT first. If context comes from an uploaded bill, EOB, claim, or plan document, cite the source and explain any math step by step.\n"
    "Never include local file paths in the answer. If context is missing, say what information is needed instead of guessing. Do not provide legal, tax, or medical advice.\n"
    "For definitions, answer briefly with a simple example and source when available."
)

MODEL_ROLE_KEYWORDS = {
    "extract": ("llama-3.1", "llama3.1", "meta-llama", "llama", "parse", "vl", "vision", "ocr", "asr"),
    "reason": ("nemotron", "llama-3.1", "llama3.1", "meta-llama", "llama"),
    "translate": ("aya", "tiny-aya", "expanse", "multilingual"),
    "embed": ("embed", "embedding"),
}

LANGUAGE_NAMES = {
    'en': 'English', 'es': 'Spanish', 'fr': 'French', 'hi': 'Hindi',
    'zh': 'Chinese', 'ar': 'Arabic', 'pt': 'Portuguese', 'de': 'German',
    'ja': 'Japanese', 'ko': 'Korean', 'ru': 'Russian', 'vi': 'Vietnamese',
}

ONLINE_GLOSSARY_URL = "https://www.healthcare.gov/glossary/"

_LLM_CACHE = {}
_LLM_CACHE_LOCK = threading.Lock()

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
        # Offline lexical fallback. Whole-question substring matching misses
        # queries such as "what is a deductible?".
        stopwords = {'a', 'an', 'and', 'define', 'definition', 'for', 'is', 'meaning',
                     'of', 'please', 'the', 'to', 'what', 'whats'}
        query_terms = set(re.findall(r"[a-z0-9]+", str(q).lower())) - stopwords
        scored = []
        for text in self.texts:
            text_terms = set(re.findall(r"[a-z0-9]+", text.lower()))
            score = len(query_terms & text_terms)
            if score:
                scored.append((score, text))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [text for _, text in scored[:topk]]

index = RAGIndex(FAISS_DIR)

# Seed local Markdown/text/PDF knowledge files. When a .txt and .pdf share a
# stem, prefer the text copy to avoid indexing the same content twice.
def seed_from_kb():
    selected = {}
    suffix_priority = {'.txt': 0, '.md': 1, '.pdf': 2}
    for f in KB_DIR.iterdir():
        suffix = f.suffix.lower()
        if suffix not in suffix_priority:
            continue
        current = selected.get(f.stem.lower())
        if current is None or suffix_priority[suffix] < suffix_priority[current.suffix.lower()]:
            selected[f.stem.lower()] = f

    files = list(selected.values())
    indexed_sources = set()
    for meta in index.meta:
        if isinstance(meta, dict) and meta.get('source'):
            indexed_sources.add(str(meta.get('source')))

    for f in files:
        source = str(f)
        if source in indexed_sources:
            continue
        try:
            txt = extract_text_from_pdf(f) if f.suffix.lower() == '.pdf' else f.read_text(encoding='utf8')
            chunks = chunk_text(txt)
            metas = [{'source': source, 'chunk': i, 'kind': 'knowledge_base'} for i in range(len(chunks))]
            index.add_documents(chunks, metas=metas)
        except Exception:
            pass

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


seed_from_kb()
print(f"[insurechat] RAG index loaded {len(index.texts)} chunks from knowledge_base or previous saves")


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


def _search_kb_for_terms(query, limit=4):
    if not isinstance(query, str) or not query.strip():
        return []

    normalized = query.lower()
    aliases = {
        'copay': ['copay', 'copayment', 'co-pay'],
        'deductible': ['deductible'],
        'coinsurance': ['coinsurance', 'co-insurance'],
        'out-of-pocket maximum': ['out-of-pocket', 'oop', 'maximum'],
        'allowed amount': ['allowed amount', 'allowable', 'eligible expense', 'negotiated rate'],
        'eob': ['eob', 'explanation of benefits'],
        'claim': ['claim'],
        'premium': ['premium'],
        'in-network': ['in-network', 'network'],
        'out-of-network': ['out-of-network', 'non-participating'],
        'balance billing': ['balance billing', 'balance bill'],
    }

    wanted = set()
    for canonical, terms in aliases.items():
        if any(term in normalized for term in terms):
            wanted.add(canonical)

    if not wanted and re.search(r"\b(insurance|bill|claim|deduct|copay|coverage|medical term|meaning|define)\b", normalized):
        wanted.update(['deductible', 'copay', 'coinsurance', 'allowed amount', 'eob'])

    hits = []
    for f in list(KB_DIR.glob('*.md')) + list(KB_DIR.glob('*.txt')):
        try:
            text = f.read_text(encoding='utf8')
        except Exception:
            continue
        lines = text.splitlines()
        for line in lines:
            line_lower = line.lower()
            if any(term in line_lower for term in wanted):
                cleaned = line.strip()
                if not (cleaned.startswith('|') or cleaned.startswith('- ')):
                    continue
                if cleaned.startswith('|---'):
                    continue
                if cleaned and cleaned not in hits:
                    hits.append(f"Source: {f}\n{cleaned}")
                    if len(hits) >= limit:
                        return hits

        # Plain-text glossary entries use an uppercase heading followed by a
        # definition and optional example/note paragraphs.
        query_terms = set(re.findall(r"[a-z0-9]+", normalized)) - {
            'a', 'an', 'define', 'definition', 'is', 'meaning', 'of', 'please', 'the', 'what', 'whats'
        }
        for block in re.split(r"\n\s*\n", text):
            block_lines = [line.strip() for line in block.splitlines() if line.strip()]
            if len(block_lines) < 2:
                continue
            heading = block_lines[0]
            heading_terms = set(re.findall(r"[a-z0-9]+", heading.lower()))
            if query_terms and query_terms <= heading_terms and heading.upper() == heading:
                hits.append(f"Source: {f}\n{block.strip()}")
                if len(hits) >= limit:
                    return hits
    return hits


def _clean_source_label(source):
    if not source:
        return ""
    try:
        name = Path(str(source)).name
    except Exception:
        name = str(source)
    if name == "insurance_glossary.md":
        return "insurance glossary"
    if name == "medical_insurance_terms.md":
        return "medical insurance terms"
    return name


def _parse_term_row(text):
    line = (text or "").strip()
    if line.startswith("Source:"):
        line = line.split("\n", 1)[1].strip() if "\n" in line else ""
    if line.startswith("- ") and ":" in line:
        term, definition = line[2:].split(":", 1)
        return {
            "term": term.strip(),
            "definition": definition.strip(),
            "look_for": "",
            "example": "",
        }
    if line.startswith("|"):
        cells = [cell.strip().strip('"') for cell in line.strip("|").split("|")]
        if len(cells) >= 4 and cells[0].lower() != "term":
            return {
                "term": cells[0],
                "definition": cells[1],
                "look_for": cells[2],
                "example": cells[3],
            }
    plain_lines = [part.strip() for part in line.splitlines() if part.strip()]
    if len(plain_lines) >= 2 and plain_lines[0].upper() == plain_lines[0]:
        example = ""
        definition_parts = []
        for part in plain_lines[1:]:
            if part.lower().startswith('example:'):
                example = part.split(':', 1)[1].strip()
            elif not part.lower().startswith('note:'):
                definition_parts.append(part)
        return {
            "term": plain_lines[0],
            "definition": " ".join(definition_parts),
            "look_for": "",
            "example": example,
        }
    return None


def _simple_context_answer(question, chunks):
    source_labels = []
    for chunk in chunks:
        match = re.match(r"Source:\s*(.+)\n", str(chunk))
        if match:
            label = _clean_source_label(match.group(1))
            if label and label not in source_labels:
                source_labels.append(label)

    rows = []
    for chunk in chunks:
        row = _parse_term_row(chunk)
        if row and row not in rows:
            rows.append(row)

    if rows:
        primary = rows[0]
        answer = f"{primary['term']}: {primary['definition']}"
        if primary.get('example'):
            answer += f"\n\nExample: {primary['example']}"
        if primary.get('look_for'):
            look_for = primary['look_for'].replace('"', '').rstrip('.')
            answer += f"\n\nOn a bill or EOB, look for: {look_for}."
    else:
        cleaned = []
        for chunk in chunks[:3]:
            text = re.sub(r"Source:\s*.*\n", "", str(chunk)).strip()
            if text:
                cleaned.append(text)
        answer = "\n\n".join(cleaned) or "I do not have enough local context yet. Upload a bill/EOB/claim or ask about a common insurance term."

    if source_labels:
        answer += f"\n\nSource: {source_labels[0]}"
    return answer


def ask_question(q, lang='en'):
    # Retrieve
    # short-circuit greetings to avoid dumping documents for 'hi' etc.
    greetings = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}
    if isinstance(q, str) and q.strip().lower() in greetings:
        return "Hi — I can help with your bill or insurance questions. Ask me about a term (e.g. 'what is a copay') or upload a bill."
    # Try the local glossary before translating. This handles an English
    # question with a non-English answer language without an unnecessary model call.
    preliminary_kb_hits = _search_kb_for_terms(q)

    # Translate the question locally with Aya for English-language retrieval.
    q_for_search = q
    try:
        if lang and lang.lower() != 'en' and not preliminary_kb_hits:
            q_for_search = translate_text(q, target_lang='en', source_lang=lang)
    except Exception:
        q_for_search = q

    chunks = index.query(q_for_search, topk=6)
    # If no retrieved chunks and this looks like a definition request, try KB files directly
    import re
    is_definition = False
    try:
        if re.search(r"\b(what is|what's|define|definition of|meaning of)\b", (q or '').lower()):
            is_definition = True
        # also if user asked a single short token like 'copay'
        if not is_definition and isinstance(q, str) and len(q.split()) == 1 and len(q) < 30:
            is_definition = True
    except Exception:
        is_definition = False

    kb_hits = preliminary_kb_hits or _search_kb_for_terms(q_for_search)
    if kb_hits:
        chunks = kb_hits + [chunk for chunk in chunks if chunk not in kb_hits]
    elif is_definition:
        # A similarity hit is not enough evidence that the requested term has
        # a local definition.
        chunks = []
    # Build a compact context: include short excerpts and source metadata (if available)
    context_parts = []
    for chunk in chunks:
        try:
            idx = index.texts.index(chunk)
            meta = index.meta[idx] if idx < len(index.meta) else {}
            src = meta.get('source', '') if isinstance(meta, dict) else ''
        except Exception:
            src = ''
        excerpt = (chunk or '').strip()[:500]
        if src:
            context_parts.append(f"Source: {_clean_source_label(src)}\n{excerpt}")
        else:
            context_parts.append(excerpt)

    context = "\n\n---\n\n".join(context_parts)

    if not context and is_definition:
        return (
            "I could not find that definition in the local knowledge base. "
            "I have this trusted glossary reference that may contain what you need: "
            f"{ONLINE_GLOSSARY_URL}"
        )

    prompt_suffix = (
        "Answer concisely. Use plain language for a non-U.S. user. "
        "If amounts are involved, show the calculation steps and assumptions. Include Source when available."
    )
    prompt = (
        f"{SYSTEM_PROMPT}\n\nDOCUMENT CONTEXT:\n{context}\n\n"
        f"Question: {q}\n{prompt_suffix}"
    )

    if Llama is None:
        return _simple_context_answer(q, chunks)

    llm_path = _select_model_for_role("reason")
    if not llm_path:
        return _simple_context_answer(q, chunks)

    llm = _get_llm(llm_path)

    # Keep prompts compact by limiting context length
    resp = llm(prompt=prompt, max_tokens=512)
    text = resp.get('choices', [{}])[0].get('text') or resp.get('text') or ''
    # If multilingual output requested and translator model available, translate back to requested language
    # Always return English text from this function; translation is handled by the caller.
    return text


def _find_gguf_by_keyword(keyword):
    ggufs = list(MODELS_DIR.glob('*.gguf'))
    for p in ggufs:
        if keyword.lower() in p.name.lower():
            return str(p)
    return None


def _select_model_for_role(role):
    ggufs = list(MODELS_DIR.glob('*.gguf'))
    if not ggufs:
        return None

    keywords = MODEL_ROLE_KEYWORDS.get(role, ())
    for keyword in keywords:
        found = _find_gguf_by_keyword(keyword)
        if found:
            return found

    # Never substitute a translation model for reasoning (or vice versa).
    return None


def _get_llm(model_path):
    """Load each local GGUF once and reuse it for later requests."""
    if Llama is None or not model_path:
        return None
    with _LLM_CACHE_LOCK:
        if model_path not in _LLM_CACHE:
            _LLM_CACHE[model_path] = Llama(
                model_path=model_path,
                n_ctx=1024,
                n_threads=max(1, os.cpu_count() // 2),
                n_gpu_layers=-1,
                verbose=False,
            )
        return _LLM_CACHE[model_path]


def translate_text(text, target_lang='en', source_lang='English'):
    """Translate with the local Aya GGUF only; this function never uses the internet."""
    target_code = (target_lang or 'en').lower()
    if target_code == (source_lang or '').lower():
        return text

    if Llama is not None:
        aya_path = _select_model_for_role("translate")
        if aya_path:
            source_name = LANGUAGE_NAMES.get((source_lang or '').lower(), source_lang or 'the source language')
            target_name = LANGUAGE_NAMES.get(target_code, target_lang)
            llm = _get_llm(aya_path)
            resp = llm.create_chat_completion(
                messages=[
                    {
                        'role': 'system',
                        'content': (
                            f"Translate from {source_name} to {target_name}. Return only the translation, "
                            "without notes, labels, or the original text."
                        ),
                    },
                    {'role': 'user', 'content': text},
                ],
                max_tokens=min(384, max(96, len(text) * 2)),
                temperature=0.1,
            )
            translated = resp.get('choices', [{}])[0].get('message', {}).get('content', '')
            return translated.strip() or text

    return text


def _money_to_float(value):
    if value is None:
        return None
    try:
        return float(str(value).replace('$', '').replace(',', '').strip())
    except Exception:
        return None


def _format_money(value):
    if value is None:
        return None
    return f"${value:,.2f}"


def _extract_labeled_amount(text, labels):
    for label in labels:
        pattern = rf"{label}\s*(?:amount|charge|paid|due|owed|responsibility|:)?\s*\$?\s*([0-9][0-9,]*(?:\.[0-9]{{2}})?)"
        match = re.search(pattern, text, re.I)
        if match:
            return match.group(1).replace(',', '')
    return None


def estimate_patient_responsibility(fields):
    billed = _money_to_float(fields.get('billed_amount'))
    allowed = _money_to_float(fields.get('allowed_amount'))
    plan_paid = _money_to_float(fields.get('plan_paid'))
    deductible = _money_to_float(fields.get('deductible'))
    copay = _money_to_float(fields.get('copay'))
    coinsurance = _money_to_float(fields.get('coinsurance'))
    noncovered = _money_to_float(fields.get('noncovered_amount'))
    explicit_owed = _money_to_float(fields.get('patient_owed'))

    steps = []
    if explicit_owed is not None:
        steps.append(f"Document lists patient responsibility as {_format_money(explicit_owed)}.")
        return {
            'estimated_patient_responsibility': _format_money(explicit_owed),
            'calculation_steps': steps,
            'confidence': 'high',
        }

    total = 0.0
    if deductible:
        total += deductible
        steps.append(f"Deductible applied: {_format_money(deductible)}.")
    if copay:
        total += copay
        steps.append(f"Copay: {_format_money(copay)}.")
    if coinsurance:
        total += coinsurance
        steps.append(f"Coinsurance: {_format_money(coinsurance)}.")
    if noncovered:
        total += noncovered
        steps.append(f"Non-covered amount: {_format_money(noncovered)}.")

    if total:
        return {
            'estimated_patient_responsibility': _format_money(total),
            'calculation_steps': steps,
            'confidence': 'medium',
        }

    if allowed is not None and plan_paid is not None:
        estimate = max(0.0, allowed - plan_paid)
        steps.append(f"Allowed amount minus plan paid: {_format_money(allowed)} - {_format_money(plan_paid)}.")
        return {
            'estimated_patient_responsibility': _format_money(estimate),
            'calculation_steps': steps,
            'confidence': 'medium',
        }

    if billed is not None and plan_paid is not None:
        estimate = max(0.0, billed - plan_paid)
        steps.append(f"Billed amount minus plan paid: {_format_money(billed)} - {_format_money(plan_paid)}.")
        steps.append("This is less reliable because patient responsibility is usually based on allowed amount, not billed charge.")
        return {
            'estimated_patient_responsibility': _format_money(estimate),
            'calculation_steps': steps,
            'confidence': 'low',
        }

    return {
        'estimated_patient_responsibility': None,
        'calculation_steps': ['Not enough labeled amounts were found to estimate patient responsibility.'],
        'confidence': 'low',
    }


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

    schema = {
        'document_type': None,
        'member_id': None,
        'patient_name': None,
        'provider_name': None,
        'service_date': None,
        'billed_amount': None,
        'allowed_amount': None,
        'plan_paid': None,
        'adjustment_or_discount': None,
        'deductible': None,
        'copay': None,
        'coinsurance': None,
        'noncovered_amount': None,
        'patient_owed': None,
        'CPT_codes': [],
        'ICD10_codes': [],
        'warnings': [],
    }

    extract_model_path = _select_model_for_role("extract") or _select_model_for_role("reason")
    if Llama is not None and extract_model_path:
        llm = _get_llm(extract_model_path)
        prompt = (
            "Extract the following fields from the document and return valid JSON only. "
            "Fields: document_type, member_id, patient_name, provider_name, service_date, billed_amount, "
            "allowed_amount, plan_paid, adjustment_or_discount, deductible, copay, coinsurance, "
            "noncovered_amount, patient_owed, CPT_codes (list), ICD10_codes (list), warnings (list). "
            "Use null for missing fields. Preserve dollar amounts as strings.\n\n"
            f"Document text:\n{text[:20000]}\n\nReturn only JSON."
        )
        resp = llm(prompt=prompt, max_tokens=512)
        out = resp.get('choices',[{}])[0].get('text') or resp.get('text') or ''
        # try to parse JSON from output
        try:
            parsed = json.loads(out)
            if isinstance(parsed, dict):
                calc = estimate_patient_responsibility(parsed)
                parsed.update(calc)
            return json.dumps(parsed, ensure_ascii=False, indent=2)
        except Exception:
            # fallthrough to regex fallback
            pass

    # Regex fallback: best-effort extraction
    res = dict(schema)
    lower_text = text.lower()
    if 'explanation of benefits' in lower_text or 'this is not a bill' in lower_text:
        res['document_type'] = 'explanation_of_benefits'
    elif 'claim' in lower_text:
        res['document_type'] = 'claim'
    elif 'amount due' in lower_text or 'balance due' in lower_text:
        res['document_type'] = 'provider_bill'

    res['billed_amount'] = _extract_labeled_amount(text, [
        r'amount billed', r'billed', r'provider charge', r'total charges?', r'charge'
    ])
    res['allowed_amount'] = _extract_labeled_amount(text, [
        r'allowed amount', r'allowed', r'eligible expense', r'negotiated rate', r'allowable'
    ])
    res['plan_paid'] = _extract_labeled_amount(text, [
        r'plan paid', r'insurance paid', r'paid by plan', r'benefit paid', r'paid'
    ])
    res['adjustment_or_discount'] = _extract_labeled_amount(text, [
        r'adjustment', r'discount', r'network discount', r'provider discount'
    ])
    res['deductible'] = _extract_labeled_amount(text, [r'deductible'])
    res['copay'] = _extract_labeled_amount(text, [r'copay', r'co-pay', r'copayment'])
    res['coinsurance'] = _extract_labeled_amount(text, [r'coinsurance', r'co-insurance'])
    res['noncovered_amount'] = _extract_labeled_amount(text, [
        r'non-covered', r'not covered', r'noncovered'
    ])
    res['patient_owed'] = _extract_labeled_amount(text, [
        r'patient responsibility', r'member responsibility', r'you owe', r'amount due', r'balance due',
        r'patient owes?', r'total due'
    ])

    # amounts
    amounts = re.findall(r"\$?\s*(\d{1,3}(?:[,\d{3}]*)(?:\.\d{2})?)", text)
    if amounts:
        # heuristics: billed, allowed, owed -> pick up to first three
        def clean(a):
            return a.replace(',', '')
        cleaned = [clean(a) for a in amounts]
        if len(cleaned) >= 1 and not res['billed_amount']:
            res['billed_amount'] = cleaned[0]
        if len(cleaned) >= 2 and not res['allowed_amount']:
            res['allowed_amount'] = cleaned[1]
        if len(cleaned) >= 3 and not res['patient_owed']:
            res['patient_owed'] = cleaned[2]

    # CPT codes: 5-digit numbers
    cpts = re.findall(r"\b(\d{5})\b", text)
    res['CPT_codes'] = list(dict.fromkeys(cpts))
    icds = re.findall(r"\b([A-Z][0-9][0-9AB](?:\.[A-Z0-9]{1,4})?)\b", text)
    res['ICD10_codes'] = list(dict.fromkeys(icds))

    # member id heuristic
    mem = re.search(r"Member(?: ID| #|:)?\s*([A-Z0-9\-]{5,})", text, re.I)
    if mem:
        res['member_id'] = mem.group(1)

    # dates
    dt = re.search(r"(\d{1,2}/\d{1,2}/\d{2,4})", text)
    if dt:
        res['service_date'] = dt.group(1)

    res.update(estimate_patient_responsibility(res))
    if not res.get('allowed_amount'):
        res['warnings'].append('Allowed amount was not found; patient responsibility may be harder to verify.')
    if res.get('document_type') == 'provider_bill':
        res['warnings'].append('Compare this provider bill with the insurer EOB before assuming the amount is final.')

    return json.dumps(res, ensure_ascii=False, indent=2)

with gr.Blocks() as demo:
    gr.Markdown("# InsureChat - local medical insurance helper")
    with gr.Row():
        with gr.Column(scale=2):
            chat = gr.Chatbot()
            lang_dropdown = gr.Dropdown(choices=[
                ("English","en"),("Spanish","es"),("French","fr"),("Hindi","hi"),
                ("Chinese","zh"),("Arabic","ar"),("Portuguese","pt"),("German","de"),
                ("Japanese","ja"),("Korean","ko"),("Russian","ru"),("Vietnamese","vi")
            ], value='en', label='Answer language')
            inp = gr.Textbox(placeholder='Ask about copay, deductible, an EOB, a claim, or an uploaded bill')
            submit = gr.Button('Ask')
        with gr.Column(scale=1):
            file_input = gr.File(label='Upload PDF/JPG/PNG/TXT')
            ingest_btn = gr.Button('Ingest file')
            status = gr.Textbox(label='Status')

            extract_btn = gr.Button('Extract Bill / Claim (JSON)')
            extract_output = gr.Textbox(label='Structured insurance fields and estimate', lines=14)

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

        # Produce an English grounded answer, then translate it exactly once.
        ans = ask_question(question, lang=lang)

        # Return only the selected language instead of English plus a translation.
        try:
            if lang and lang.lower() != 'en':
                translated = translate_text(ans, target_lang=lang, source_lang='en')
                if translated and translated.strip() and translated.strip() != ans.strip():
                    ans = translated.strip()
                else:
                    ans = (
                        f"Translation to {LANGUAGE_NAMES.get(lang, lang)} is unavailable because the local "
                        "Aya translation model is not installed.\n\n" + ans
                    )
        except Exception:
            ans = f"Translation failed; showing the English answer instead.\n\n{ans}"

        normalized.append({'role': 'assistant', 'content': ans})
        return normalized

    submit.click(chat_submit, inputs=[inp, lang_dropdown, chat], outputs=chat)

if __name__ == '__main__':
    demo.launch()
