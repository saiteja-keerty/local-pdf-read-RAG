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
import gc
from contextlib import contextmanager
from pathlib import Path
import re
import io
from difflib import get_close_matches
try:
    from . import sbc_utils
except Exception:
    # When the module is executed as a script (e.g. `python insurechat/app.py`) the
    # package-relative import can fail with "attempted relative import with no
    # known parent package". Fall back to an absolute import so the module can
    # be run both as a package and as a script (useful for Hugging Face Spaces).
    import sbc_utils

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
    import fitz
except Exception:
    fitz = None

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
    'hi-latn': 'Hindi (Romanized)',
    'zh': 'Chinese', 'ar': 'Arabic', 'pt': 'Portuguese', 'de': 'German',
    'ja': 'Japanese', 'ko': 'Korean', 'ru': 'Russian', 'vi': 'Vietnamese',
}

ONLINE_GLOSSARY_URL = "https://www.healthcare.gov/glossary/"

INSURANCE_TERM_ALIASES = {
    'health insurance': ['health insurance', 'medical insurance', 'insurance'],
    'copay': ['copay', 'copayment', 'co-pay'],
    'deductible': ['deductible'],
    'coinsurance': ['coinsurance', 'co-insurance'],
    'out-of-pocket maximum': ['out-of-pocket maximum', 'out of pocket maximum', 'oop max'],
    'allowed amount': ['allowed amount', 'allowable', 'eligible expense', 'negotiated rate'],
    'discount': ['discount', 'adjustment', 'network discount', 'provider discount'],
    'eob': ['eob', 'explanation of benefits'],
    'claim': ['claim'],
    'premium': ['premium'],
    'in-network': ['in-network', 'in network'],
    'out-of-network': ['out-of-network', 'out of network', 'non-participating'],
    'balance billing': ['balance billing', 'balance bill'],
}

_ACTIVE_LLM = None
_ACTIVE_LLM_PATH = None
_LLM_LOCK = threading.RLock()

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

    def query(self, q, topk=6, kind=None):
        def matches_kind(idx):
            if kind is None:
                return True
            meta = self.meta[idx] if idx < len(self.meta) else {}
            return isinstance(meta, dict) and meta.get('kind') == kind

        if self.embedding_model is not None and self.use_faiss and self.faiss_index is not None:
            qv = self._embed([q])
            search_count = len(self.texts) if kind is not None else topk
            D, I = self.faiss_index.search(qv, max(search_count, topk))
            results = []
            for idx in I[0]:
                if 0 <= idx < len(self.texts) and matches_kind(idx):
                    results.append(self.texts[idx])
                    if len(results) >= topk:
                        break
            return results
        # Offline lexical fallback. Whole-question substring matching misses
        # queries such as "what is a deductible?".
        stopwords = {'a', 'an', 'and', 'define', 'definition', 'for', 'is', 'meaning',
                     'of', 'please', 'the', 'to', 'what', 'whats'}
        query_terms = set(re.findall(r"[a-z0-9]+", str(q).lower())) - stopwords
        scored = []
        for idx, text in enumerate(self.texts):
            if not matches_kind(idx):
                continue
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
    pages = []
    if PdfReader is not None:
        try:
            reader = PdfReader(path)
            pages = [p.extract_text() or "" for p in reader.pages]
        except Exception:
            pages = []

    text = "\n\n".join(pages).strip()
    if text:
        return text

    # Scanned PDFs have no embedded text. Render each page and OCR it locally.
    if fitz is not None and pytesseract is not None and Image is not None:
        try:
            ocr_pages = []
            with fitz.open(path) as document:
                for page in document:
                    pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                    image = Image.open(io.BytesIO(pixmap.tobytes("png")))
                    ocr_pages.append(pytesseract.image_to_string(image))
            return "\n\n".join(ocr_pages).strip()
        except Exception:
            return ""
    return ""


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


def _extract_document_text(path, fileobj=None):
    """Extract text using file content first, then its extension."""
    target = fileobj if fileobj is not None else path

    # Gradio uploads can have misleading extensions. Pillow verifies whether
    # the bytes are actually an image before PDF parsing is attempted.
    if Image is not None:
        try:
            if hasattr(target, 'read'):
                target.seek(0)
                data = target.read()
                target.seek(0)
                with Image.open(io.BytesIO(data)) as image:
                    image.load()
                return ocr_image(io.BytesIO(data)), 'image'
            with Image.open(target) as image:
                image.verify()
            return ocr_image(target), 'image'
        except Exception:
            pass

    suffix = Path(str(path)).suffix.lower() if path else ''
    if suffix == '.pdf':
        return extract_text_from_pdf(path), 'pdf'
    if suffix in {'.txt', '.md'}:
        try:
            return Path(path).read_text(encoding='utf8'), 'text'
        except Exception:
            return '', 'text'
    return '', 'unknown'


def _extraction_error(kind):
    if kind == 'image' and (pytesseract is None or Image is None):
        return "OCR is unavailable. Install pytesseract and Pillow, and install the Tesseract OCR application."
    if kind == 'pdf' and PdfReader is None:
        return "PDF support is unavailable. Install pypdf."
    if kind == 'pdf' and (fitz is None or pytesseract is None):
        return "No embedded PDF text was found. Install pymupdf and pytesseract to OCR scanned PDFs."
    return "No readable text was found. Check that the file is a valid PDF, image, or UTF-8 text file."

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

    text, kind = _extract_document_text(path, fileobj=fileobj)

    if not text:
        return _extraction_error(kind)

    chunks = chunk_text(text)
    return f"Ingested {path} (chunks={len(chunks)})"


def _document_context_state(file, parsed):
    """Attach the current upload's text to session state without persisting it."""
    if not parsed:
        return {}

    path = None
    fileobj = None
    if isinstance(file, dict):
        path = file.get('tmp_path') or file.get('name') or file.get('file')
        fileobj = file.get('file')
    else:
        path = getattr(file, 'name', None) or file
        fileobj = getattr(file, 'file', None)

    text, _ = _extract_document_text(path, fileobj=fileobj)
    state = dict(parsed)
    state['_document_chunks'] = chunk_text(text) if text else []
    state['_source_name'] = Path(str(path)).name if path else 'uploaded document'
    return state


def _search_kb_for_terms(query, limit=4):
    if not isinstance(query, str) or not query.strip():
        return []

    normalized = query.lower()
    wanted = set()
    for canonical, terms in INSURANCE_TERM_ALIASES.items():
        if any(term in normalized for term in terms):
            wanted.add(canonical)

    if not wanted and re.search(r"\b(insurance|bill|claim|deduct|copay|coverage|medical term|meaning|define)\b", normalized):
        wanted.update(['deductible', 'copay', 'coinsurance', 'allowed amount', 'eob'])

    exact_hits = []
    related_hits = []
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
                if cleaned:
                    hit = f"Source: {f}\n{cleaned}"
                    heading = cleaned.strip('|').split('|', 1)[0].strip().lower()
                    target = exact_hits if any(term in heading for term in wanted) else related_hits
                    if hit not in exact_hits and hit not in related_hits:
                        target.append(hit)

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
                hit = f"Source: {f}\n{block.strip()}"
                if hit not in exact_hits and hit not in related_hits:
                    exact_hits.append(hit)
    return (exact_hits + related_hits)[:limit]


def _correct_insurance_terms(query):
    """Correct close insurance-term misspellings without rewriting normal text."""
    if not isinstance(query, str):
        return query, []

    vocabulary = {name for name in INSURANCE_TERM_ALIASES if ' ' not in name and '-' not in name}
    for aliases in INSURANCE_TERM_ALIASES.values():
        vocabulary.update(alias for alias in aliases if ' ' not in alias and '-' not in alias)
    corrections = []

    def replace_token(match):
        token = match.group(0)
        lowered = token.lower()
        if lowered in vocabulary or lowered in {'allowed'} or len(lowered) < 5:
            return token
        close = get_close_matches(lowered, vocabulary, n=1, cutoff=0.72)
        if not close:
            return token
        matched = close[0]
        canonical = next(
            (name for name, aliases in INSURANCE_TERM_ALIASES.items()
             if matched == name or matched in aliases),
            matched,
        )
        corrections.append((token, canonical))
        return canonical

    return re.sub(r"[A-Za-z][A-Za-z-]*", replace_token, query), corrections


def _is_definition_question(question):
    text = (question or '').strip().lower()
    if re.search(r"\b(what is|what's|define|definition of|meaning of)\b", text):
        return True
    return len(text.split()) == 1 and len(text) < 30


def _definition_term(question):
    text = re.sub(r"[^a-z0-9\s-]", "", (question or '').strip().lower())
    text = re.sub(r"^(?:what is|whats|define|definition of|meaning of)\s+", "", text)
    return text.strip()


def _definition_suggestions(question, limit=3):
    requested = _definition_term(question)
    if not requested:
        return []
    return get_close_matches(requested, list(INSURANCE_TERM_ALIASES), n=limit, cutoff=0.5)


def _answer_from_active_bill(question, parsed):
    if not isinstance(parsed, dict) or not parsed or parsed.get('error'):
        return None

    q = (question or '').lower()

    def has_any(phrases):
        return any(re.search(rf"\b{re.escape(phrase)}\b", q) for phrase in phrases)

    bill_field_words = (
        'bill', 'document', 'upload', 'image', 'owe', 'owed', 'pay', 'balance',
        'discount', 'adjustment', 'service date', 'date of service', 'allowed amount',
        'insurance paid', 'plan paid', 'patient responsibility', 'total charge', 'total billed',
    )
    if _is_definition_question(question) and not has_any(bill_field_words):
        return None

    bill_words = (
        'bill', 'document', 'upload', 'image', 'owe', 'owed', 'pay', 'balance',
        'charge', 'charged', 'total', 'discount', 'adjustment', 'service date',
        'date of service', 'analyze', 'analyse', 'summary', 'summarize', 'explain',
        'warning', 'allowed amount', 'insurance paid', 'plan paid', 'patient responsibility',
    )
    if not has_any(bill_words):
        return None

    owed = parsed.get('patient_owed') or parsed.get('estimated_patient_responsibility')
    billed = parsed.get('billed_amount')
    discount = parsed.get('adjustment_or_discount')
    allowed = parsed.get('allowed_amount')
    plan_paid = parsed.get('plan_paid')
    service_date = parsed.get('service_date')

    if has_any(('owe', 'owed', 'pay', 'balance', 'patient responsibility')):
        if owed:
            return f"The bill lists your patient responsibility as {_format_money(_money_to_float(owed)) or owed}."
        return "The uploaded bill does not clearly show a patient-responsibility or balance amount."
    if has_any(('discount', 'adjustment')):
        return (f"The bill shows an adjustment or discount of {_format_money(_money_to_float(discount))}."
                if discount else "No labeled adjustment or discount was found on the uploaded bill.")
    if has_any(('service date', 'date of service')):
        return f"The service date shown is {service_date}." if service_date else "No service date was found."
    if 'allowed amount' in q:
        return (f"The allowed amount shown is {_format_money(_money_to_float(allowed))}."
                if allowed else "No allowed amount was found. Compare this provider bill with the insurer EOB.")
    if has_any(('insurance paid', 'plan paid')):
        return (f"The plan-paid amount shown is {_format_money(_money_to_float(plan_paid))}."
                if plan_paid else "No plan-paid amount was found on this provider bill.")
    if has_any(('charge', 'charged', 'total')) and billed:
        return f"The total billed amount is {_format_money(_money_to_float(billed))}."

    details = []
    if billed:
        details.append(f"Total billed: {_format_money(_money_to_float(billed))}")
    if discount:
        details.append(f"Adjustment/discount: {_format_money(_money_to_float(discount))}")
    if owed:
        details.append(f"Patient responsibility: {_format_money(_money_to_float(owed)) or owed}")
    if service_date:
        details.append(f"Service date: {service_date}")
    warnings = parsed.get('warnings') or []
    if not details and not warnings:
        return None

    answer = "Uploaded bill summary:\n" + "\n".join(f"- {item}" for item in details)
    if warnings:
        answer += "\n\nImportant:\n" + "\n".join(f"- {warning}" for warning in warnings)
    return answer


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


def _clean_model_answer(answer):
    """Remove common local-model boilerplate and repeated closing text."""
    text = (answer or '').strip()
    text = re.sub(r"^Answer:\s*", "", text, flags=re.I)
    stop_patterns = (
        r"\n\s*Please provide (?:your )?feedback.*",
        r"\n\s*Best regards,.*",
        r"\n\s*Note:.*",
    )
    for pattern in stop_patterns:
        text = re.sub(pattern, "", text, flags=re.I | re.S).strip()
    return text


def _normalize_romanized_hindi(question):
    """Normalize common Romanized Hindi insurance questions before translation."""
    text = (question or '').strip().lower()
    term_aliases = {
        'cope': 'copay', 'copay': 'copay', 'deductible': 'deductible',
        'coinsurance': 'coinsurance', 'premium': 'premium',
        'insurance': 'health insurance', 'insurence': 'health insurance',
    }
    match = re.fullmatch(r"(.+?)\s+(?:kya|kyaa)\s+(?:hai|h)\??", text)
    if match:
        term = term_aliases.get(match.group(1).strip(), match.group(1).strip())
        return f"what is {term}?"
    return question


def _normalize_input_question(question, input_lang):
    if input_lang == 'hi-latn':
        return _normalize_romanized_hindi(question)
    if input_lang == 'hi':
        normalized = (question or '').strip().lower().rstrip('?!। ')
        hindi_definitions = {
            'कोपे क्या है': 'what is copay?',
            'कॉपी क्या है': 'what is copay?',
            'स्वास्थ्य बीमा क्या है': 'what is health insurance?',
            'बीमा क्या है': 'what is health insurance?',
        }
        return hindi_definitions.get(normalized, question)
    return question


def ask_question(q, input_lang='en', active_document=None):
    # Retrieve
    # short-circuit greetings to avoid dumping documents for 'hi' etc.
    greetings = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}
    if isinstance(q, str) and q.strip().lower() in greetings:
        return "Hi — I can help with your bill or insurance questions. Ask me about a term (e.g. 'what is a copay') or upload a bill."
    source_question = _normalize_input_question(q, input_lang)
    english_q = source_question
    try:
        if input_lang and input_lang.lower() != 'en' and source_question == q:
            english_q = translate_text(source_question, target_lang='en', source_lang=input_lang)
    except Exception:
        english_q = source_question

    corrected_q, corrections = _correct_insurance_terms(english_q)

    def finish(answer):
        if corrections:
            original, corrected = corrections[0]
            return f"I interpreted '{original}' as '{corrected}'.\n\n{answer}"
        return answer

    bill_answer = _answer_from_active_bill(corrected_q, active_document)
    if bill_answer:
        return finish(bill_answer)

    # Try the local glossary after normalizing the question into English.
    preliminary_kb_hits = _search_kb_for_terms(corrected_q)

    q_for_search = corrected_q

    active_chunks = active_document.get('_document_chunks', []) if isinstance(active_document, dict) else []
    chunks = list(active_chunks[:6]) if active_chunks else index.query(
        q_for_search, topk=6, kind='knowledge_base'
    )
    # If no retrieved chunks and this looks like a definition request, try KB files directly
    is_definition = _is_definition_question(corrected_q)

    kb_hits = [] if active_chunks else (preliminary_kb_hits or _search_kb_for_terms(q_for_search))
    if kb_hits:
        chunks = kb_hits + [chunk for chunk in chunks if chunk not in kb_hits]
    elif is_definition and not active_chunks:
        # A similarity hit is not enough evidence that the requested term has
        # a local definition.
        chunks = []
    # Build a compact context: include short excerpts and source metadata (if available)
    context_parts = []
    for chunk in chunks:
        if active_chunks:
            src = active_document.get('_source_name', 'uploaded document')
        else:
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

    # Quick numeric Q&A using SBC parsing utilities when an uploaded plan/SBC is active
    if active_chunks:
        try:
            full_text = "\n\n".join(active_chunks)
            plan = sbc_utils.parse_plan_terms(full_text)
            # If user asks for a deductible
            if re.search(r"\bwhat is the overall deductible\b|\boverall deductible\b|\bdeductible\b", corrected_q, re.I):
                d = plan.get('overall_deductible_network_individual')
                if d is not None:
                    return finish(f"The plan's in-network individual deductible appears to be {_format_money(d)}.")
            # If user asks a dollar-based 'how much would I pay' question
            m = re.search(r"\$(\s?[0-9,]+)", corrected_q)
            if m:
                amt = float(re.sub(r"[^0-9.]", "", m.group(0)))
                service_type = 'hospital' if re.search(r"hospital|facility|inpatient|delivery", corrected_q, re.I) else 'service'
                est = sbc_utils.estimate_member_payment(amt, service_type, 'network', plan)
                return finish(est)
        except Exception:
            pass

    if is_definition and not active_chunks:
        if kb_hits:
            answer = _simple_context_answer(corrected_q, kb_hits)
            return finish(f"{answer}\n\nLearn more: {ONLINE_GLOSSARY_URL}")

        suggestions = _definition_suggestions(corrected_q)
        suggestion_text = ""
        if suggestions:
            suggestion_text = " Did you mean " + ", ".join(f"'{term}'" for term in suggestions) + "?"
        return finish(
            f"I could not find a local definition for '{_definition_term(corrected_q)}'."
            f"{suggestion_text}\n\nTrusted glossary: {ONLINE_GLOSSARY_URL}"
        )

    # General questions without an active upload stay on fast local RAG.
    # Llama is reserved for reasoning over the current PDF, image, or text file.
    if not active_chunks:
        return finish(_simple_context_answer(corrected_q, chunks))

    prompt_suffix = (
        "Answer in no more than three short paragraphs using plain language. "
        "Only show calculations when the user asks about an amount or calculation. Include Source when available. "
        "Do not add an 'Answer' heading, notes about editing the response, feedback requests, signatures, "
        "contact information, or repeated closing text."
    )
    prompt = (
        f"{SYSTEM_PROMPT}\n\nDOCUMENT CONTEXT:\n{context}\n\n"
        f"Question: {corrected_q}\n{prompt_suffix}"
    )

    if Llama is None:
        return finish(_simple_context_answer(corrected_q, chunks))

    llm_path = _select_model_for_role("reason")
    if not llm_path:
        return finish(_simple_context_answer(corrected_q, chunks))

    # Keep prompts compact by limiting context length.
    with _use_llm(llm_path) as llm:
        resp = llm(
            prompt=prompt,
            max_tokens=256,
            temperature=0.1,
            stop=["\nNote:", "\nPlease provide", "\nBest regards,"],
        )
    text = resp.get('choices', [{}])[0].get('text') or resp.get('text') or ''
    # If multilingual output requested and translator model available, translate back to requested language
    # Always return English text from this function; translation is handled by the caller.
    return _clean_model_answer(text)


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


def _unload_active_llm():
    """Release the active GGUF before another role loads a different model."""
    global _ACTIVE_LLM, _ACTIVE_LLM_PATH
    if _ACTIVE_LLM is not None:
        close = getattr(_ACTIVE_LLM, 'close', None)
        if callable(close):
            close()
    _ACTIVE_LLM = None
    _ACTIVE_LLM_PATH = None
    gc.collect()


@contextmanager
def _use_llm(model_path):
    """Keep one local GGUF loaded and prevent it being closed during inference."""
    global _ACTIVE_LLM, _ACTIVE_LLM_PATH
    if Llama is None or not model_path:
        yield None
        return

    with _LLM_LOCK:
        if _ACTIVE_LLM_PATH != model_path:
            _unload_active_llm()
            _ACTIVE_LLM = Llama(
                model_path=model_path,
                n_ctx=4096,
                n_threads=max(1, os.cpu_count() // 2),
                n_gpu_layers=-1,
                verbose=False,
            )
            _ACTIVE_LLM_PATH = model_path
        yield _ACTIVE_LLM


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
            with _use_llm(aya_path) as llm:
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
        # Keep the match on one OCR line so an address or account number on a
        # later line cannot become a medical charge.
        pattern = rf"{label}[ \t]*(?:amount|charge|paid|due|owed|responsibility|:)?[ \t]*\$?[ \t]*([0-9][0-9,]*(?:\.[0-9]{{2}})?)"
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

    text, kind = _extract_document_text(path)

    if not text:
        return json.dumps({'error': _extraction_error(kind)}, ensure_ascii=False)

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
        prompt = (
            "Extract the following fields from the document and return valid JSON only. "
            "Fields: document_type, member_id, patient_name, provider_name, service_date, billed_amount, "
            "allowed_amount, plan_paid, adjustment_or_discount, deductible, copay, coinsurance, "
            "noncovered_amount, patient_owed, CPT_codes (list), ICD10_codes (list), warnings (list). "
            "Use null for missing fields. Preserve dollar amounts as strings.\n\n"
            f"Document text:\n{text[:10000]}\n\nReturn only JSON."
        )
        try:
            with _use_llm(extract_model_path) as llm:
                resp = llm(prompt=prompt, max_tokens=512)
            out = resp.get('choices',[{}])[0].get('text') or resp.get('text') or ''
            parsed = json.loads(out)
            if isinstance(parsed, dict):
                calc = estimate_patient_responsibility(parsed)
                parsed.update(calc)
            return json.dumps(parsed, ensure_ascii=False, indent=2)
        except Exception:
            # Fall through to deterministic extraction if inference or JSON parsing fails.
            pass

    # Regex fallback: best-effort extraction
    res = dict(schema)
    lower_text = text.lower()
    if 'explanation of benefits' in lower_text or 'this is not a bill' in lower_text:
        res['document_type'] = 'explanation_of_benefits'
    elif 'claim' in lower_text:
        res['document_type'] = 'claim'
    elif 'amount due' in lower_text or 'balance due' in lower_text or 'hospital statement' in lower_text:
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
        r'patient owes?', r'total due', r'balanc(?:e)?'
    ])

    # Require a CPT/procedure label so ZIP codes and account fragments are not
    # misreported as medical procedure codes.
    cpts = re.findall(r"(?:CPT|procedure(?: code)?)\s*[:#-]?\s*(\d{5})\b", text, re.I)
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


def _parsed_document_state(structured_json):
    try:
        parsed = json.loads(structured_json)
        return parsed if isinstance(parsed, dict) and not parsed.get('error') else {}
    except Exception:
        return {}


def ingest_and_activate(file):
    status_message = ingest_file(file)
    structured_json = extract_structured(file)
    parsed = _document_context_state(file, _parsed_document_state(structured_json))
    if parsed:
        status_message += " The parsed bill is now active in chat."
    return status_message, parsed, structured_json


def extract_and_activate(file):
    structured_json = extract_structured(file)
    parsed = _document_context_state(file, _parsed_document_state(structured_json))
    status_message = "Parsed bill is now active in chat." if parsed else "Bill extraction failed."
    return structured_json, parsed, status_message

with gr.Blocks() as demo:
    active_document = gr.State({})
    gr.Markdown("# InsureChat - local medical insurance helper")
    with gr.Row():
        with gr.Column(scale=2):
            chat = gr.Chatbot()
            input_lang_dropdown = gr.Dropdown(choices=[
                ("English","en"),("Spanish","es"),("French","fr"),("Hindi","hi"),
                ("Hindi (Romanized)","hi-latn"),("Chinese","zh"),("Arabic","ar"),
                ("Portuguese","pt"),("German","de"),("Japanese","ja"),("Korean","ko"),
                ("Russian","ru"),("Vietnamese","vi")
            ], value='en', label='Question language')
            answer_lang_dropdown = gr.Dropdown(choices=[
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

    ingest_btn.click(
        ingest_and_activate,
        inputs=file_input,
        outputs=[status, active_document, extract_output],
    )
    extract_btn.click(
        extract_and_activate,
        inputs=file_input,
        outputs=[extract_output, active_document, status],
    )

    # Chat handler: accepts separate question and answer languages.
    def chat_submit(question, input_lang, answer_lang, active_bill=None, chat_history=None):
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
        ans = ask_question(question, input_lang=input_lang, active_document=active_bill)

        # Return only the selected language instead of English plus a translation.
        try:
            if answer_lang and answer_lang.lower() != 'en':
                translated = translate_text(ans, target_lang=answer_lang, source_lang='en')
                if translated and translated.strip() and translated.strip() != ans.strip():
                    ans = translated.strip()
                else:
                    ans = (
                        f"Translation to {LANGUAGE_NAMES.get(answer_lang, answer_lang)} is unavailable because the local "
                        "Aya translation model is not installed.\n\n" + ans
                    )
        except Exception:
            ans = f"Translation failed; showing the English answer instead.\n\n{ans}"

        normalized.append({'role': 'assistant', 'content': ans})
        return normalized

    submit.click(
        chat_submit,
        inputs=[inp, input_lang_dropdown, answer_lang_dropdown, active_document, chat],
        outputs=chat,
    )

if __name__ == '__main__':
    demo.launch()
