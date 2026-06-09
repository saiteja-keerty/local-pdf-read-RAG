import gradio as gr

# Try to import the original (heavy) dependencies; if they fail (e.g. torch DLL issues),
# fall back to lightweight implementations that avoid torch/transformers.
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.llms import Ollama
    from langchain_core.prompts import PromptTemplate
    HEAVY_BACKEND = True
except Exception as _err:
    HEAVY_BACKEND = False
    print("Falling back to lightweight PDF loader/retriever due to import error:", _err)
    # Lightweight PDF loader using pypdf
    from pypdf import PdfReader

    class _SimpleDoc:
        def __init__(self, text, page_index=0):
            self.page_content = text
            self.metadata = {"page": page_index}

    def PyPDFLoader(path):
        class L:
            def __init__(self, p):
                self.p = p

            def load(self):
                reader = PdfReader(self.p)
                docs = []
                for i, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    docs.append(_SimpleDoc(text, i))
                return docs

        return L(path)

    # Simple character splitter
    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size=500, chunk_overlap=100):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap

        def split_documents(self, documents):
            out = []
            for d in documents:
                text = d.page_content
                if not text:
                    continue
                start = 0
                while start < len(text):
                    end = start + self.chunk_size
                    chunk = text[start:end]
                    out.append(_SimpleDoc(chunk, d.metadata.get("page", 0)))
                    start = max(end - self.chunk_overlap, end)
            return out

    # Simple retriever using TF-IDF if available, otherwise substring match
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        class SimpleRetriever:
            def __init__(self, docs):
                self.docs = docs
                self.texts = [d.page_content for d in docs]
                self.vectorizer = TfidfVectorizer().fit(self.texts)
                self.vectors = self.vectorizer.transform(self.texts)

            def invoke(self, query, topk=3):
                qv = self.vectorizer.transform([query])
                sims = cosine_similarity(qv, self.vectors)[0]
                idxs = sims.argsort()[::-1][:topk]
                return [self.docs[i] for i in idxs]

    except Exception:
        class SimpleRetriever:
            def __init__(self, docs):
                self.docs = docs

            def invoke(self, query, topk=3):
                hits = [d for d in self.docs if query.lower() in d.page_content.lower()]
                return hits[:topk]

    # Lightweight LLM fallback (echo / context-based) if Ollama unavailable
    class Ollama:
        def __init__(self, model=None):
            self.model = model

        def invoke(self, prompt):
            # Very small heuristic: return the context first 1000 chars as an answer stub
            if "Context:" in prompt:
                parts = prompt.split("Context:")
                if len(parts) > 1:
                    ctx = parts[1].split("Question:")[0].strip()
                    return ctx[:1000] or "(no context found)"
            return "(LLM fallback)"

vectorstore = None
retriever = None
llm = None

def process_pdf(file):
    global vectorstore, retriever, llm

    import traceback

    def _resolve_path(f):
        # Accept a file path string, a file-like with .name, or a Gradio dict
        if isinstance(f, str):
            return f
        if isinstance(f, dict):
            return f.get("name") or f.get("tmp_path") or f.get("file")
        if hasattr(f, "name"):
            return f.name
        return None
    try:
        path = _resolve_path(file)
        print(" PDF received:", path)
        if not path:
            raise ValueError("Could not resolve uploaded file path")

        # Load PDF
        loader = PyPDFLoader(path)
        documents = loader.load()
        print(" Loaded pages:", len(documents))

        # Split text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        chunks = splitter.split_documents(documents)
        print(" Created chunks:", len(chunks))

        # Create embeddings
        print(" Creating embeddings...")
        embeddings = None
        if HEAVY_BACKEND:
            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )

        # Create vector DB
        if HEAVY_BACKEND and embeddings is not None:
            vectorstore = FAISS.from_documents(chunks, embeddings)
            retriever = vectorstore.as_retriever()
        else:
            # lightweight retriever
            retriever = SimpleRetriever(chunks)

        print(" Vector DB ready!")

        # Load LLM
        llm = Ollama(model="llama3")
        print(" Ollama LLM ready!")

        return "PDF processed successfully! You can now ask questions."
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        return f"ERROR processing PDF: {e}\n{tb}"


def chat_with_pdf(question):
    global retriever, llm
    import traceback
    try:
        if retriever is None:
            return "Please upload and process a PDF first."

        print(" Question:", question)

        docs = retriever.invoke(question)
        print(" Retrieved chunks:", len(docs))

        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
You are a helpful assistant.
Answer ONLY from the provided context.

Context:
{context}

Question:
{question}

Answer:
"""

        print(" Sending to LLM...")
        response = llm.invoke(prompt)
        print(" Response generated.")
        return response
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        return f"ERROR in chat: {e}\n{tb}"


with gr.Blocks() as demo:
    gr.Markdown("#  Local RAG Chatbot (Modern Version)")
    gr.Markdown("Upload a PDF, process it, then ask questions.")

    file_input = gr.File(label="Upload PDF", file_types=[".pdf"])
    process_button = gr.Button("Process PDF")
    status_output = gr.Textbox(label="Status")

    question_input = gr.Textbox(label="Ask a Question")
    answer_output = gr.Textbox(label="Answer")

    process_button.click(process_pdf, inputs=file_input, outputs=status_output)
    question_input.submit(chat_with_pdf, inputs=question_input, outputs=answer_output)

demo.launch()
