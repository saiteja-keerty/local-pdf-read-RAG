import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

vectorstore = None
retriever = None
llm = None

def process_pdf(file):
    global vectorstore, retriever, llm

    print(" PDF received:", file.name)

    # Load PDF
    loader = PyPDFLoader(file.name)
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
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # Create vector DB
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()

    print(" Vector DB ready!")

    # Load LLM
    llm = Ollama(model="llama3")
    print(" Ollama LLM ready!")

    return "PDF processed successfully! You can now ask questions."


def chat_with_pdf(question):
    global retriever, llm

    if retriever is None:
        return "Please upload and process a PDF first."

    print(" Question:", question)

    # Retrieve relevant chunks
    docs = retriever.invoke(question) #.get_relevant_documents(question)
    print(" Retrieved chunks:", len(docs))

    context = "\n\n".join([doc.page_content for doc in docs])

    # Create prompt manually
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
