import gradio as gr
import os
from PIL import Image
import pytesseract
import shutil
import pytesseract




from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
#from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM

from langchain_core.documents import Document

vectorstore = None
retriever = None
llm = None


def process_files(files):
    global vectorstore, retriever, llm

    all_docs = []

    for file in files:
        print(" Processing:", file.name)

        # PDF Processing
        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(file.name)
            documents = loader.load()
            print("   📄 Pages:", len(documents))
            all_docs.extend(documents)

        # Image Processing (OCR)
        elif file.name.endswith((".jpg", ".jpeg", ".png")):
            img = Image.open(file.name)
            text = pytesseract.image_to_string(img)
            print("   🖼 Extracted text length:", len(text))

            image_doc = Document(page_content=text)
            all_docs.append(image_doc)

    print(" Splitting text into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(all_docs)
    print("   Total chunks:", len(chunks))

    print(" Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    #retriever = vectorstore.as_retriever() #not reading the numbers in the bill so added logging below
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    #llm = Ollama(model="llama3")
    llm = OllamaLLM(model="llama3")

    print(" Multi-source RAG ready!")

    return f"{len(files)} files processed successfully!"


def chat_with_docs(question):
    global retriever, llm

    if retriever is None:
        return "Please upload and process files first."

    print(" Question:", question)

    # docs = retriever.invoke(question) # not reading the numbers in the bill so added logging below
    docs = retriever.invoke(question)

    print("\n--- Retrieved Context ---")
    for i, doc in enumerate(docs):
        print(f"\nChunk {i+1}:\n", doc.page_content[:500])
    print("\n-------------------------\n")
    #
    print(" Retrieved chunks:", len(docs))

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are a financial assistant.
Use ONLY the provided context.

Context:
{context}

Question:
{question}

Answer:
"""

    print(" Sending to LLM...")
    response = llm.invoke(prompt)
    print(" Response generated.")

    return str(response)



with gr.Blocks() as demo:
    gr.Markdown("#  Multi-Document RAG (PDF + Image)")
    gr.Markdown("Upload PDFs and/or Images, then ask questions.")

    file_input = gr.File(
        file_count="multiple",
        file_types=[".pdf", ".jpg", ".jpeg", ".png"]
    )

    process_button = gr.Button("Process Files")
    status_output = gr.Textbox(label="Status")

    question_input = gr.Textbox(label="Ask a Question")
    answer_output = gr.Textbox(label="Answer")

    process_button.click(process_files, inputs=file_input, outputs=status_output)
    question_input.submit(chat_with_docs, inputs=question_input, outputs=answer_output)

demo.launch()
