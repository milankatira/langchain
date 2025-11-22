import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def main():
    print("--- Chapter 4: RAG (Retrieval Augmented Generation with Local LLM) ---")

    # 1. Load Document
    loader = TextLoader("sample.txt")
    docs = loader.load()
    print(f"Loaded {len(docs)} document(s).")

    # 2. Split Document
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    splits = text_splitter.split_documents(docs)
    print(f"Split into {len(splits)} chunks.")

    # 3. Create Embeddings and Vector Store
    # We use HuggingFaceEmbeddings (runs locally).
    print("Loading local embeddings (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Creating vector store...")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
    )

    # 4. Create Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # 5. Create RAG Chain
    template = """<|system|>
Answer the question based only on the following context:
{context}</s>
<|user|>
{question}</s>
<|assistant|>"""
    prompt = ChatPromptTemplate.from_template(template)

    print("Loading local LLM...")
    llm = HuggingFacePipeline.from_model_id(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": 200,
            "temperature": 0.1,
        },
    )

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    def parse_output(message):
        text = message.content if hasattr(message, 'content') else str(message)
        if "<|assistant|>" in text:
            return text.split("<|assistant|>")[-1].strip()
        return text

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | parse_output
    )

    # 6. Ask a Question
    question = "What is LCEL?"
    print(f"\nQuestion: {question}")
    print("Generating answer...")
    response = rag_chain.invoke(question)
    print(f"Answer: {response}")

if __name__ == "__main__":
    main()
