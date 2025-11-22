import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

def main():
    print("--- Chapter 3: Memory (Chat History with Local LLM) ---")

    # 1. Setup the Chain with History
    print("Loading local model...")
    llm = HuggingFacePipeline.from_model_id(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": 150,
            "temperature": 0.7,
        },
    )

    # TinyLlama format with history
    # We manually format the history in the template or use a specific structure
    # For simplicity with RunnableWithMessageHistory, we'll stick to messages but we need to be careful.
    # However, for TinyLlama, string templates are safer.

    template = """<|system|>
You are a helpful assistant.</s>
<|user|>
{history}
{input}</s>
<|assistant|>"""

    prompt = ChatPromptTemplate.from_template(template)

    def parse_output(message):
        text = message.content if hasattr(message, 'content') else str(message)
        if "<|assistant|>" in text:
            return text.split("<|assistant|>")[-1].strip()
        return text

    chain = prompt | llm | parse_output

    # 2. Define a History Store
    store = {}

    def get_session_history(session_id: str):
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    # 3. Wrap the Chain
    # Since we are using a string template, we need to format the history messages into a string
    # before passing to the prompt. But RunnableWithMessageHistory passes a list of BaseMessages.
    # We'll use a wrapper function or a custom chain to handle this if needed.
    # For this simple example, we will rely on LangChain's default stringification of the list which might be messy,
    # OR we can just use a simpler history approach.
    # Let's try to keep it simple: The prompt expects 'history' string.

    with_message_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    # 4. Conversation Loop
    session_id = "user_123"
    print(f"Starting conversation with Session ID: {session_id}")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            break

        config = {"configurable": {"session_id": session_id}}

        response = with_message_history.invoke(
            {"input": user_input},
            config=config
        )

        print(f"AI: {response}")

if __name__ == "__main__":
    main()
