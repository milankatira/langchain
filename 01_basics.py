import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables (optional for local models, but good practice)
load_dotenv()

def main():
    print("--- Chapter 1: Basics (Local LLMs & Prompts) ---")

    # 1. Initialize the Local LLM
    # We use HuggingFacePipeline to load a model locally.
    # "google/gemma-2b-it" is a lightweight instruction-tuned model.
    # Note: This will download the model (approx 2-4GB) on first run.
    print("Loading local model (this may take a while on first run)...")
    llm = HuggingFacePipeline.from_model_id(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": 100,
            "top_k": 50,
            "temperature": 0.1,
        },
    )

    # 2. Create a Prompt Template
    # TinyLlama works best when instructions are very clear in the user message or system message.
    template = """<|system|>
You are a translator. You must translate the user's input from {input_language} to {output_language}. Do not answer the question, just translate it.</s>
<|user|>
Translate this: {input}</s>
<|assistant|>"""

    prompt = ChatPromptTemplate.from_template(template)

    # 3. Create a Chain (Manual Invocation)
    input_data = {
        "input_language": "English",
        "output_language": "French",
        "input": "I love programming."
    }

    formatted_prompt = prompt.invoke(input_data)
    print(f"\nFormatted Prompt:\n{formatted_prompt}")

    # Invoke the LLM
    print("\nGenerating response...")
    response = llm.invoke(formatted_prompt)

    # 4. Using Output Parsers
    # The local model pipeline often returns the full text (prompt + response).
    # We need to extract just the response part.

    # Simple string parsing to get text after <|assistant|>
    response_text = response.content if hasattr(response, 'content') else str(response)
    if "<|assistant|>" in response_text:
        final_answer = response_text.split("<|assistant|>")[-1].strip()
    else:
        final_answer = response_text

    print(f"\nRaw Response:\n{response_text}")
    print(f"\nParsed Output:\n{final_answer}")

if __name__ == "__main__":
    main()
