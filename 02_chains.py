import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def main():
    print("--- Chapter 2: Chains (LCEL with Local LLM) ---")

    # 1. Setup components
    print("Loading local model...")
    llm = HuggingFacePipeline.from_model_id(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": 100,
            "temperature": 0.7,
        },
    )

    # TinyLlama format
    template = """<|system|>
You are a helpful assistant that tells jokes about {topic}.</s>
<|user|>
Tell me a joke.</s>
<|assistant|>"""

    prompt = ChatPromptTemplate.from_template(template)

    # Custom parser to handle local model output (stripping prompt)
    def parse_output(message):
        text = message.content if hasattr(message, 'content') else str(message)
        if "<|assistant|>" in text:
            return text.split("<|assistant|>")[-1].strip()
        return text

    # 2. Create a Chain using LCEL
    chain = prompt | llm | parse_output

    # 3. Invoke the Chain
    print("\n--- Invoking Chain ---")
    response = chain.invoke({"topic": "programming"})
    print(response)

    # 4. Batch Processing
    print("\n--- Batch Processing ---")
    topics = [{"topic": "cats"}, {"topic": "dogs"}] # Reduced batch size for local execution speed
    responses = chain.batch(topics)

    for topic, resp in zip(topics, responses):
        print(f"Topic: {topic['topic']} -> Joke: {resp}")

if __name__ == "__main__":
    main()
