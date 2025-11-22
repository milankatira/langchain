# Learn LangChain (Local Models Edition)

This project contains a series of Python scripts designed to teach you the core concepts of [LangChain](https://python.langchain.com/) using **local Hugging Face models**. No OpenAI API key is required!

## Prerequisites

-   Python 3.8+
-   A computer with decent RAM (8GB+ recommended) as models run locally.

## Setup

1.  **Clone or Download** this repository.
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Environment Variables**:
    You can create a `.env` file, but it's not strictly required for local models unless you use other tools.

## Chapters

### [Chapter 1: Basics (Local LLMs)](01_basics.py)

**Concepts**: `HuggingFacePipeline`, Prompt Templates.

-   Loads `TinyLlama/TinyLlama-1.1B-Chat-v1.0` locally.
-   **Note**: The first run will download the model (approx 2-4GB).

Run:

```bash
python 01_basics.py
```

### [Chapter 2: Chains (LCEL)](02_chains.py)

**Concepts**: LangChain Expression Language (LCEL), Piping (`|`).

-   Connects the local LLM with prompts and parsers.

Run:

```bash
python 02_chains.py
```

### [Chapter 3: Memory (Chat History)](03_memory.py)

**Concepts**: `RunnableWithMessageHistory`.

-   Adds memory to the local LLM chain.

Run:

```bash
python 03_memory.py
```

### [Chapter 4: RAG (Retrieval Augmented Generation)](04_rag.py)

**Concepts**: `HuggingFaceEmbeddings`, Chroma Vector Store.

-   Uses `sentence-transformers/all-MiniLM-L6-v2` for local embeddings.
-   Retrieves context from `sample.txt` to answer questions.

Run:

```bash
python 04_rag.py
```

### [Chapter 5: Agents (ReAct)](05_agents.py)

**Concepts**: ReAct Pattern, `create_react_agent`.

-   Uses the ReAct pattern (Reasoning + Acting) which is more suitable for smaller local models than function calling.
-   Defines custom tools (Calculator, Word Length).

Run:

```bash
python 05_agents.py
```

## Troubleshooting

-   **Slow Performance**: Local models run on your CPU/GPU. It might be slow on older hardware.
-   **Memory Errors**: If you run out of RAM, try a smaller model or close other applications.
