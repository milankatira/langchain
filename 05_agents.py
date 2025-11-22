import os
import re
from typing import TypedDict, Annotated, Union
from dotenv import load_dotenv
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END

load_dotenv()

# Define State
class AgentState(TypedDict):
    input: str
    chat_history: list
    scratchpad: str
    final_answer: Union[str, None]

def main():
    print("--- Chapter 5: Agents (Custom ReAct with Local LLM) ---")

    # 1. Define Tools
    @tool
    def get_word_length(word: str) -> int:
        """Returns the length of a word."""
        return len(word)

    @tool
    def add_numbers(a: int, b: int) -> int:
        """Adds two numbers together."""
        return int(a) + int(b)

    tools = {t.name: t for t in [get_word_length, add_numbers]}

    # 2. Setup LLM
    print("Loading local model...")
    llm = HuggingFacePipeline.from_model_id(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": 200,
            "temperature": 0.1,
            "do_sample": True,
        },
    )

    # 3. Define Prompt
    template = """<|system|>
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Example:
Question: What is the length of 'dog'?
Thought: I should check the length of the word 'dog'.
Action: get_word_length
Action Input: dog
Observation: 3
Thought: I have the length.
Final Answer: 3
</s>
<|user|>
Question: {input}
Thought:{scratchpad}</s>
<|assistant|>"""

    prompt = PromptTemplate.from_template(template)

    # 4. Define Nodes
    def reason_node(state: AgentState):
        # Prepare prompt
        tool_names = ", ".join(tools.keys())
        tool_descriptions = "\n".join([f"{t.name}: {t.description}" for t in tools.values()])

        formatted_prompt = prompt.format(
            input=state["input"],
            tools=tool_descriptions,
            tool_names=tool_names,
            scratchpad=state.get("scratchpad", "")
        )

        # Invoke LLM
        response = llm.invoke(formatted_prompt)

        # Extract just the new generation (simple logic)
        # TinyLlama might repeat the prompt, so we need to be careful.
        # But HuggingFacePipeline usually returns the full text.
        # We'll try to extract the part after the prompt if possible, or just use the whole thing
        # and parse the *last* Action.

        # Better: just append the response to scratchpad?
        # But response includes the prompt.

        # Let's assume response is full text.
        # We find the part after <|assistant|>
        if "<|assistant|>" in response:
            generated = response.split("<|assistant|>")[-1].strip()
        else:
            generated = response.strip()

        return {"scratchpad": state.get("scratchpad", "") + generated}

    def act_node(state: AgentState):
        scratchpad = state["scratchpad"]

        # Parse Action and Action Input
        # Look for the last "Action:" and "Action Input:"
        action_match = re.search(r"Action:\s*(.*?)\n", scratchpad)
        input_match = re.search(r"Action Input:\s*(.*?)\n", scratchpad)

        if action_match and input_match:
            action = action_match.group(1).strip()
            action_input = input_match.group(1).strip()

            if action in tools:
                tool = tools[action]
                try:
                    # Simple argument parsing (assuming single arg or simple string)
                    # For add_numbers, we might need to parse "1, 2"
                    if action == "add_numbers":
                        args = [int(x.strip()) for x in action_input.split(",")]
                        observation = tool.invoke({"a": args[0], "b": args[1]})
                    else:
                        observation = tool.invoke(action_input)
                except Exception as e:
                    observation = f"Error: {e}"
            else:
                observation = f"Error: Tool {action} not found."

            new_scratchpad = f"\nObservation: {observation}\nThought:"
            return {"scratchpad": scratchpad + new_scratchpad}

        return {"scratchpad": scratchpad} # No action found

    def check_continue(state: AgentState):
        scratchpad = state["scratchpad"]
        if "Final Answer:" in scratchpad:
            # Extract final answer
            answer = scratchpad.split("Final Answer:")[-1].strip()
            state["final_answer"] = answer
            return "end"
        elif "Action:" in scratchpad and "Observation:" not in scratchpad.split("Action:")[-1]:
            return "act"
        else:
            # Fallback: If the model answered directly without following format
            # We take the last generation as the answer
            # This handles cases where small models ignore instructions
            last_gen = scratchpad.split("Observation:")[-1].strip()
            if "Thought:" in last_gen:
                last_gen = last_gen.split("Thought:")[-1].strip()

            state["final_answer"] = last_gen
            return "end"

    # 5. Define Graph
    workflow = StateGraph(AgentState)
    workflow.add_node("reason", reason_node)
    workflow.add_node("act", act_node)

    workflow.set_entry_point("reason")

    workflow.add_conditional_edges(
        "reason",
        check_continue,
        {
            "act": "act",
            "end": END
        }
    )

    workflow.add_edge("act", "reason")

    app = workflow.compile()

    # 6. Run
    print("\n--- Running Agent ---")
    query = "What is the length of the word 'LangChain'?"
    print(f"Query: {query}")

    inputs = {"input": query, "scratchpad": "", "chat_history": [], "final_answer": None}

    # Run the graph
    result = app.invoke(inputs)

    final_answer = result.get('final_answer')
    if not final_answer:
        # Try to extract from scratchpad if not set
        scratchpad = result.get('scratchpad', '')
        if "Final Answer:" in scratchpad:
            final_answer = scratchpad.split("Final Answer:")[-1].strip()

    print(f"\nFinal Answer: {final_answer}")
    # print(f"\n--- Debug Trace ---\n{result['scratchpad']}")

if __name__ == "__main__":
    main()
