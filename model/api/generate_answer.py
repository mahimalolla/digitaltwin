from llama_cpp import Llama
import os

# Load GGUF model
llm = Llama(
    model_path="model/llms/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=8,
    n_gpu_layers=1  # Increase if you have GPU VRAM (e.g., 6–8GB+)
)

def generate_answer(query: str, context: str) -> str:
    """
    Generate an answer using the LLM with provided query and context.
    
    Args:
        query (str): The user’s question.
        context (str): Retrieved context from semantic search.
    
    Returns:
        str: LLM-generated answer.
    """
def generate_answer(query: str, context: str) -> str:
    prompt = f"""You are Warren Buffett's financial digital twin, trained on his shareholder letters, interviews, and writings.

Use the following context to answer the user's question concisely and accurately.

Context:
{context}

Question:
{query}

Answer:"""
    output = llm(prompt.strip(), max_tokens=300, stop=["</s>"])
    return output["choices"][0]["text"].strip()
