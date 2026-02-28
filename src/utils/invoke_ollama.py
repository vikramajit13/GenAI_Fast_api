from typing import List, Dict, Any
import ollama

# This is fine for now
# Make this into a generci llm prompter so that this is not tightly coupled with ollama in future


def query_with_context(
    user_query: str ,
    chunks: List[str],
    selected_indices: List[int],
    trace: bool = True,
) -> str:
    # 1) Context formatting: add chunk ids + clear separators (better than <...>)
    context_text = "\n\n---\n\n".join(
        [f"[CHUNK {i}]\n{chunks[i].strip()}" for i in selected_indices]
    )

    # if required items is there
    # check for items


    if trace:
        print("\n=== RAG TRACE ===")
        print("Query:", user_query)
        print("Selected indices:", selected_indices)
        print("\n=== CONTEXT (first 800 chars) ===")
        print(context_text[:800])
        print("Context chars:", len(context_text))

    # 2) Strong system message for rules
    system_prompt = (
        "You are a careful assistant.\n"
        "Rules:\n"
        "- Use ONLY the provided context.\n"
        "- If the answer is not explicitly in the context, say exactly: "
        '"I don\'t know based on the provided context."\n'
        "- For each bullet, include a short supporting quote from the context.\n"
        "- Keep the answer concise.\n"
    )

    # 3) User prompt: structured output reduces drift
    user_prompt = f"""
Context:
{context_text}

Question:
{user_query}

Answer format (exactly 3 bullets):
- Democracy: <answer> (quote: "...")
- Corporate taxation: <answer> (quote: "...")
- Reproductive rights: <answer> (quote: "...")
""".strip()

    response = ollama.chat(
        model="llama3.1:8b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        # If supported by your Ollama version:
        # options={"temperature": 0}
    )

    # handle both return types
    try:
        return response["message"]["content"]
    except Exception:
        return response.message.content
