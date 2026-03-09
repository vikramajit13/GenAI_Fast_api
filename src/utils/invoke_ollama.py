from typing import List, Dict, Any
import ollama

# This is fine for now
# Make this into a generci llm prompter so that this is not tightly coupled with ollama in future


def query_with_context(
    user_query: str,
    chunks: list[dict],
    trace: bool = True,
) -> str:
    context_text = "\n\n---\n\n".join(
        [
            f"[CHUNK {c['chunk_index']}]\n[SENTENCE {c['sentence_index']}]\n{c['text'].strip()}"
            for c in chunks
        ]
    )

    if trace:
        print("\n=== RAG TRACE ===")
        print("Query:", user_query)
        print("\n=== CONTEXT (first 1200 chars) ===")
        print(context_text[:1200])
        print("Context chars:", len(context_text))
        print("Num chunks:", len(chunks))

    system_prompt = """
You are a retrieval-grounded assistant.

Rules:
1. Use only the provided context.
2. Do not use outside knowledge.
3. If the answer is not explicitly supported, say exactly:
   "I don't know based on the provided context."
4. Every answer must include a direct quote and chunk citation.
5. The Answer field must be a short complete sentence, not just a keyword or phrase fragment.
6. Do not output generic placeholders like "foreign policy challenge".
7. Do not repeat headings or output multiple answer blocks for the same field.
8. Use the most direct supporting quote available in the provided chunks.
""".strip()

    user_prompt = f"""
Context:
{context_text}

Question:
{user_query}

Return exactly in this format:

Democracy:
- Answer: <short answer>
- Quote: "<direct quote>"
- Source: [CHUNK X],[SENTENCE Z]

Foreign policy challenge:
- Answer: <one concrete issue only>
- Quote: "<direct quote>"
- Source: [CHUNK Y],[SENTENCE Z]

If either item is not supported by the context, write exactly:
I don't know based on the provided context.
""".strip()

    response = ollama.chat(
        model="llama3.1:8b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0},
    )

    try:
        return response["message"]["content"]
    except Exception:
        return response.message.content
    
def get_lexical_query(user_query):

    prompt = f"""
You convert a user query into a short keyword search query.

Rules:
- Return 3 to 6 keywords only
- Use words that appear in the query
- Do not invent new terms
- No punctuation
- No explanations

User query:
{user_query}

Keywords:
"""

    response = ollama.generate(
        model="llama3.1:8b",
        prompt=prompt,
        options={"temperature":0, "num_predict":20}
    )

    return response["response"].strip()