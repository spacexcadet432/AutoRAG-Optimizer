import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_answer(question: str, retrieved_chunks: list, prompt_style: str = "basic"):
    """
    Generate grounded answer using retrieved chunks and prompt style.
    """

    if not retrieved_chunks:
        return "No relevant information found."

    context = "\n\n".join(
        [f"Context {i+1}:\n{chunk}" for i, chunk in enumerate(retrieved_chunks)]
    )

    if prompt_style == "strict":
        prompt = f"""
You are a strict AI assistant.

Use ONLY the provided context to answer the question.
Do NOT use outside knowledge.
If the answer is not explicitly stated in the context, say:
"Insufficient information in the provided context."

--------------------
{context}
--------------------

Question: {question}

Answer:
"""
    else:  # basic
        prompt = f"""
You are a helpful AI assistant.

Answer the question using the provided context.

--------------------
{context}
--------------------

Question: {question}

Answer:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You answer based on given context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()