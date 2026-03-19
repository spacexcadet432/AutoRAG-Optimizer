import os
from typing import List

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def build_prompt(question: str, retrieved_chunks: List[str], prompt_style: str = "basic") -> str:
    """
    Build a prompt string from question, retrieved chunks, and a prompt style.
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
    else:
        prompt = f"""
You are a helpful AI assistant.

Answer the question using the provided context.

--------------------
{context}
--------------------

Question: {question}

Answer:
"""

    return prompt


def generate_from_prompt(prompt: str) -> str:
    """
    Call the chat completion API with a pre-built prompt.
    """

    if prompt == "No relevant information found.":
        return prompt

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You answer based on given context."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    return response.choices[0].message.content.strip()


def generate_answer(question: str, retrieved_chunks: List[str], prompt_style: str = "basic") -> str:
    """
    Backwards-compatible helper that builds a prompt and generates an answer.
    """

    prompt = build_prompt(question, retrieved_chunks, prompt_style=prompt_style)
    return generate_from_prompt(prompt)

