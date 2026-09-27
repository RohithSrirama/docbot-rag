"""
generation.py — LLM answer generation via Groq API.

Provides two functions:
    generate_answer()        — blocking, prints tokens to stdout (used by CLI)
    generate_answer_stream() — generator, yields tokens (used by Flask SSE)
"""

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL     = "openai/gpt-oss-120b"
MAX_TOKENS = 600

# ── Shared system prompt template ────────────────────────────────────────────

def _build_system_prompt(context: str) -> str:
    return f"""You are DocBot, an internal assistant for the company. \
You answer questions strictly based on the three company documents in your knowledge base:

  1. HR Handbook (hr_handbook.pdf) — employee policies, leave, sick days, working hours, \
remote work rules, medical certificates, and HR procedures.
  2. Office Guide (office_guide.txt) — office conduct, workspace rules, equipment usage, \
confidentiality, visitor policy, and general workplace guidelines.
  3. Refund Policy (refund_policy.pdf) — customer refund eligibility, timelines, exceptions, \
billing disputes, chargeback rules, and the refund request process.

KNOWLEDGE BASE (retrieved excerpts):
{context}

RULES:
1. For questions ABOUT THE DOCUMENTS:
   - Answer ONLY from the knowledge base excerpts above.
   - If the answer is not present, say: "I don't have information about that in the company documents."
   - Never invent or assume facts not explicitly stated in the knowledge base.

2. For META QUESTIONS (about you or your capabilities):
   - You are DocBot, built to help staff quickly find answers from the company documents listed above.
   - Do not reveal that you are an AI, LLM, or trained by any company.

3. For GREETINGS and CASUAL messages:
   - Respond briefly and professionally, then invite the user to ask about the documents.

FORMATTING RULES (strictly enforced):
- Output plain text only. No exceptions.
- Do NOT use asterisks (*), underscores (_), pound signs (#), backticks (`), or any markdown symbols.
- Do NOT use bullet points with dashes or dots.
- Do NOT bold or italicize any text.
- Write in clear, flowing sentences. Use plain numbered lines (1. 2. 3.) only if truly necessary.
- Keep answers concise and direct.
- If the question is completely unrelated to the documents, say: \
"I'm designed to answer questions about the company documents. \
Try asking about HR policies, office rules, or the refund process."
"""


# ── Blocking version (CLI) ────────────────────────────────────────────────────

def generate_answer(question: str, context: str) -> str:
    """
    Generate a complete answer and print it token-by-token to stdout.
    Returns the full response string when done.
    Used by main.py (CLI mode).
    """
    messages = [
        {"role": "system", "content": _build_system_prompt(context)},
        {"role": "user",   "content": question},
    ]

    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0,
        max_tokens=MAX_TOKENS,
        stream=True,
    )

    full_response = ""
    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            print(token, end="", flush=True)
            full_response += token

    print()   # newline after streaming ends
    return full_response


# ── Streaming generator (Flask SSE) ──────────────────────────────────────────

def generate_answer_stream(question: str, context: str):
    """
    Yield content tokens one at a time for use with Flask SSE streaming.
    Used by app.py.
    """
    messages = [
        {"role": "system", "content": _build_system_prompt(context)},
        {"role": "user",   "content": question},
    ]

    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0,
        max_tokens=MAX_TOKENS,
        stream=True,
    )

    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            yield token