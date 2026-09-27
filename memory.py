"""
memory.py — Conversation history management and query rewriting.
"""

from groq import Groq


def add_message(messages: list, role: str, content: str) -> None:
    """Append a message dict to the conversation history list."""
    messages.append({"role": role, "content": content})


def get_history(messages: list, max_messages: int = 6) -> list:
    """Return the most recent max_messages entries from history."""
    return messages[-max_messages:]


def rewrite_query(query: str, history: list, client: Groq) -> str:
    """
    Rewrite a follow-up question into a clear, standalone question
    using the conversation history as context.

    If the model returns an empty or very short result, the original
    query is returned unchanged so retrieval is never silently skipped.

    Args:
        query:   The user's latest message.
        history: Recent conversation history (list of role/content dicts).
        client:  Groq API client instance.

    Returns:
        A standalone question string.
    """
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Rewrite the user's question as a clear standalone question "
                        "using the conversation context.\n"
                        "Only add context if the question clearly refers to something "
                        "mentioned earlier.\n"
                        "If the question is about a completely NEW topic, return it unchanged.\n"
                        "Return ONLY the rewritten question — no explanation, no punctuation change."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Conversation:\n{history}\n\nFollow-up question: {query}",
                },
            ],
            temperature=0,
            max_tokens=200,
            stop=["\n"],
        )
        result = response.choices[0].message.content or ""
        result = result.strip()

        # Accept the rewrite only if it is a meaningful, complete sentence
        if result and len(result.split()) >= 3:
            return result

    except Exception:
        # Network/API error — fall back silently
        pass

    return query