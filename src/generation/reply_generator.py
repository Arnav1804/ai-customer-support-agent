"""Generate a reply constrained by retrieved historical thread resolutions."""
from __future__ import annotations

from common.llm import ChatClient, LLMError
from retrieval.similar_threads import RetrievedThread


def build_prompt(message: str, intent: str, retrieved_threads: list[RetrievedThread]) -> tuple[str, str]:
    if not retrieved_threads:
        raise ValueError("At least one retrieved thread is required for a grounded reply.")
    evidence = "\n\n".join(
        f"[Historical thread {item.thread_id}; similarity {item.similarity:.3f}]\n{item.text[:2500]}"
        for item in retrieved_threads
    )
    system = (
        "Draft a concise public support reply. Ground every suggested action in the historical "
        "resolutions provided. Do not invent policies, outcomes, links, credits, refunds, or account facts. "
        "Do not request passwords, card numbers, or other sensitive data. If the evidence is insufficient, "
        "say that a support specialist will review the issue. Return only the reply text."
    )
    user = (
        f"Customer message:\n{message}\n\nPredicted intent: {intent}\n\n"
        f"Retrieved historical resolutions:\n{evidence}\n\nDraft one reply grounded in that evidence."
    )
    return system, user


def draft_reply(
    message: str, intent: str, retrieved_threads: list[RetrievedThread], client: ChatClient
) -> str:
    """Draft one grounded reply. LLM failures are propagated to the pipeline caller."""
    if not message or not message.strip():
        raise ValueError("message must not be empty")
    if not intent:
        raise ValueError("intent must not be empty")
    system, user = build_prompt(message, intent, retrieved_threads)
    reply = client.complete(system, user).strip()
    if not reply:
        raise LLMError("Reply generator returned an empty response.")
    return reply
