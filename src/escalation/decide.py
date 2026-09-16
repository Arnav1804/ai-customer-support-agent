"""Explicit, auditable escalation rules for the support pipeline."""
from __future__ import annotations

import re
from dataclasses import dataclass


HIGH_RISK_INTENTS = {"Safety & Vehicle Concern"}
FRAUD_OR_COMPROMISE = re.compile(r"\b(fraud|fraudulent|hacked|unauthori[sz]ed|stolen)\b", re.IGNORECASE)
NEGATIVE_SENTIMENT = re.compile(
    r"\b(angry|furious|terrible|worst|disgusting|outrageous|unsafe|scared|harass(?:ed|ment)?|"
    r"never\s+again|lawsuit|police)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class EscalationDecision:
    action: str  # auto-handle or escalate
    reason: str


def decide_escalation(message: str, intent: str, confidence: float, drafted_reply: str) -> EscalationDecision:
    """Apply ordered rules and return exactly one action with one sentence of reasoning."""
    if confidence < 0 or confidence > 1:
        raise ValueError("confidence must be from 0.0 to 1.0")
    if not drafted_reply or not drafted_reply.strip():
        return EscalationDecision("escalate", "Escalated because a grounded draft reply was not available.")
    if confidence < 0.6:
        return EscalationDecision("escalate", f"Escalated because classifier confidence ({confidence:.2f}) is below 0.60.")
    if intent in HIGH_RISK_INTENTS:
        return EscalationDecision("escalate", f"Escalated because {intent} is a high-risk support category.")
    if FRAUD_OR_COMPROMISE.search(message):
        return EscalationDecision("escalate", "Escalated because the message indicates possible fraud or account compromise.")
    if NEGATIVE_SENTIMENT.search(message):
        return EscalationDecision("escalate", "Escalated because the message contains strong negative or safety-related language.")
    return EscalationDecision("auto-handle", "Auto-handled because confidence is at least 0.60 and no explicit escalation rule matched.")
