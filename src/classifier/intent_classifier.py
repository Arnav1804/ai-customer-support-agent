"""Few-shot LLM classifier for the fixed hand-labelling taxonomy."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from common.llm import ChatClient, LLMError


INTENTS = (
    "Fare & Pricing Issue", "Payment & Charges", "Account & Login",
    "App & Technical Issue", "Driver Issue", "Safety & Vehicle Concern",
    "Ride & Pickup Issue", "Lost & Found", "UberEATS Order Issue",
    "Promotions & Discounts",
)
FEW_SHOT_EXAMPLES = (
    ("Why is the estimated price so much higher during rush hour?", "Fare & Pricing Issue"),
    ("My card was charged twice for one trip.", "Payment & Charges"),
    ("I cannot sign in after changing my phone number.", "Account & Login"),
    ("The app closes whenever I try to request a ride.", "App & Technical Issue"),
    ("The driver refused to take me to my destination.", "Driver Issue"),
    ("My driver was speeding and I did not feel safe.", "Safety & Vehicle Concern"),
    ("My scheduled pickup never arrived.", "Ride & Pickup Issue"),
    ("I left my backpack in the car.", "Lost & Found"),
    ("My delivery arrived cold and missing an item.", "UberEATS Order Issue"),
    ("My promo code was not applied to this ride.", "Promotions & Discounts"),
)


@dataclass(frozen=True)
class IntentPrediction:
    intent: str
    confidence: float  # model self-rating, constrained to [0, 1]


def _json_object(text: str) -> dict:
    """Accept a JSON object, including a model response wrapped in a code fence."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise LLMError(f"Classifier returned invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise LLMError("Classifier response was not a JSON object.")
    return value


def build_prompt(message: str) -> tuple[str, str]:
    examples = "\n".join(f"Message: {example}\nIntent: {intent}" for example, intent in FEW_SHOT_EXAMPLES)
    system = (
        "Classify an Uber support customer message into exactly one supplied intent. "
        "Do not create a new label. Return only JSON with keys intent and confidence. "
        "confidence must be your self-rated decimal confidence from 0.0 to 1.0."
    )
    user = (
        f"Allowed intents: {json.dumps(INTENTS)}\n\n"
        f"Illustrative examples:\n{examples}\n\n"
        f"Message to classify: {message}\n\n"
        "Return only: {\"intent\": \"one allowed intent\", \"confidence\": 0.0}"
    )
    return system, user


def classify_message(message: str, client: ChatClient) -> IntentPrediction:
    """Classify one non-empty message; API errors are raised for the caller to handle."""
    if not message or not message.strip():
        raise ValueError("message must not be empty")
    system, user = build_prompt(message)
    payload = _json_object(client.complete(system, user))
    intent = payload.get("intent")
    confidence = payload.get("confidence")
    if intent not in INTENTS:
        raise LLMError(f"Classifier returned an unsupported intent: {intent!r}")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
        raise LLMError("Classifier confidence must be a number from 0.0 to 1.0.")
    return IntentPrediction(intent=intent, confidence=float(confidence))
