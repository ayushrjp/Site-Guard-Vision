"""Local Ollama + ChromaDB retrieval-augmented Safety Assistant."""

from .safety_assistant import (
    INSUFFICIENT_MESSAGE,
    OFFLINE_MESSAGE,
    SafetyAssistant,
    get_assistant,
)

__all__ = [
    "INSUFFICIENT_MESSAGE",
    "OFFLINE_MESSAGE",
    "SafetyAssistant",
    "get_assistant",
]
