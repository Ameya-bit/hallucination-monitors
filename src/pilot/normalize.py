"""Answer normalization and matching for short-form QA (SQuAD/TriviaQA style)."""
import re
import string

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = set(string.punctuation)


def normalize(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in _PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(pred: str, aliases: list[str]) -> bool:
    p = normalize(pred)
    return any(p == normalize(a) for a in aliases if a)


def contains_match(pred: str, aliases: list[str]) -> bool:
    """True if any normalized alias appears as a token span inside the prediction."""
    p = f" {normalize(pred)} "
    return any(f" {normalize(a)} " in p for a in aliases if a and normalize(a))
