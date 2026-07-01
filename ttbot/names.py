from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
import re


def normalize_name(value: str) -> str:
    value = value.lower()
    value = value.replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


ALIASES = {
    "mr cb": "Mr. C.B.",
    "m r cb": "Mr. C.B.",
    "mrcb": "Mr. C.B.",
    "ks miracle": "K.S.Miracle",
    "k s miracle": "K.S.Miracle",
    "ksmiracle": "K.S.Miracle",
    "tm opera o": "TM Opera O",
    "t m opera o": "TM Opera O",
    "tm opera 0": "TM Opera O",
    "t m opera 0": "TM Opera O",
    "t opera o": "TM Opera O",
    "t opera 0": "TM Opera O",
    "sakura bakushin o": "Sakura Bakushin O",
    "sakura bakushin 0": "Sakura Bakushin O",
    "calstone light o": "Calstone Light O",
    "calstone light 0": "Calstone Light O",
}


@dataclass(frozen=True)
class NameMatch:
    query: str
    name: str
    score: float


class NameMatcher:
    def __init__(self, names: list[str]) -> None:
        self.names = names
        self.by_normalized = {normalize_name(name): name for name in names}
        self.aliases = {normalize_name(alias): official for alias, official in ALIASES.items()}

    @classmethod
    def from_reference_file(cls, path: Path) -> "NameMatcher":
        names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return cls(names)

    def match(self, query: str, *, threshold: float = 0.72) -> NameMatch | None:
        normalized = normalize_name(query)
        if not normalized:
            return None
        if normalized in self.aliases:
            return NameMatch(query, self.aliases[normalized], 1.0)
        if normalized in self.by_normalized:
            return NameMatch(query, self.by_normalized[normalized], 1.0)

        best_name = ""
        best_score = 0.0
        for official in self.names:
            official_norm = normalize_name(official)
            score = SequenceMatcher(None, normalized, official_norm).ratio()
            if len(normalized) >= 4 and (normalized in official_norm or official_norm in normalized):
                score = max(score, 0.9)
            if score > best_score:
                best_name = official
                best_score = score
        if best_score < threshold:
            return None
        return NameMatch(query, best_name, best_score)

    def match_in_text(self, text: str, *, threshold: float = 0.62) -> NameMatch | None:
        cleaned = normalize_name(text)
        if not cleaned:
            return None
        direct = self.match(cleaned, threshold=threshold)
        if direct:
            return direct

        tokens = cleaned.split()
        best: NameMatch | None = None
        for start in range(len(tokens)):
            for end in range(start + 1, min(len(tokens), start + 6) + 1):
                piece = " ".join(tokens[start:end])
                match = self.match(piece, threshold=threshold)
                if match and (best is None or match.score > best.score):
                    best = NameMatch(text, match.name, match.score)
        return best
