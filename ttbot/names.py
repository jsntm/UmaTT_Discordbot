from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Mapping


def normalize_name(value: str) -> str:
    value = value.lower().replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


@dataclass(frozen=True)
class NameMatch:
    query: str
    name: str
    score: float


@dataclass(frozen=True)
class OutfitMatch:
    query: str
    outfit: str
    score: float


def _load_json_dict(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read reference file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Reference file must contain a JSON object: {path}")
    return payload


class NameMatcher:
    def __init__(
        self,
        names: list[str] | Mapping[str, str],
        *,
        aliases: Mapping[str, str] | None = None,
        outfits_by_id: Mapping[str, list[str]] | None = None,
        outfit_aliases: Mapping[str, str] | None = None,
        thumbs_dir: Path | None = None,
    ) -> None:
        if isinstance(names, Mapping):
            self.ids_by_name = {str(name): str(gametora_id) for name, gametora_id in names.items()}
            self.names = list(self.ids_by_name)
        else:
            self.names = list(names)
            self.ids_by_name = {}
        self.by_normalized = {normalize_name(name): name for name in self.names}
        self.aliases = self._validated_aliases(aliases or {}, self.by_normalized, "uma name")
        self.outfits_by_id = {str(key): [str(value) for value in values] for key, values in (outfits_by_id or {}).items()}
        all_outfits = {
            normalize_name(outfit): outfit
            for outfits in self.outfits_by_id.values()
            for outfit in outfits
        }
        self.outfit_aliases = self._validated_aliases(outfit_aliases or {}, all_outfits, "outfit")
        self.thumbs_dir = thumbs_dir

    @staticmethod
    def _validated_aliases(
        aliases: Mapping[str, str],
        official_by_normalized: Mapping[str, str],
        label: str,
    ) -> dict[str, str]:
        validated = {}
        for alias, target in aliases.items():
            official = official_by_normalized.get(normalize_name(str(target)))
            if not official:
                raise RuntimeError(f"Alias `{alias}` points to unknown {label} `{target}`")
            validated[normalize_name(str(alias))] = official
        return validated

    @classmethod
    def from_reference_files(
        cls,
        names_path: Path,
        aliases_path: Path,
        outfits_path: Path,
        outfit_aliases_path: Path,
        thumbs_dir: Path,
    ) -> "NameMatcher":
        names = _load_json_dict(names_path)
        aliases = _load_json_dict(aliases_path)
        raw_outfits = _load_json_dict(outfits_path)
        outfit_aliases = _load_json_dict(outfit_aliases_path)
        outfits = {
            str(gametora_id): [str(outfit) for outfit in values]
            for gametora_id, values in raw_outfits.items()
            if isinstance(values, list)
        }
        return cls(
            {str(name): str(gametora_id) for name, gametora_id in names.items()},
            aliases={str(key): str(value) for key, value in aliases.items()},
            outfits_by_id=outfits,
            outfit_aliases={str(key): str(value) for key, value in outfit_aliases.items()},
            thumbs_dir=thumbs_dir,
        )

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

    def match_outfit(self, uma_name: str, query: str, *, threshold: float = 0.72) -> OutfitMatch | None:
        gametora_id = self.gametora_id(uma_name)
        outfits = self.outfits_by_id.get(gametora_id, []) if gametora_id else []
        normalized = normalize_name(query)
        if not outfits or not normalized:
            return None
        by_normalized = {normalize_name(outfit): outfit for outfit in outfits}
        alias_target = self.outfit_aliases.get(normalized)
        if alias_target and alias_target in outfits:
            return OutfitMatch(query, alias_target, 1.0)
        if normalized in by_normalized:
            return OutfitMatch(query, by_normalized[normalized], 1.0)

        best_outfit = ""
        best_score = 0.0
        for outfit in outfits:
            outfit_norm = normalize_name(outfit)
            score = SequenceMatcher(None, normalized, outfit_norm).ratio()
            if len(normalized) >= 4 and (normalized in outfit_norm or outfit_norm in normalized):
                score = max(score, 0.9)
            if score > best_score:
                best_outfit = outfit
                best_score = score
        if best_score < threshold:
            return None
        return OutfitMatch(query, best_outfit, best_score)

    def gametora_id(self, uma_name: str) -> str | None:
        official = self.by_normalized.get(normalize_name(uma_name))
        return self.ids_by_name.get(official, None) if official else None

    def thumbnail_path(self, uma_name: str, outfit_name: str) -> Path | None:
        gametora_id = self.gametora_id(uma_name)
        outfits = self.outfits_by_id.get(gametora_id, []) if gametora_id else []
        official_outfit = next((outfit for outfit in outfits if normalize_name(outfit) == normalize_name(outfit_name)), None)
        if not gametora_id or not official_outfit or self.thumbs_dir is None:
            return None
        outfit_id = outfits.index(official_outfit) + 1
        path = self.thumbs_dir / gametora_id / f"{gametora_id}{outfit_id:02d}.png"
        return path if path.exists() else None
