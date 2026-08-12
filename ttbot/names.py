from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, Mapping


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
        self.variants_by_name = {name: {normalize_name(name)} for name in self.names}
        for name, variants in self.variants_by_name.items():
            normalized = normalize_name(name)
            if normalized.endswith(" o"):
                variants.add(normalized[:-2])
        for alias, official in self.aliases.items():
            self.variants_by_name[official].add(alias)
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

    def _candidate_names(self, candidates: Iterable[str] | None) -> list[str]:
        if candidates is None:
            return self.names
        allowed = {normalize_name(candidate) for candidate in candidates}
        return [name for name in self.names if normalize_name(name) in allowed]

    def match(
        self,
        query: str,
        *,
        threshold: float = 0.72,
        candidates: Iterable[str] | None = None,
    ) -> NameMatch | None:
        normalized = normalize_name(query)
        if not normalized:
            return None
        candidate_names = self._candidate_names(candidates)
        candidate_set = set(candidate_names)
        if normalized in self.aliases and self.aliases[normalized] in candidate_set:
            return NameMatch(query, self.aliases[normalized], 1.0)
        if normalized in self.by_normalized and self.by_normalized[normalized] in candidate_set:
            return NameMatch(query, self.by_normalized[normalized], 1.0)

        best_name = ""
        best_score = 0.0
        for official in candidate_names:
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

    @staticmethod
    def _ordered_token_positions(query_tokens: list[str], name_tokens: list[str]) -> list[int] | None:
        best: tuple[tuple[int, int], list[int]] | None = None
        for start in range(len(query_tokens)):
            if query_tokens[start] != name_tokens[0]:
                continue
            positions = [start]
            cursor = start + 1
            for name_token in name_tokens[1:]:
                try:
                    position = query_tokens.index(name_token, cursor)
                except ValueError:
                    break
                positions.append(position)
                cursor = position + 1
            if len(positions) != len(name_tokens):
                continue
            gap_count = positions[-1] - positions[0] + 1 - len(positions)
            ranking = gap_count, positions[0]
            if best is None or ranking < best[0]:
                best = ranking, positions
        return best[1] if best else None

    def _match_ordered_name_tokens(
        self,
        text: str,
        candidate_names: list[str],
    ) -> NameMatch | None:
        # OCR can interleave a title with the name, so retain ordered name tokens and ignore words between them.
        query_tokens = normalize_name(text).split()
        best: tuple[tuple[int, int, int, int], str, int] | None = None
        for official in candidate_names:
            for variant in self.variants_by_name[official]:
                name_tokens = variant.split()
                if not name_tokens or len(name_tokens) > len(query_tokens):
                    continue
                positions = self._ordered_token_positions(query_tokens, name_tokens)
                if positions is None:
                    continue
                gap_count = positions[-1] - positions[0] + 1 - len(positions)
                touches_edge = int(positions[0] == 0) + int(positions[-1] == len(query_tokens) - 1)
                ranking = len(name_tokens), sum(map(len, name_tokens)), -gap_count, touches_edge
                if best is None or ranking > best[0]:
                    best = ranking, official, gap_count
        if best is None:
            return None
        _, official, gap_count = best
        return NameMatch(text, official, max(0.62, 1.0 - gap_count * 0.04))

    def match_in_text(
        self,
        text: str,
        *,
        threshold: float = 0.62,
        candidates: Iterable[str] | None = None,
    ) -> NameMatch | None:
        cleaned = normalize_name(text)
        if not cleaned:
            return None
        candidate_names = self._candidate_names(candidates)
        ordered = self._match_ordered_name_tokens(cleaned, candidate_names)
        if ordered:
            return ordered

        direct = self.match(cleaned, threshold=threshold, candidates=candidate_names)
        if direct:
            return direct

        tokens = cleaned.split()
        best: NameMatch | None = None
        for start in range(len(tokens)):
            for end in range(start + 1, min(len(tokens), start + 6) + 1):
                piece = " ".join(tokens[start:end])
                match = self.match(piece, threshold=threshold, candidates=candidate_names)
                if (
                    match
                    and len(piece.split()) == 1
                    and len(normalize_name(match.name).split()) > 1
                    and normalize_name(piece) not in self.aliases
                ):
                    continue
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
