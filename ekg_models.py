from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any


@dataclass
class Fragment:
    document_id: str
    filename: str
    role: str
    location: str
    text: str
    metric_label: str = ""
    year: Optional[int] = None
    value: Optional[float] = None
    unit: str = ""
    unit_family: str = "other"
    structured: bool = False
    extraction_route: str = "PARSED"


@dataclass
class MetricSpec:
    key: str
    block: str
    label: str
    query: str
    aliases: list[str]
    kind: str = "number"
    year: Optional[int] = None


@dataclass
class NegativeSpec:
    key: str
    block: str
    label: str
    query: str
    strict_patterns: list[str]


SOURCE_GROUP_LABELS = {
    "government_sources": "Государственные источники",
    "other_sources": "Прочие источники",
}

SOURCE_GROUP_PRIORITY = {
    "government_sources": 0,
    "other_sources": 1,
}


def normalize_source_group(value: Any) -> str:
    text = str(value or "other_sources").strip()
    return text if text in SOURCE_GROUP_PRIORITY else "other_sources"
