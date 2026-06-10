from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Optional
import math
import re

import numpy as np
import pandas as pd

from ekg_models import MetricSpec, SOURCE_GROUP_PRIORITY, normalize_source_group
from ekg_specs import metric_specs, negative_specs
from ekg_utils import clean_text


class SimpleBM25:
    def __init__(self, texts: Iterable[str], k1: float = 1.5, b: float = 0.75):
        self.docs = [clean_text(text).split() for text in texts]
        self.k1 = k1
        self.b = b
        self.n = len(self.docs)
        self.avgdl = sum(len(doc) for doc in self.docs) / max(1, self.n)
        self.df = Counter()
        for doc in self.docs:
            self.df.update(set(doc))

    def scores(self, query: str) -> np.ndarray:
        tokens = clean_text(query).split()
        output = np.zeros(self.n)
        for idx, doc in enumerate(self.docs):
            tf = Counter(doc)
            for term in tokens:
                if tf.get(term, 0) == 0:
                    continue
                df = self.df[term]
                idf = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
                denominator = tf[term] + self.k1 * (
                    1 - self.b + self.b * len(doc) / self.avgdl
                )
                output[idx] += idf * tf[term] * (self.k1 + 1) / denominator
        return output


class RetrievalEngine:
    def __init__(self, fragments: pd.DataFrame, model: Any):
        self.fragments = fragments.reset_index(drop=True)
        self.model = model
        self.bm25 = SimpleBM25(self.fragments["text"].fillna("").tolist())

    def strict_candidate(self, spec: MetricSpec) -> Optional[pd.Series]:
        candidates = self.fragments[
            (self.fragments["structured"] == True)
            & (self.fragments["year"] == spec.year)
        ].copy()
        if spec.kind == "money":
            candidates = candidates[candidates["unit_family"].isin(["money", "other"])]
        aliases = [clean_text(alias) for alias in spec.aliases]
        candidates = candidates[
            candidates["metric_label"].fillna("").map(clean_text).apply(
                lambda label: any(alias in label for alias in aliases)
            )
        ]
        if candidates.empty:
            return None
        candidates["role_priority"] = candidates["role"].map(
            lambda role: SOURCE_GROUP_PRIORITY.get(normalize_source_group(role), 99)
        )
        candidates["route_priority"] = candidates["extraction_route"].map(
            lambda route: 0 if route == "VERIFIED_KNOWN_DOCUMENT_ADAPTER" else 1
        )
        candidates["label_length"] = candidates["metric_label"].fillna("").str.len()
        return candidates.sort_values(["role_priority", "route_priority", "label_length"]).iloc[0]

    def semantic_candidate(self, spec: MetricSpec) -> Optional[pd.Series]:
        candidates = self.fragments.copy()
        if spec.year is not None:
            same_year = candidates[candidates["year"] == spec.year]
            
            no_year_text = candidates[(candidates["year"].isna()) & (candidates["structured"] == False)]
            if not same_year.empty:
                candidates = pd.concat([same_year, no_year_text], ignore_index=False).drop_duplicates()
        if candidates.empty:
            return None
        scores = self.bm25.scores(spec.query)
        candidates["_bm25"] = [scores[i] for i in candidates.index]
        pool = candidates.sort_values("_bm25", ascending=False).head(20)
        query_vec = self.model.encode(
            ["search_query: " + clean_text(spec.query)],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        pass_vecs = self.model.encode(
            ["search_document: " + clean_text(text) for text in pool["text"].tolist()],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        pool = pool.copy()
        pool["_semantic_score"] = pass_vecs @ query_vec
        pool["_source_priority"] = pool["role"].map(
            lambda role: SOURCE_GROUP_PRIORITY.get(normalize_source_group(role), 99)
        )
        return pool.sort_values(["_semantic_score", "_source_priority"], ascending=[False, True]).iloc[0]

    def build_input_candidates(self, scoring_year: int) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for spec in metric_specs(scoring_year):
            strict = self.strict_candidate(spec)
            if strict is not None:
                candidate = strict
                method = "STRICT_MATCH"
                accepted = True
                score = 1.0
            else:
                candidate = self.semantic_candidate(spec)
                method = "RoSBERTa_SUGGESTION" if candidate is not None else "NOT_FOUND"
                accepted = False
                score = float(candidate.get("_semantic_score", 0)) if candidate is not None else None
            rows.append(
                {
                    "accept_for_score": accepted,
                    "key": spec.key,
                    "block": spec.block,
                    "label": spec.label,
                    "year": spec.year,
                    "value_used": float(candidate["value"]) if candidate is not None and pd.notna(candidate["value"]) else 0.0,
                    "unit": candidate["unit"] if candidate is not None else "",
                    "method": method,
                    "e5_score": score,
                    "source": candidate["filename"] if candidate is not None else "",
                    "location": candidate["location"] if candidate is not None else "",
                    "candidate_text": candidate["text"][:240] if candidate is not None else "",
                    "review_note": (
                        "Автоматически принято: точное совпадение по структурированной строке."
                        if accepted else
                        ("Информация не найдена; в расчёте использовано 0." if candidate is None else
                         "Найден только семантический кандидат RoSBERTa; по умолчанию в расчёте использовано 0, пока значение не подтверждено.")
                    ),
                }
            )
        return pd.DataFrame(rows)

    def build_negative_candidates(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        text_series = self.fragments["text"].fillna("")
        for spec in negative_specs():
            strict_indices = [
                idx for idx, text in enumerate(text_series)
                if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in spec.strict_patterns)
            ]
            strict = self.fragments.iloc[strict_indices[0]] if strict_indices else None
            semantic = None
            if strict is None:
                
                temp_spec = MetricSpec(spec.key, spec.block, spec.label, spec.query, [], "text", None)
                semantic = self.semantic_candidate(temp_spec)
            candidate = strict if strict is not None else semantic
            rows.append(
                {
                    "confirmed_adverse_fact": False,
                    "key": spec.key,
                    "label": spec.label,
                    "method": "STRICT_TEXT_CANDIDATE_REVIEW" if strict is not None else "RoSBERTa_REVIEW_ONLY",
                    "source": candidate["filename"] if candidate is not None else "",
                    "location": candidate["location"] if candidate is not None else "",
                    "candidate_text": candidate["text"][:260] if candidate is not None else "",
                }
            )
        return pd.DataFrame(rows)

