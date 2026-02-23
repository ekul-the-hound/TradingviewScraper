from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .ann_index import AnnIndex
from .ast_normalizer import category_from_features, extract_structural_features
from .similarity_engine import SimilarityEngine
from .validator import validate_indicator_code


@dataclass(frozen=True)
class IngestionRecord:
    name: str
    fingerprint: str
    category: str
    similarity_cluster: str
    similarity_score: float
    source_url: str
    ingestion_date: str


class IndicatorWarehouse:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.metadata_dir = root / "metadata"
        self.indicators_dir = root / "indicators"
        self.fingerprints_dir = root / "fingerprints"
        self.embeddings_dir = root / "embeddings"
        self.logs_dir = root / "ingestion_logs"
        self.index_path = self.metadata_dir / "index.json"
        self.similarity_path = self.metadata_dir / "similarity_index.parquet"
        self.fingerprint_index_path = self.fingerprints_dir / "fingerprint_index.parquet"
        self.vector_store_path = self.embeddings_dir / "vector_store.parquet"
        self.daily_log_path = self.logs_dir / "daily_log.csv"
        self.ann_path = self.embeddings_dir / "ann_vectors.npz"
        self.engine = SimilarityEngine()
        self.ann = AnnIndex.load(dimension=10, storage_path=self.ann_path)
        self._ensure_layout()

    def _ensure_layout(self) -> None:
        for path in (
            self.metadata_dir,
            self.indicators_dir,
            self.fingerprints_dir,
            self.embeddings_dir,
            self.logs_dir,
            self.root / "clusters",
        ):
            path.mkdir(parents=True, exist_ok=True)
        for category in ("trend", "momentum", "volatility", "volume", "mean_reversion", "composite"):
            (self.indicators_dir / category).mkdir(parents=True, exist_ok=True)
        if not self.index_path.exists():
            self.index_path.write_text(json.dumps({"indicators": {}, "fingerprints": {}}, indent=2))

    def _load_index(self) -> dict[str, Any]:
        return json.loads(self.index_path.read_text())

    def _save_index(self, index: dict[str, Any]) -> None:
        self.index_path.write_text(json.dumps(index, indent=2, sort_keys=True))

    def _features_to_vector(self, features: dict[str, Any]) -> list[float]:
        primitives = [features["primitives"][name] for name in sorted(features["primitives"]) ]
        shape = features["shape"]
        return primitives + [shape["depth"], shape["branching_factor"], shape["rolling_window_count"], shape["cross_logic_count"]]

    def ingest(self, *, name: str, code: str, source_url: str, family_cluster: str | None = None) -> IngestionRecord:
        validation = validate_indicator_code(code)
        if not validation.valid:
            raise ValueError(validation.error)

        features = extract_structural_features(code)
        category = category_from_features(features)
        fingerprint = str(features["fingerprint"])
        now = datetime.now(timezone.utc).isoformat()

        index = self._load_index()
        existing_indicators = index["indicators"]

        similarity_score = 0.0
        similarity_to = ""
        if existing_indicators:
            vector = self._features_to_vector(features)
            neighbors = self.ann.query(vector, top_k=10)
            for neighbor_name, _ in neighbors:
                candidate = existing_indicators[neighbor_name]
                result = self.engine.score(features, candidate["features"])
                if result.similarity_score > similarity_score:
                    similarity_score = result.similarity_score
                    similarity_to = neighbor_name

        cluster = family_cluster or (similarity_to if similarity_score >= 0.70 else name)
        record = {
            "name": name,
            "fingerprint": fingerprint,
            "category": category,
            "parameters": int(features["parameter_count"]),
            "source_url": source_url,
            "ingestion_date": now,
            "similar_to": similarity_to,
            "similarity_score": similarity_score,
            "family_cluster": cluster,
            "features": features,
        }

        indicator_path = self.indicators_dir / category / f"{name}.py"
        indicator_path.write_text(code)
        existing_indicators[name] = record
        index["fingerprints"].setdefault(fingerprint, [])
        if name not in index["fingerprints"][fingerprint]:
            index["fingerprints"][fingerprint].append(name)
        self._save_index(index)

        vector = self._features_to_vector(features)
        self.ann.add(name, vector)
        self.ann.save()

        self._materialize_tables(index)

        return IngestionRecord(
            name=name,
            fingerprint=fingerprint,
            category=category,
            similarity_cluster=cluster,
            similarity_score=similarity_score,
            source_url=source_url,
            ingestion_date=now,
        )

    def _materialize_tables(self, index: dict[str, Any]) -> None:
        rows = []
        fingerprints = []
        for item in index["indicators"].values():
            rows.append(
                {
                    "name": item["name"],
                    "fingerprint": item["fingerprint"],
                    "category": item["category"],
                    "primitive_vector": json.dumps(item["features"]["primitives"], sort_keys=True),
                    "parameter_count": item["parameters"],
                    "similarity_cluster": item["family_cluster"],
                    "similarity_score": item["similarity_score"],
                    "ingestion_date": item["ingestion_date"],
                    "source_url": item["source_url"],
                }
            )
            fingerprints.append(
                {
                    "name": item["name"],
                    "fingerprint": item["fingerprint"],
                    "normalized_ast": item["features"]["fingerprint"],
                }
            )

        df = pd.DataFrame(rows)
        fdf = pd.DataFrame(fingerprints)
        if not df.empty:
            df.to_parquet(self.similarity_path, index=False)
            df.to_parquet(self.vector_store_path, index=False)
            fdf.to_parquet(self.fingerprint_index_path, index=False)
            log_df = df[["name", "ingestion_date", "source_url"]]
            log_df.to_csv(self.daily_log_path, index=False)
