from __future__ import annotations

import csv
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import duckdb
except ModuleNotFoundError:
    duckdb = None

from .ast_normalizer import normalize_code
from .category_detector import detect_category
from .similarity_engine import extract_features, similarity_between, top_k_neighbors, vectorize_features
from .validator import validate_indicator_code


@dataclass
class IndicatorRecord:
    name: str
    fingerprint: str
    category: str
    primitive_vector: dict[str, int]
    parameter_count: int
    similarity_cluster: str
    similarity_score: float
    ingestion_date: str
    source_url: str
    similar_to: list[str]


class IndicatorWarehouse:
    def __init__(self, root: str | Path = "indicator_vault") -> None:
        self.root = Path(root)
        self.indicators_root = self.root / "indicators"
        self.metadata_dir = self.root / "metadata"
        self.tools_dir = self.root / "tools"
        self.fingerprints_path = self.root / "fingerprints" / "fingerprint_index.parquet"
        self.similarity_path = self.root / "metadata" / "similarity_index.parquet"
        self.vector_store_path = self.root / "embeddings" / "vector_store.parquet"
        self.index_path = self.root / "metadata" / "index.json"
        self.log_path = self.root / "ingestion_logs" / "daily_log.csv"
        self.db_path = self.root / "metadata" / "indicators.duckdb"
        self.sqlite_path = self.root / "metadata" / "indicators.sqlite"
        self.family_graph_path = self.root / "clusters" / "family_graph.graphml"
        self._ensure_layout()
        self._init_database()

    def _ensure_layout(self) -> None:
        paths = [
            self.root,
            self.metadata_dir,
            self.root / "ingestion_logs",
            self.root / "embeddings",
            self.root / "clusters",
            self.root / "fingerprints",
            self.indicators_root / "trend",
            self.indicators_root / "momentum",
            self.indicators_root / "volatility",
            self.indicators_root / "volume",
            self.indicators_root / "mean_reversion",
            self.indicators_root / "composite",
        ]
        for path in paths:
            path.mkdir(parents=True, exist_ok=True)
        if not self.index_path.exists():
            self.index_path.write_text(
                json.dumps(
                    {
                        "name_to_fingerprint": {},
                        "fingerprint_to_similar": {},
                        "category": {},
                        "parameters": {},
                        "source_url": {},
                        "ingestion_date": {},
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        if not self.log_path.exists():
            with self.log_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["timestamp", "name", "status", "reason"])

    def _init_database(self) -> None:
        if duckdb is not None:
            with duckdb.connect(str(self.db_path)) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS indicators (
                        name VARCHAR PRIMARY KEY,
                        fingerprint VARCHAR,
                        category VARCHAR,
                        primitive_vector JSON,
                        parameter_count INTEGER,
                        similarity_cluster VARCHAR,
                        similarity_score DOUBLE,
                        ingestion_date VARCHAR,
                        source_url VARCHAR
                    )
                    """
                )
            return
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS indicators (
                    name TEXT PRIMARY KEY,
                    fingerprint TEXT,
                    category TEXT,
                    primitive_vector TEXT,
                    parameter_count INTEGER,
                    similarity_cluster TEXT,
                    similarity_score REAL,
                    ingestion_date TEXT,
                    source_url TEXT
                )
                """
            )

    def ingest(self, name: str, code: str, source_url: str) -> IndicatorRecord:
        validation = validate_indicator_code(code)
        if not validation.is_valid:
            self._append_log(name, "invalid", validation.reason)
            raise ValueError(validation.reason)

        now = datetime.now(timezone.utc).isoformat()
        normalized = normalize_code(code)
        category, _ = detect_category(code)
        _, features = extract_features(code)

        similar_to, similarity_score, similarity_cluster = self._compute_similarity(name, code)

        record = IndicatorRecord(
            name=name,
            fingerprint=normalized.fingerprint,
            category=category,
            primitive_vector=features.primitive_signature,
            parameter_count=features.parameter_count,
            similarity_cluster=similarity_cluster,
            similarity_score=similarity_score,
            ingestion_date=now,
            source_url=source_url,
            similar_to=similar_to,
        )

        self._write_indicator_file(record, code)
        self._update_index(record)
        self._upsert_database(record)
        self._append_parquet_rows(record, features)
        self._append_log(name, "ingested", "ok")
        return record

    def _write_indicator_file(self, record: IndicatorRecord, code: str) -> None:
        file_path = self.indicators_root / record.category / f"{record.name}.py"
        file_path.write_text(code.strip() + "\n", encoding="utf-8")

    def _load_index(self) -> dict[str, Any]:
        return json.loads(self.index_path.read_text(encoding="utf-8"))

    def _update_index(self, record: IndicatorRecord) -> None:
        index = self._load_index()
        index["name_to_fingerprint"][record.name] = record.fingerprint
        index["fingerprint_to_similar"][record.fingerprint] = record.similar_to
        index["category"][record.name] = record.category
        index["parameters"][record.name] = record.parameter_count
        index["source_url"][record.name] = record.source_url
        index["ingestion_date"][record.name] = record.ingestion_date
        self.index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")

    def _upsert_database(self, record: IndicatorRecord) -> None:
        params = [
            record.name,
            record.fingerprint,
            record.category,
            json.dumps(record.primitive_vector),
            record.parameter_count,
            record.similarity_cluster,
            record.similarity_score,
            record.ingestion_date,
            record.source_url,
        ]
        if duckdb is not None:
            with duckdb.connect(str(self.db_path)) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO indicators
                    (name, fingerprint, category, primitive_vector, parameter_count,
                     similarity_cluster, similarity_score, ingestion_date, source_url)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    params,
                )
            return
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO indicators
                (name, fingerprint, category, primitive_vector, parameter_count,
                 similarity_cluster, similarity_score, ingestion_date, source_url)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                params,
            )

    def _append_parquet_rows(self, record: IndicatorRecord, features: Any) -> None:
        fingerprint_row = pd.DataFrame([
            {
                "name": record.name,
                "fingerprint": record.fingerprint,
                "ingestion_date": record.ingestion_date,
            }
        ])
        similarity_row = pd.DataFrame([
            {
                "name": record.name,
                "similar_to": json.dumps(record.similar_to),
                "similarity_score": record.similarity_score,
                "family_cluster": record.similarity_cluster,
                "ingestion_date": record.ingestion_date,
            }
        ])
        vector_row = pd.DataFrame([
            {
                "name": record.name,
                "vector": json.dumps(vectorize_features(features)),
                "ingestion_date": record.ingestion_date,
            }
        ])
        self._append_or_create_parquet(self.fingerprints_path, fingerprint_row)
        self._append_or_create_parquet(self.similarity_path, similarity_row)
        self._append_or_create_parquet(self.vector_store_path, vector_row)

    @staticmethod
    def _append_or_create_parquet(path: Path, row: pd.DataFrame) -> None:
        if path.exists():
            current = pd.read_parquet(path)
            updated = pd.concat([current, row], ignore_index=True)
            updated.to_parquet(path, index=False)
        else:
            row.to_parquet(path, index=False)

    def _append_log(self, name: str, status: str, reason: str) -> None:
        with self.log_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([datetime.now(timezone.utc).isoformat(), name, status, reason])

    def _compute_similarity(self, name: str, code: str) -> tuple[list[str], float, str]:
        existing_codes: list[tuple[str, str]] = []
        for category_dir in self.indicators_root.iterdir():
            if not category_dir.is_dir():
                continue
            for file in category_dir.glob("*.py"):
                existing_codes.append((file.stem, file.read_text(encoding="utf-8")))

        if not existing_codes:
            return [], 0.0, f"cluster_{name}"

        _, new_features = extract_features(code)
        new_vector = vectorize_features(new_features)

        neighbor_vectors = []
        vector_map: dict[str, str] = {}
        for existing_name, existing_code in existing_codes:
            _, features = extract_features(existing_code)
            neighbor_vectors.append((existing_name, vectorize_features(features)))
            vector_map[existing_name] = existing_code

        top_neighbors = top_k_neighbors(new_vector, neighbor_vectors, k=10)

        best_name = ""
        best_score = 0.0
        for neighbor_name, _ in top_neighbors:
            result = similarity_between(code, vector_map[neighbor_name])
            if result.similarity_score > best_score:
                best_name = neighbor_name
                best_score = result.similarity_score

        if best_name:
            cluster = f"cluster_{min(name, best_name)}"
            return [best_name], best_score, cluster
        return [], 0.0, f"cluster_{name}"
