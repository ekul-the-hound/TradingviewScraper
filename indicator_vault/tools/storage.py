from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

import duckdb
import pandas as pd


@dataclass(frozen=True)
class IndicatorRecord:
    name: str
    fingerprint: str
    category: str
    primitive_vector: Dict[str, int]
    parameter_count: int
    similarity_cluster: str
    similarity_score: float
    ingestion_date: str
    source_url: str


class IndicatorStorage:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.metadata_dir = root / "metadata"
        self.embeddings_dir = root / "embeddings"
        self.fingerprints_dir = root / "fingerprints"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.fingerprints_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.metadata_dir / "index.json"
        self.duckdb_path = self.metadata_dir / "warehouse.duckdb"
        self._initialize_index()
        self._initialize_db()

    def _initialize_index(self) -> None:
        if not self.index_path.exists():
            self.index_path.write_text(json.dumps({"name_to_fingerprint": {}, "fingerprint_to_similar": {}}))

    def _initialize_db(self) -> None:
        conn = duckdb.connect(str(self.duckdb_path))
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS indicators (
                name VARCHAR,
                fingerprint VARCHAR,
                category VARCHAR,
                primitive_vector JSON,
                parameter_count INTEGER,
                similarity_cluster VARCHAR,
                similarity_score DOUBLE,
                ingestion_date DATE,
                source_url VARCHAR
            )
            """
        )
        conn.close()

    def upsert_indicator(self, record: IndicatorRecord, similar_to: List[str]) -> None:
        index = json.loads(self.index_path.read_text())
        index["name_to_fingerprint"][record.name] = record.fingerprint
        index["fingerprint_to_similar"][record.fingerprint] = similar_to
        self.index_path.write_text(json.dumps(index, indent=2, sort_keys=True))
        conn = duckdb.connect(str(self.duckdb_path))
        conn.execute("DELETE FROM indicators WHERE name = ?", [record.name])
        conn.execute(
            """
            INSERT INTO indicators
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                record.name,
                record.fingerprint,
                record.category,
                json.dumps(record.primitive_vector, sort_keys=True),
                record.parameter_count,
                record.similarity_cluster,
                record.similarity_score,
                record.ingestion_date,
                record.source_url,
            ],
        )
        conn.close()

    def export_similarity_index(self) -> None:
        conn = duckdb.connect(str(self.duckdb_path))
        df = conn.execute(
            "SELECT name, fingerprint, similarity_cluster, similarity_score, category, ingestion_date FROM indicators"
        ).fetch_df()
        conn.close()
        df.to_parquet(self.metadata_dir / "similarity_index.parquet", index=False)

    def append_ingestion_log(self, name: str, source_url: str, status: str) -> None:
        log_path = self.root / "ingestion_logs" / "daily_log.csv"
        row = pd.DataFrame(
            [
                {
                    "ingestion_date": date.today().isoformat(),
                    "name": name,
                    "source_url": source_url,
                    "status": status,
                }
            ]
        )
        if log_path.exists() and log_path.stat().st_size > 0:
            row.to_csv(log_path, mode="a", header=False, index=False)
        else:
            row.to_csv(log_path, index=False)

    def query(self, sql: str) -> pd.DataFrame:
        conn = duckdb.connect(str(self.duckdb_path))
        df = conn.execute(sql).fetch_df()
        conn.close()
        return df

    def save_vector_store(self, rows: List[Dict[str, Any]]) -> None:
        pd.DataFrame(rows).to_parquet(self.embeddings_dir / "vector_store.parquet", index=False)

    def save_fingerprint_index(self, rows: List[Dict[str, Any]]) -> None:
        pd.DataFrame(rows).to_parquet(self.fingerprints_dir / "fingerprint_index.parquet", index=False)
