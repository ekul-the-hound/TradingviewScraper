from __future__ import annotations

import ast
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from .ast_normalizer import (
    assign_category,
    detect_category_flags,
    extract_primitive_signature,
    extract_subtree_frequency,
    extract_tree_shape,
    normalize_code,
)
from .similarity_engine import SimilarityBreakdown, compute_similarity
from .validator import validate_python_indicator


@dataclass
class IndicatorRecord:
    name: str
    code: str
    source_url: str
    ingestion_date: str
    fingerprint: str
    category: str
    ast_vector: dict[str, int]
    primitive_vector: dict[str, int]
    tree_shape: dict[str, float]
    parameter_count: int
    similar_to: str | None
    similarity_score: float
    family_cluster: str


class IndicatorWarehouse:
    def __init__(self, root: str | Path = "indicator_vault") -> None:
        self.root = Path(root)
        self.indicators_root = self.root / "indicators"
        self.metadata_root = self.root / "metadata"
        self.fingerprints_root = self.root / "fingerprints"
        self.embeddings_root = self.root / "embeddings"
        self.logs_root = self.root / "ingestion_logs"
        self.db_path = self.metadata_root / "warehouse.duckdb"
        self.index_path = self.metadata_root / "index.json"
        self.similarity_path = self.metadata_root / "similarity_index.parquet"
        self.fingerprint_index_path = self.fingerprints_root / "fingerprint_index.parquet"
        self.vector_store_path = self.embeddings_root / "vector_store.parquet"
        self.daily_log_path = self.logs_root / "daily_log.csv"

    def bootstrap(self) -> None:
        self.metadata_root.mkdir(parents=True, exist_ok=True)
        self.fingerprints_root.mkdir(parents=True, exist_ok=True)
        self.embeddings_root.mkdir(parents=True, exist_ok=True)
        self.logs_root.mkdir(parents=True, exist_ok=True)
        self._init_db()
        if not self.index_path.exists():
            self.index_path.write_text(json.dumps({"name_to_fingerprint": {}, "fingerprints": {}}, indent=2), encoding="utf-8")
        if not self.daily_log_path.exists():
            pd.DataFrame(columns=["ingestion_date", "name", "status", "message"]).to_csv(self.daily_log_path, index=False)

    def ingest(self, name: str, code: str, source_url: str) -> IndicatorRecord:
        self.bootstrap()
        ok, message = validate_python_indicator(code)
        now = datetime.now(UTC).isoformat()
        if not ok:
            self._append_log(now, name, "failed", message)
            raise ValueError(message)

        normalization = normalize_code(code)
        ast_vector = extract_subtree_frequency(normalization.normalized_tree)
        primitive_vector = extract_primitive_signature(normalization.normalized_tree)
        tree_shape = extract_tree_shape(normalization.normalized_tree)
        flags = detect_category_flags(normalization.normalized_tree)
        category = assign_category(flags)
        parameter_count = self._parameter_count(code)

        nearest = self.find_nearest_neighbors(ast_vector, top_k=10)
        similar_to = None
        score = 0.0
        family_cluster = normalization.fingerprint[:12]

        for candidate in nearest:
            breakdown = compute_similarity(
                normalization.fingerprint,
                candidate["fingerprint"],
                ast_vector,
                candidate["ast_vector"],
                primitive_vector,
                candidate["primitive_vector"],
                tree_shape,
                candidate["tree_shape"],
                parameter_count,
                candidate["parameter_count"],
            )
            if breakdown.final_score > score:
                similar_to = candidate["name"]
                score = breakdown.final_score
                if breakdown.classification in {"near_duplicate", "variant", "same_family"}:
                    family_cluster = candidate["family_cluster"]

        record = IndicatorRecord(
            name=name,
            code=code,
            source_url=source_url,
            ingestion_date=now,
            fingerprint=normalization.fingerprint,
            category=category,
            ast_vector=ast_vector,
            primitive_vector=primitive_vector,
            tree_shape=tree_shape,
            parameter_count=parameter_count,
            similar_to=similar_to,
            similarity_score=score,
            family_cluster=family_cluster,
        )

        self._store_indicator_file(record)
        self._upsert_index(record)
        self._upsert_database(record)
        self._upsert_parquet_indices(record)
        self._append_log(now, name, "ingested", "ok")
        return record

    def find_nearest_neighbors(self, ast_vector: dict[str, int], top_k: int = 10) -> list[dict[str, Any]]:
        vectors = self._load_vector_store()
        if vectors.empty:
            return []
        features = ["add", "sub", "mul", "div", "rolling", "ewm", "comparisons", "logical", "shift"]
        query = [float(ast_vector.get(k, 0)) for k in features]
        try:
            from annoy import AnnoyIndex

            idx = AnnoyIndex(len(features), "angular")
            for i, row in vectors.iterrows():
                idx.add_item(i, [float(row[f]) for f in features])
            idx.build(10)
            neighbors = idx.get_nns_by_vector(query, min(top_k, len(vectors)))
            result = vectors.iloc[neighbors].to_dict(orient="records")
            return result
        except Exception:
            vectors = vectors.copy()
            vectors["distance"] = vectors.apply(
                lambda row: sum((float(row[f]) - query[j]) ** 2 for j, f in enumerate(features)), axis=1
            )
            return vectors.sort_values("distance").head(top_k).drop(columns=["distance"]).to_dict(orient="records")

    def query(self, sql: str) -> pd.DataFrame:
        with duckdb.connect(str(self.db_path)) as con:
            return con.execute(sql).fetchdf()

    def _init_db(self) -> None:
        with duckdb.connect(str(self.db_path)) as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS indicators (
                    name TEXT PRIMARY KEY,
                    fingerprint TEXT,
                    category TEXT,
                    primitive_vector TEXT,
                    parameter_count INTEGER,
                    similarity_cluster TEXT,
                    similarity_score DOUBLE,
                    ingestion_date TEXT,
                    source_url TEXT
                )
                """
            )

    def _parameter_count(self, code: str) -> int:
        tree = ast.parse(code)
        func = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
        return len(func.args.args)

    def _store_indicator_file(self, record: IndicatorRecord) -> None:
        category_dir = self.indicators_root / record.category
        category_dir.mkdir(parents=True, exist_ok=True)
        file_path = category_dir / f"{record.name}.py"
        file_path.write_text(record.code.strip() + "\n", encoding="utf-8")

    def _upsert_index(self, record: IndicatorRecord) -> None:
        payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        payload["name_to_fingerprint"][record.name] = record.fingerprint
        payload["fingerprints"][record.fingerprint] = {
            "name": record.name,
            "category": record.category,
            "parameters": record.parameter_count,
            "source_url": record.source_url,
            "ingestion_date": record.ingestion_date,
            "similar_to": record.similar_to,
            "similarity_score": record.similarity_score,
            "family_cluster": record.family_cluster,
        }
        self.index_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _upsert_database(self, record: IndicatorRecord) -> None:
        with duckdb.connect(str(self.db_path)) as con:
            con.execute(
                """
                INSERT OR REPLACE INTO indicators (
                    name, fingerprint, category, primitive_vector, parameter_count,
                    similarity_cluster, similarity_score, ingestion_date, source_url
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    record.name,
                    record.fingerprint,
                    record.category,
                    json.dumps(record.primitive_vector, sort_keys=True),
                    record.parameter_count,
                    record.family_cluster,
                    record.similarity_score,
                    record.ingestion_date,
                    record.source_url,
                ],
            )

    def _upsert_parquet_indices(self, record: IndicatorRecord) -> None:
        sim_df = self._safe_read_parquet(self.similarity_path, [
            "name",
            "similar_to",
            "similarity_score",
            "family_cluster",
            "ingestion_date",
        ])
        sim_df = pd.concat(
            [
                sim_df[sim_df["name"] != record.name],
                pd.DataFrame([
                    {
                        "name": record.name,
                        "similar_to": record.similar_to,
                        "similarity_score": record.similarity_score,
                        "family_cluster": record.family_cluster,
                        "ingestion_date": record.ingestion_date,
                    }
                ]),
            ],
            ignore_index=True,
        )
        sim_df.to_parquet(self.similarity_path, index=False)

        fp_df = self._safe_read_parquet(self.fingerprint_index_path, ["name", "fingerprint"])
        fp_df = pd.concat(
            [fp_df[fp_df["name"] != record.name], pd.DataFrame([{"name": record.name, "fingerprint": record.fingerprint}])],
            ignore_index=True,
        )
        fp_df.to_parquet(self.fingerprint_index_path, index=False)

        features = ["add", "sub", "mul", "div", "rolling", "ewm", "comparisons", "logical", "shift"]
        vector_df = self._safe_read_parquet(
            self.vector_store_path,
            ["name", "fingerprint", "family_cluster", "parameter_count", "category", "ast_vector", "primitive_vector", "tree_shape"]
            + features,
        )
        item = {
            "name": record.name,
            "fingerprint": record.fingerprint,
            "family_cluster": record.family_cluster,
            "parameter_count": record.parameter_count,
            "category": record.category,
            "ast_vector": json.dumps(record.ast_vector, sort_keys=True),
            "primitive_vector": json.dumps(record.primitive_vector, sort_keys=True),
            "tree_shape": json.dumps(record.tree_shape, sort_keys=True),
        }
        for feature in features:
            item[feature] = record.ast_vector.get(feature, 0)
        vector_df = pd.concat([vector_df[vector_df["name"] != record.name], pd.DataFrame([item])], ignore_index=True)
        vector_df.to_parquet(self.vector_store_path, index=False)

    def _load_vector_store(self) -> pd.DataFrame:
        if not self.vector_store_path.exists():
            return pd.DataFrame()
        df = pd.read_parquet(self.vector_store_path)
        if df.empty:
            return df
        df = df.copy()
        df["ast_vector"] = df["ast_vector"].map(json.loads)
        df["primitive_vector"] = df["primitive_vector"].map(json.loads)
        df["tree_shape"] = df["tree_shape"].map(json.loads)
        return df

    @staticmethod
    def _safe_read_parquet(path: Path, columns: list[str]) -> pd.DataFrame:
        if path.exists():
            return pd.read_parquet(path)
        return pd.DataFrame(columns=columns)

    def _append_log(self, ingestion_date: str, name: str, status: str, message: str) -> None:
        row = pd.DataFrame([
            {"ingestion_date": ingestion_date, "name": name, "status": status, "message": message}
        ])
        if self.daily_log_path.exists():
            row.to_csv(self.daily_log_path, mode="a", header=False, index=False)
        else:
            row.to_csv(self.daily_log_path, index=False)


def supported_queries() -> dict[str, str]:
    return {
        "volatility_with_ema": "SELECT * FROM indicators WHERE category = 'volatility' AND primitive_vector LIKE '%\\\"ema\\\": 1%'",
        "composites_three_params": "SELECT * FROM indicators WHERE category = 'composite' AND parameter_count >= 3",
        "similar_to_name": "SELECT b.* FROM indicators a JOIN indicators b ON a.similarity_cluster = b.similarity_cluster WHERE a.name = ? AND b.similarity_score > 0.85",
        "by_primitive": "SELECT * FROM indicators WHERE primitive_vector LIKE ?",
        "by_category": "SELECT * FROM indicators WHERE category = ?",
        "by_parameter_count": "SELECT * FROM indicators WHERE parameter_count = ?",
    }
