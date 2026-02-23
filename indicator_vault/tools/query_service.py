from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

try:
    import duckdb
except ModuleNotFoundError:
    duckdb = None


class IndicatorQueryService:
    def __init__(self, db_root: str | Path = "indicator_vault/metadata") -> None:
        root = Path(db_root)
        self.duckdb_path = str(root / "indicators.duckdb")
        self.sqlite_path = str(root / "indicators.sqlite")

    def _query(self, sql: str, params: list | None = None) -> pd.DataFrame:
        if duckdb is not None:
            with duckdb.connect(self.duckdb_path, read_only=True) as conn:
                return conn.execute(sql, params or []).fetch_df()
        with sqlite3.connect(self.sqlite_path) as conn:
            return pd.read_sql_query(sql, conn, params=params or [])

    def volatility_with_ema(self):
        return self._query(
            """
            SELECT * FROM indicators
            WHERE category = 'volatility' AND lower(primitive_vector) LIKE '%ema%'
            """
        )

    def composites_with_min_parameters(self, minimum: int = 3):
        return self._query(
            """
            SELECT * FROM indicators
            WHERE category = 'composite' AND parameter_count >= ?
            """,
            [minimum],
        )

    def similar_to(self, name: str, threshold: float = 0.85):
        return self._query(
            """
            SELECT * FROM indicators
            WHERE similarity_cluster = (
                SELECT similarity_cluster FROM indicators WHERE name = ?
            ) AND similarity_score > ?
            """,
            [name, threshold],
        )

    def by_primitive(self, primitive: str):
        return self._query(
            """
            SELECT * FROM indicators
            WHERE lower(primitive_vector) LIKE ?
            """,
            [f"%{primitive.lower()}%"],
        )

    def by_category(self, category: str):
        return self._query("SELECT * FROM indicators WHERE category = ?", [category])

    def by_parameter_count(self, minimum: int, maximum: int):
        return self._query(
            "SELECT * FROM indicators WHERE parameter_count BETWEEN ? AND ?",
            [minimum, maximum],
        )
