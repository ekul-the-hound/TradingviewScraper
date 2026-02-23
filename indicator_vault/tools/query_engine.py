from __future__ import annotations

from pathlib import Path

import duckdb


class QueryEngine:
    def __init__(self, parquet_path: Path) -> None:
        self.parquet_path = parquet_path

    def _connect(self) -> duckdb.DuckDBPyConnection:
        conn = duckdb.connect(database=":memory:")
        conn.execute(
            """
            CREATE TABLE indicators AS
            SELECT * FROM read_parquet(?);
            """,
            [str(self.parquet_path)],
        )
        return conn

    def volatility_using_ema(self):
        conn = self._connect()
        return conn.execute(
            """
            SELECT * FROM indicators
            WHERE category = 'volatility' AND primitive_vector LIKE '%"ema": 1%'
            """
        ).fetchdf()

    def composites_with_min_params(self, min_params: int = 3):
        conn = self._connect()
        return conn.execute(
            """
            SELECT * FROM indicators
            WHERE category = 'composite' AND parameter_count >= ?
            """,
            [min_params],
        ).fetchdf()

    def similar_to(self, cluster: str, min_score: float = 0.85):
        conn = self._connect()
        return conn.execute(
            """
            SELECT * FROM indicators
            WHERE similarity_cluster = ? AND similarity_score > ?
            """,
            [cluster, min_score],
        ).fetchdf()

    def by_primitive(self, primitive_name: str):
        conn = self._connect()
        return conn.execute(
            """
            SELECT * FROM indicators
            WHERE primitive_vector LIKE ?
            """,
            [f'%"{primitive_name}": 1%'],
        ).fetchdf()

    def by_category(self, category: str):
        conn = self._connect()
        return conn.execute("SELECT * FROM indicators WHERE category = ?", [category]).fetchdf()

    def by_parameter_count(self, min_params: int, max_params: int):
        conn = self._connect()
        return conn.execute(
            """
            SELECT * FROM indicators
            WHERE parameter_count BETWEEN ? AND ?
            """,
            [min_params, max_params],
        ).fetchdf()
