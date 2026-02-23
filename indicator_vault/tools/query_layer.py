from __future__ import annotations

from pathlib import Path

import pandas as pd

from .storage import IndicatorStorage


class QueryLayer:
    def __init__(self, root: Path) -> None:
        self.storage = IndicatorStorage(root)

    def volatility_with_ema(self) -> pd.DataFrame:
        return self.storage.query(
            """
            SELECT * FROM indicators
            WHERE category = 'volatility'
              AND json_extract(primitive_vector, '$.ema') = '1'
            """
        )

    def composites_with_min_parameters(self, min_parameters: int = 3) -> pd.DataFrame:
        return self.storage.query(
            f"SELECT * FROM indicators WHERE category = 'composite' AND parameter_count >= {min_parameters}"
        )

    def similar_to(self, cluster_name: str, minimum_score: float = 0.85) -> pd.DataFrame:
        return self.storage.query(
            f"""
            SELECT * FROM indicators
            WHERE similarity_cluster = '{cluster_name}'
              AND similarity_score > {minimum_score}
            """
        )

    def by_primitive(self, primitive: str) -> pd.DataFrame:
        return self.storage.query(
            f"SELECT * FROM indicators WHERE json_extract(primitive_vector, '$.{primitive}') = '1'"
        )

    def by_category(self, category: str) -> pd.DataFrame:
        return self.storage.query(f"SELECT * FROM indicators WHERE category = '{category}'")

    def by_parameter_count(self, exact_count: int) -> pd.DataFrame:
        return self.storage.query(f"SELECT * FROM indicators WHERE parameter_count = {exact_count}")
