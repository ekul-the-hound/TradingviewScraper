from __future__ import annotations

from pathlib import Path

import pandas as pd

from .clustering import write_family_graph


def bootstrap_repository(root: Path) -> None:
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    (root / "embeddings").mkdir(parents=True, exist_ok=True)
    (root / "clusters").mkdir(parents=True, exist_ok=True)
    (root / "fingerprints").mkdir(parents=True, exist_ok=True)
    if not (root / "metadata" / "index.json").exists():
        (root / "metadata" / "index.json").write_text('{"name_to_fingerprint": {}, "fingerprint_to_similar": {}}')
    pd.DataFrame([], columns=["name", "fingerprint", "similarity_cluster", "similarity_score", "category", "ingestion_date"]).to_parquet(
        root / "metadata" / "similarity_index.parquet", index=False
    )
    pd.DataFrame([], columns=["name", "vector", "fingerprint"]).to_parquet(root / "embeddings" / "vector_store.parquet", index=False)
    pd.DataFrame([], columns=["name", "fingerprint"]).to_parquet(root / "fingerprints" / "fingerprint_index.parquet", index=False)
    write_family_graph([], root / "clusters" / "family_graph.graphml")
