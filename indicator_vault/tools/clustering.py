from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import networkx as nx


def write_family_graph(edges: Iterable[Tuple[str, str, float]], output_path: Path) -> None:
    graph = nx.Graph()
    for left, right, score in edges:
        graph.add_edge(left, right, weight=score)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, output_path)
