from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RoutingDecision:
    provider: str
    reason: str


class CostEfficientRouter:
    def __init__(self, local_failure_rate_threshold: float = 0.2) -> None:
        self.local_failure_rate_threshold = local_failure_rate_threshold

    def route(self, complexity_score: float, rolling_local_failure_rate: float) -> RoutingDecision:
        if complexity_score <= 0.65 and rolling_local_failure_rate <= self.local_failure_rate_threshold:
            return RoutingDecision(provider="local_ollama", reason="cheap_primary")
        return RoutingDecision(provider="minimax", reason="fallback_for_quality")
