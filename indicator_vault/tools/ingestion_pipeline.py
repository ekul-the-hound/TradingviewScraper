from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Protocol

from .ann_index import ANNIndex
from .category_detector import CategoryDetector
from .similarity_engine import IndicatorFeatures, SimilarityEngine
from .storage import IndicatorRecord, IndicatorStorage
from .translator import TranslationPrompts
from .validator import IndicatorValidator


class TranslationModel(Protocol):
    def translate(self, system_prompt: str, user_prompt: str) -> str:
        ...


@dataclass(frozen=True)
class IngestionInput:
    name: str
    pine_code: str
    source_url: str


@dataclass(frozen=True)
class IngestionResult:
    name: str
    category: str
    fingerprint: str
    similarity_score: float
    similar_to: str


class HybridRouter:
    def __init__(self, local_model: TranslationModel, fallback_model: TranslationModel, fallback_ratio: float = 0.2) -> None:
        self.local_model = local_model
        self.fallback_model = fallback_model
        self.fallback_ratio = fallback_ratio

    def select_model(self, script_length: int) -> TranslationModel:
        if script_length > 800 or (script_length % 10) < int(self.fallback_ratio * 10):
            return self.fallback_model
        return self.local_model


class IngestionPipeline:
    def __init__(self, root: Path, local_model: TranslationModel, fallback_model: TranslationModel) -> None:
        self.root = root
        self.prompts = TranslationPrompts()
        self.validator = IndicatorValidator()
        self.similarity = SimilarityEngine()
        self.category_detector = CategoryDetector()
        self.storage = IndicatorStorage(root)
        self.router = HybridRouter(local_model, fallback_model)
        self.ann = ANNIndex(vector_size=9)
        self._features: Dict[str, IndicatorFeatures] = {}

    def ingest_one(self, payload: IngestionInput) -> IngestionResult:
        model = self.router.select_model(len(payload.pine_code))
        user_prompt = self.prompts.build_user_prompt(payload.name, payload.pine_code)
        py_code = model.translate(self.prompts.system_prompt, user_prompt)
        validation = self.validator.validate_python_indicator(py_code)
        if not validation.valid:
            self.storage.append_ingestion_log(payload.name, payload.source_url, f"validation_failed:{validation.message}")
            raise ValueError(validation.message)
        indicator_path = self._indicator_path(payload.name, py_code)
        category_result = self.category_detector.detect(py_code)
        current_features = self.similarity.extract_features(py_code)
        vector = list(current_features.ast_frequency.values())
        ann_neighbors = self.ann.query_by_vector(vector, top_k=10)
        peer_features = [(name, self._features[name]) for name, _ in ann_neighbors if name in self._features]
        similar_to, similarity_score = self.similarity.nearest_family_cluster(current_features, peer_features)
        if not similar_to and self._features:
            similar_to, similarity_score = self.similarity.nearest_family_cluster(current_features, self._features.items())
        self._features[payload.name] = current_features
        self.ann.add(payload.name, vector)
        self.ann.build()
        record = IndicatorRecord(
            name=payload.name,
            fingerprint=current_features.fingerprint,
            category=category_result.category,
            primitive_vector=current_features.primitive_signature,
            parameter_count=current_features.parameter_count,
            similarity_cluster=similar_to,
            similarity_score=similarity_score,
            ingestion_date=date.today().isoformat(),
            source_url=payload.source_url,
        )
        self.storage.upsert_indicator(record, [similar_to] if similar_to else [])
        self.storage.export_similarity_index()
        self.storage.save_vector_store(
            [
                {
                    "name": name,
                    "vector": list(features.ast_frequency.values()),
                    "fingerprint": features.fingerprint,
                }
                for name, features in self._features.items()
            ]
        )
        self.storage.save_fingerprint_index(
            [{"name": name, "fingerprint": features.fingerprint} for name, features in self._features.items()]
        )
        self.storage.append_ingestion_log(payload.name, payload.source_url, "ok")
        return IngestionResult(
            name=payload.name,
            category=category_result.category,
            fingerprint=current_features.fingerprint,
            similarity_score=similarity_score,
            similar_to=similar_to,
        )

    def _indicator_path(self, name: str, code: str) -> Path:
        category = self.category_detector.detect(code).category
        path = self.root / "indicators" / category / f"{name}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code)
        return path
