from __future__ import annotations

from typing import Dict, Iterable, Optional

from src.services.detection_service import DetectionService


class DetectionServiceManager:
    """Registry that maps model keys to detection service instances."""

    def __init__(
        self,
        *,
        services: Dict[str, DetectionService],
        default_model: str,
        aliases: Optional[Dict[str, Iterable[str]]] = None,
    ) -> None:
        if not services:
            raise ValueError("At least one detection service must be provided.")

        normalized = {self._normalize_key(key): value for key, value in services.items()}
        self._services = normalized

        alias_mapping: Dict[str, str] = {}
        if aliases:
            for canonical, alias_list in aliases.items():
                canonical_key = self._normalize_key(canonical)
                if canonical_key not in normalized:
                    continue
                for alias in alias_list:
                    alias_mapping[self._normalize_key(alias)] = canonical_key
        self._aliases = alias_mapping

        default_key = self._normalize_key(default_model)
        if default_key not in normalized:
            raise ValueError(f"Default model '{default_model}' is not registered.")
        self._default_key = default_key

    @staticmethod
    def _normalize_key(key: str) -> str:
        return key.strip().lower()

    def resolve(self, model_name: Optional[str] = None) -> DetectionService:
        """Return the detection service for the requested model key."""
        if model_name is None:
            return self._services[self._default_key]

        key = self._normalize_key(model_name)
        if key in self._services:
            return self._services[key]
        if key in self._aliases:
            resolved = self._aliases[key]
            return self._services[resolved]

        available = ", ".join(sorted(self._services.keys()))
        raise KeyError(
            f"Unknown detection model '{model_name}'. Available models: {available}"
        )

    def available_models(self) -> Dict[str, DetectionService]:
        """Expose the registered services (read-only)."""
        return dict(self._services)

    @property
    def default_model(self) -> str:
        return self._default_key
