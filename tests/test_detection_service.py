from __future__ import annotations

import pytest

from config.runtime import get_settings


@pytest.mark.skipif(
    not get_settings().weights_path.exists(),
    reason="GroundingDINO weights are not available.",
)
def test_detection_service_factory_creates_service():
    from src.services.factory import create_detection_service

    service = create_detection_service()
    assert service is not None

