from __future__ import annotations

import json

import pytest

from qc_pipeline.gemini_validator import GeminiValidator


def test_parse_candidate_extracts_expected_fields() -> None:
    validator = GeminiValidator(api_key="test-key")
    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": json.dumps(
                                {
                                    "prediction_label": "traffic_light",
                                    "confidence": 0.87,
                                    "rationale": "Object resembles a traffic light.",
                                }
                            )
                        }
                    ]
                }
            }
        ]
    }

    response = validator._parse_candidate(payload)
    assert response.prediction_label == "traffic_light"
    assert response.confidence == pytest.approx(0.87)
    assert response.rationale == "Object resembles a traffic light."


def test_parse_candidate_rejects_invalid_json() -> None:
    validator = GeminiValidator(api_key="test-key")
    payload = {"candidates": [{"content": {"parts": [{"text": "not json"}]}}]}
    with pytest.raises(ValueError):
        validator._parse_candidate(payload)

