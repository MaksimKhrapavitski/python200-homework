
"""Tests for weather API response validation."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from weatherkit.schemas import WeatherResponse


def test_real_weather_response() -> None:
    """Validate the course JSON and confirm all 168 observations exist."""
    # Resolve the path from this test file, not the terminal's working directory.
    # A plain relative path may fail when pytest runs from another directory.
    json_path = Path(__file__).parent.parent / "weather_raw.json"

    with json_path.open(encoding="utf-8") as file:
        raw_data = json.load(file)

    response = WeatherResponse.model_validate(raw_data)

    assert len(response.hourly.time) == 168


def test_invalid_latitude_raises() -> None:
    """Reject latitude outside the valid geographic range."""
    data = {
        "latitude": 200.0,
        "longitude": -80.0,
        "timezone": "GMT",
        "elevation": 200.0,
        "hourly": {
            "time": ["2026-04-08T00:00"],
            "temperature_2m": [10.0],
            "precipitation": [0.0],
        },
    }

    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(data)


def test_mismatched_hourly_lengths_raise() -> None:
    """Reject parallel hourly lists with different lengths."""
    data = {
        "latitude": 35.2,
        "longitude": -80.8,
        "timezone": "GMT",
        "elevation": 200.0,
        "hourly": {
            "time": [
                "2026-04-08T00:00",
                "2026-04-08T01:00",
                "2026-04-08T02:00",
            ],
            "temperature_2m": [10.0, 11.0],
            "precipitation": [0.0, 0.1, 0.0],
        },
    }

    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(data)


def test_null_temperature_raises() -> None:
    """Reject null in a list of floating-point temperatures."""
    data = {
        "latitude": 35.2,
        "longitude": -80.8,
        "timezone": "GMT",
        "elevation": 200.0,
        "hourly": {
            "time": [
                "2026-04-08T00:00",
                "2026-04-08T01:00",
            ],
            "temperature_2m": [10.0, None],
            "precipitation": [0.0, 0.1],
        },
    }

    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(data)
