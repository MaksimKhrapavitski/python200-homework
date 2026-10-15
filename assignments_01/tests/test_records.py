
"""Tests for conversion into HourlyReading objects."""

import pytest

from weatherkit.records import HourlyReading, to_readings
from weatherkit.schemas import WeatherResponse


@pytest.fixture
def sample_response() -> WeatherResponse:
    """Create a small validated weather response."""
    return WeatherResponse.model_validate(
        {
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
                "temperature_2m": [10.0, 12.5, 8.0],
                "precipitation": [0.0, 0.2, 1.5],
            },
        }
    )


def test_readings_preserve_order(
    sample_response: WeatherResponse,
) -> None:
    """Preserve the first and last timestamps."""
    readings = to_readings(sample_response)

    assert len(readings) == 3
    assert readings[0].timestamp == "2026-04-08T00:00"
    assert readings[-1].timestamp == "2026-04-08T02:00"


def test_reading_values_match_input_indices(
    sample_response: WeatherResponse,
) -> None:
    """Keep timestamp, temperature, and precipitation aligned."""
    readings = to_readings(sample_response)

    assert readings[1].timestamp == "2026-04-08T01:00"
    assert readings[1].temperature_c == 12.5
    assert readings[1].precipitation_mm == 0.2


def test_identical_readings_compare_equal() -> None:
    """Dataclass instances with identical fields compare equal."""
    reading_a = HourlyReading(
        "2026-04-08T00:00", 10.0, 0.0
    )
    reading_b = HourlyReading(
        "2026-04-08T00:00", 10.0, 0.0
    )

    assert reading_a == reading_b
