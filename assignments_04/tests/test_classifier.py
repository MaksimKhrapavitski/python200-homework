
from pathlib import Path

import pytest

from weather_model import WeatherClassifier, Prediction


# Path to the saved model
MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "weather_classifier.pkl"
)


@pytest.fixture(scope="module")
def classifier():
    """Create one classifier for all tests in this module."""
    return WeatherClassifier(MODEL_PATH)



def test_good_day_is_labeled_good(classifier):
    """A clear, dry day should be good for running."""
    result = classifier.predict([
        {
            "temperature_2m_max": 18,
            "temperature_2m_min": 10,
            "precipitation_sum": 0.0,
            "wind_speed_10m_max": 12,
        }
    ])

    assert result[0].label == "good"


def test_rainy_day_is_labeled_skip(classifier):
    """A very rainy day should be labeled skip."""
    result = classifier.predict([
        {
            "temperature_2m_max": 18,
            "temperature_2m_min": 10,
            "precipitation_sum": 15.0,
            "wind_speed_10m_max": 12,
        }
    ])

    assert result[0].label == "skip"


def test_probability_is_between_zero_and_one(classifier):
    """Prediction probability must be between 0 and 1."""
    result = classifier.predict([
        {
            "temperature_2m_max": 18,
            "temperature_2m_min": 10,
            "precipitation_sum": 0.0,
            "wind_speed_10m_max": 12,
        }
    ])

    assert 0.0 <= result[0].probability <= 1.0


def test_missing_feature_raises_value_error(classifier):
    """Missing required features should raise ValueError."""
    with pytest.raises(ValueError, match="missing required features"):
        classifier.predict([
            {
                "temperature_2m_max": 18,
                "temperature_2m_min": 10,
                "precipitation_sum": 0.0,
                # wind_speed_10m_max is missing
            }
        ])


def test_empty_input_returns_empty_list(classifier):
    """An empty input list should return an empty list."""
    result = classifier.predict([])

    assert result == []
