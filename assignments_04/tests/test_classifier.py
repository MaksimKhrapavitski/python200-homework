import joblib
import pandas as pd
import pytest

from sklearn.tree import DecisionTreeClassifier

from weather_model import WeatherClassifier

@pytest.fixture
def classifier(tmp_path):
    """Create and save a small model for testing."""

    model_path = tmp_path / "test_weather_model.pkl"

    # Create a small training dataset
    X = pd.DataFrame([
        {
            "temperature_2m_max": 18,
            "temperature_2m_min": 10,
            "precipitation_sum": 0.0,
            "wind_speed_10m_max": 12,
        },
        {
            "temperature_2m_max": 18,
            "temperature_2m_min": 10,
            "precipitation_sum": 0.5,
            "wind_speed_10m_max": 12,
        },
        {
            "temperature_2m_max": 18,
            "temperature_2m_min": 10,
            "precipitation_sum": 5.0,
            "wind_speed_10m_max": 12,
        },
        {
            "temperature_2m_max": 18,
            "temperature_2m_min": 10,
            "precipitation_sum": 10.0,
            "wind_speed_10m_max": 12,
        },
        {
            "temperature_2m_max": 18,
            "temperature_2m_min": 10,
            "precipitation_sum": 15.0,
            "wind_speed_10m_max": 12,
        },
    ])

    y = [1, 1, 1, 0, 0]

    # Train a small classifier
    model = DecisionTreeClassifier(max_depth=1, random_state=42)
    model.fit(X, y)

    # Save the model to the temporary directory
    joblib.dump(model, model_path)

    return WeatherClassifier(model_path)

def test_three_day_predictions_keep_order(classifier):
    """Predictions should remain in the same order as input days."""

    days = [
        {
            "temperature_2m_max": 18,
            "temperature_2m_min": 10,
            "precipitation_sum": 0.0,
            "wind_speed_10m_max": 12,
        },
        {
            "temperature_2m_max": 18,
            "temperature_2m_min": 10,
            "precipitation_sum": 15.0,
            "wind_speed_10m_max": 12,
        },
        {
            "temperature_2m_max": 18,
            "temperature_2m_min": 10,
            "precipitation_sum": 0.5,
            "wind_speed_10m_max": 12,
        },
    ]

    results = classifier.predict(days)

    assert len(results) == 3
    assert [result.label for result in results] == [
        "good",
        "skip",
        "good",
    ]


def test_missing_model_file_raises_error(tmp_path):
    """A missing model file should raise FileNotFoundError."""

    missing_path = tmp_path / "missing_model.pkl"

    with pytest.raises(FileNotFoundError, match="No model file"):
        WeatherClassifier(missing_path)


@pytest.mark.parametrize(
    "precipitation, expected_label",
    [
        (0.0, "good"),
        (15.0, "skip"),
    ],
)
def test_predictions_with_different_rainfall(
    classifier, precipitation, expected_label
):
    """Predictions should reflect different rainfall conditions."""

    day = {
        "temperature_2m_max": 18,
        "temperature_2m_min": 10,
        "precipitation_sum": precipitation,
        "wind_speed_10m_max": 12,
    }

    result = classifier.predict([day])

    assert result[0].label == expected_label

def test_empty_input_returns_empty_list(classifier):
    """An empty input should return an empty list."""

    result = classifier.predict([])

    assert result == []