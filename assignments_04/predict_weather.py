
from pathlib import Path

from weather_model import WeatherClassifier


MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "weather_classifier.pkl"
)


def main():
    """Run weather predictions for sample days."""
    classifier = WeatherClassifier(MODEL_PATH)

    new_days = [
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
    ]

    results = classifier.predict(new_days)

    for day, result in zip(new_days, results):
        print(
            f"Prediction: {result.label}, "
            f"Probability: {result.probability:.2%}"
        )


if __name__ == "__main__":
    main()
