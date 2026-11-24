
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

    # Four sample days with different weather conditions
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
        {
            "temperature_2m_max": 18,
            "temperature_2m_min": 10,
            "precipitation_sum": 5.0,
            "wind_speed_10m_max": 12,
        },
        {
            "temperature_2m_max": 8,
            "temperature_2m_min": 1,
            "precipitation_sum": 2.0,
            "wind_speed_10m_max": 30,
        },
    ]

    results = classifier.predict(new_days)

    for index, (day, result) in enumerate(zip(new_days, results), start=1):
        print(
            f"Day {index}: "
            f"Max temp: {day['temperature_2m_max']}°C, "
            f"Min temp: {day['temperature_2m_min']}°C, "
            f"Rain: {day['precipitation_sum']} mm, "
            f"Wind: {day['wind_speed_10m_max']} km/h | "
            f"Prediction: {result.label}, "
            f"Probability: {result.probability:.2%}"
        )


# This guard ensures main() runs only when this file is executed directly.
# It prevents the script from running automatically when another module imports it.
if __name__ == "__main__":
    main()



# Task 6: Reflection
#
# 1. Selecting the four features in a fixed order prevents a bug where
#    feature values could be assigned to the wrong columns if the caller
#    provides them in a different order. This could lead to incorrect
#    predictions.
#
# 2. Training a tiny model inside a fixture makes tests independent of
#    the real saved model file. Tests become more reliable and repeatable
#    because they do not depend on an external model artifact or its
#    availability. It also allows tests to use controlled data.
#
# 3. WeatherClassifier is ready for a pipeline because it can load the
#    model once and predict multiple weather records in a batch. It also
#    checks for missing features and returns predictions in the same
#    order as the input records.
#    Before production, I would add stronger input validation to handle
#    invalid or unrealistic weather values.
