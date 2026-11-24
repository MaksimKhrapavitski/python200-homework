
import json


# Question 1: ModelInfo

class ModelInfo:
    """Read model metadata from a JSON file."""

    def __init__(self, path):
        with open(path, "r") as file:
            self.data = json.load(file)

    def feature_count(self) -> int:
        """Return the number of features."""
        return len(self.data["features"])


# Create a small JSON file
metadata = {
    "features": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max"
    ]
}

with open("model_metadata.json", "w") as file:
    json.dump(metadata, file, indent=4)


# Create ModelInfo object
info = ModelInfo("model_metadata.json")

print("Feature count:", info.feature_count())


# Why is loading the file in __init__ better?
# Because the JSON file is loaded only once when the object
# is created, instead of being loaded every time feature_count()
# is called.




from dataclasses import dataclass


# Question 2: DayForecast

@dataclass
class DayForecast:
    """Store a simple daily weather forecast."""

    date: str
    high_c: float
    will_rain: bool

    def summary(self) -> str:
        """Return a readable weather summary."""
        if self.will_rain:
            rain_status = "rain expected"
        else:
            rain_status = "no rain expected"

        return f"{self.date}: high {self.high_c}C, {rain_status}"


# Create two DayForecast objects
forecast1 = DayForecast("2023-06-15", 24.1, True)
forecast2 = DayForecast("2023-06-16", 27.5, False)

print(forecast1.summary())
print(forecast2.summary())


# What does @dataclass generate automatically?
# It generates methods such as __init__() and __repr__(),
# so we do not need to write them manually.



import pytest


# Question 3: Parse Temperature

def parse_temperature(text: str) -> float:
    """Convert a temperature string to a float."""
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"Invalid temperature: {text}")


print("Parsed temperature:", parse_temperature("24.1"))


def test_parse_temperature_valid():
    """Test a valid temperature string."""
    assert parse_temperature("24.1") == pytest.approx(24.1)


def test_parse_temperature_bad_input_raises():
    """Test that invalid input raises ValueError."""
    with pytest.raises(ValueError, match="temperature"):
        parse_temperature("hot")


# Why use match in pytest.raises()?
# Without match, the test could pass for a ValueError
# with an unexpected message or caused by a different problem.
