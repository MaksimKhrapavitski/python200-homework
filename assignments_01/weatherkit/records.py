
"""Dataclasses and conversion functions for weather records."""

from dataclasses import dataclass

from weatherkit.schemas import WeatherResponse


@dataclass
class HourlyReading:
    """One hourly observation with temperature in Celsius and precipitation in mm."""

    timestamp: str
    temperature_c: float
    precipitation_mm: float


def to_readings(response: WeatherResponse) -> list[HourlyReading]:
    """Convert validated columnar weather data into hourly records.

    Args:
        response: A validated Open-Meteo API response.

    Returns:
        A list of HourlyReading objects in chronological input order.
    """
    hourly = response.hourly

    return [
        HourlyReading(
            timestamp=hourly.time[i],
            temperature_c=hourly.temperature_2m[i],
            precipitation_mm=hourly.precipitation[i],
        )
        for i in range(len(hourly.time))
    ]


# HourlyReading is a dataclass because the external data has already
# been validated at the boundary by WeatherResponse. Inside the boundary,
# we need a simple object to hold clean data, not another validation model.
