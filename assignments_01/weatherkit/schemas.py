
"""Pydantic schemas for validating the Open-Meteo API response."""

from pydantic import BaseModel, Field, model_validator


class HourlyBlock(BaseModel):
    """Columnar hourly weather measurements returned by the API."""

    time: list[str]
    temperature_2m: list[float]
    precipitation: list[float]

    @model_validator(mode="after")
    def check_equal_lengths(self) -> "HourlyBlock":
        """Ensure all hourly measurement lists have equal lengths."""
        lengths = {
            len(self.time),
            len(self.temperature_2m),
            len(self.precipitation),
        }

        if len(lengths) != 1:
            raise ValueError(
                "time, temperature_2m, and precipitation "
                "must all have the same length"
            )

        return self


class WeatherResponse(BaseModel):
    """Validated top-level response from the Open-Meteo weather API."""

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    timezone: str
    elevation: float
    hourly: HourlyBlock
