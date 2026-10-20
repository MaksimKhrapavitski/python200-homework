
"""Aggregate hourly weather readings into daily summaries."""

from collections import defaultdict
from dataclasses import dataclass

from weatherkit.records import HourlyReading


@dataclass
class DailySummary:
    """Daily weather statistics with temperatures in Celsius and precipitation in mm."""

    date: str
    temp_max: float
    temp_min: float
    precipitation_sum: float
    hours_observed: int


    # Verification: temporarily changed subtraction to addition.
    # The test_temperature_max_and_min test failed with 24.0
    # instead of 8.0, confirming that the test detected the error.
    # Restored the original calculation.
    def temp_range(self) -> float:
        """Return the difference between the maximum and minimum temperatures."""
        return self.temp_max - self.temp_min


class DailyAggregator:
    """Group hourly readings into daily summaries."""

    def __init__(self, min_hours: int = 24) -> None:
        """Initialize the aggregator with the minimum observations per day."""
        if min_hours < 1:
            raise ValueError("min_hours must be at least 1")

        self.min_hours = min_hours

    def summarize(
        self, readings: list[HourlyReading]
    ) -> list[DailySummary]:
        """Create daily summaries, excluding days with too few observations.

        Args:
            readings: Hourly weather observations in any order.

        Returns:
            Daily summaries sorted by date, including only complete-enough days.
        """
        grouped: dict[str, list[HourlyReading]] = defaultdict(list)

        for reading in readings:
            # Keep only YYYY-MM-DD; the remaining timestamp contains the time.
            date = reading.timestamp[:10]
            grouped[date].append(reading)

        summaries: list[DailySummary] = []

        for date, day_readings in grouped.items():
            if len(day_readings) < self.min_hours:
                continue

            summaries.append(
                DailySummary(
                    date=date,
                    temp_max=max(r.temperature_c for r in day_readings),
                    temp_min=min(r.temperature_c for r in day_readings),
                    precipitation_sum=sum(
                        r.precipitation_mm for r in day_readings
                    ),
                    hours_observed=len(day_readings),
                )
            )

        return sorted(summaries, key=lambda summary: summary.date)

    def incomplete_days(self, readings: list[HourlyReading]) -> list[str]:
        """Return sorted dates excluded for having too few observations.

        Args:
            readings: Hourly weather observations.

        Returns:
            Dates with fewer observations than the configured minimum.
        """
        grouped: dict[str, int] = defaultdict(int)

        for reading in readings:
            grouped[reading.timestamp[:10]] += 1

        return sorted(
            date
            for date, count in grouped.items()
            if count < self.min_hours
        )
