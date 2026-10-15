
"""Tests for daily weather aggregation."""

import pytest

from weatherkit.records import HourlyReading
from weatherkit.summarize import DailyAggregator, DailySummary


@pytest.fixture
def sample_readings() -> list[HourlyReading]:
    """Provide hourly readings spanning two calendar dates."""
    return [
        HourlyReading("2026-04-08T00:00", 10.0, 0.2),
        HourlyReading("2026-04-08T01:00", 16.0, 0.3),
        HourlyReading("2026-04-08T02:00", 8.0, 0.5),
        HourlyReading("2026-04-09T00:00", 12.0, 0.1),
        HourlyReading("2026-04-09T01:00", 5.0, 0.4),
    ]


def test_grouping_creates_two_summaries(
    sample_readings: list[HourlyReading],
) -> None:
    """Readings from two dates produce two daily summaries."""
    aggregator = DailyAggregator(min_hours=1)

    summaries = aggregator.summarize(sample_readings)

    assert len(summaries) == 2
    assert [s.date for s in summaries] == ["2026-04-08", "2026-04-09"]


def test_temperature_max_and_min(
    sample_readings: list[HourlyReading],
) -> None:
    """Calculate the maximum and minimum temperature for a day."""
    summaries = DailyAggregator(min_hours=1).summarize(sample_readings)

    first_day = summaries[0]
    assert first_day.temp_max == 16.0
    assert first_day.temp_min == 8.0
    assert first_day.temp_range() == 8.0


def test_precipitation_sum(
    sample_readings: list[HourlyReading],
) -> None:
    """Add precipitation values for the same day."""
    summaries = DailyAggregator(min_hours=1).summarize(sample_readings)

    assert summaries[0].precipitation_sum == pytest.approx(1.0)
    assert summaries[1].precipitation_sum == pytest.approx(0.5)


def test_incomplete_day_is_dropped(
    sample_readings: list[HourlyReading],
) -> None:
    """Exclude days with fewer than the required observations."""
    aggregator = DailyAggregator(min_hours=3)

    summaries = aggregator.summarize(sample_readings)

    assert [s.date for s in summaries] == ["2026-04-08"]
    assert aggregator.incomplete_days(sample_readings) == ["2026-04-09"]


def test_lower_min_hours_keeps_same_day(
    sample_readings: list[HourlyReading],
) -> None:
    """Lowering the minimum allows the shorter day to be included."""
    aggregator = DailyAggregator(min_hours=2)

    summaries = aggregator.summarize(sample_readings)

    assert len(summaries) == 2
    assert summaries[1].hours_observed == 2


@pytest.mark.parametrize(
    ("temp_max", "temp_min", "expected_range"),
    [
        (16.0, 8.0, 8.0),
        (10.0, 10.0, 0.0),
        (5.0, -5.0, 10.0),
        (0.0, -10.0, 10.0),
    ],
)
def test_temp_range(
    temp_max: float,
    temp_min: float,
    expected_range: float,
) -> None:
    """Calculate temperature range for different values."""
    summary = DailySummary(
        date="2026-04-08",
        temp_max=temp_max,
        temp_min=temp_min,
        precipitation_sum=0.0,
        hours_observed=24,
    )

    assert summary.temp_range() == expected_range
