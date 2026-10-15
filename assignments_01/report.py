"""Load weather data, validate it, and print daily summaries."""

import json
from pathlib import Path

from weatherkit import WeatherResponse, to_readings, DailyAggregator


def main() -> None:
    """Load, validate, aggregate, and report weather observations."""
    # Step 1: Load the raw JSON file.
    data_path = Path(__file__).parent / "weather_raw.json"

    with data_path.open(encoding="utf-8") as file:
        raw_data = json.load(file)

    # Step 2: Validate the API response.
    response = WeatherResponse.model_validate(raw_data)

    # Step 3: Convert columnar data into HourlyReading objects.
    readings = to_readings(response)

    # Step 4: Aggregate hourly readings into daily summaries.
    aggregator = DailyAggregator(min_hours=24)
    summaries = aggregator.summarize(readings)

    # Step 5: Print the daily summary table.
    print(f"Location: {response.latitude}, {response.longitude}")
    print(f"Timezone: {response.timezone}")
    print()

    print(
        f"{'Date':<12}"
        f"{'High (°C)':>12}"
        f"{'Low (°C)':>12}"
        f"{'Precip. (mm)':>16}"
        f"{'Range (°C)':>13}"
    )
    print("-" * 65)

    for summary in summaries:
        print(
            f"{summary.date:<12}"
            f"{summary.temp_max:>12.1f}"
            f"{summary.temp_min:>12.1f}"
            f"{summary.precipitation_sum:>16.1f}"
            f"{summary.temp_range():>13.1f}"
        )

    # Step 6: Warn about days excluded because they had insufficient data.
    incomplete = aggregator.incomplete_days(readings)

    if incomplete:
        print()
        print(
            "WARNING: Incomplete days dropped: "
            + ", ".join(incomplete)
        )
    else:
        print()
        print("All days have enough hourly observations.")

    # Importing this module will not execute main() because of the guard below.
    # Without the guard, importing report.py would run the entire pipeline,
    # print the report, and potentially perform other unintended work.


if __name__ == "__main__":
    main()


# Task 7: Reflection
#
# 1. Rejecting the whole response when one temperature is null is useful
#    when complete, reliable measurements are required for a calculation
#    or model. However, for a monitoring dashboard, one missing measurement
#    should not necessarily discard the other 167 observations.
#    To tolerate missing temperatures, change temperature_2m from
#    list[float] to list[float | None]. Then decide how to handle None
#    values when converting and summarizing readings.
#
# 2. If the pipeline runs at noon, the current day may have only 12
#    observations. With min_hours=24, that day is dropped because it is
#    incomplete. incomplete_days() identifies the dropped date so the
#    pipeline can report the missing data instead of silently omitting it.
#
# 3. A package makes it possible for a later pipeline to import and reuse
#    HourlyReading, to_readings, and DailyAggregator without copying their
#    implementation into another script. The package can also be tested
#    independently.