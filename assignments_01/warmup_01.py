
# --- Classes ---

# Q1: Thermometer class

class Thermometer:
    def __init__(
        self,
        location: str,
        readings: list[float] | None = None
    ):
        """Store a location and a list of temperature readings in Celsius."""
        self.location = location
        self.readings = readings if readings is not None else []

    def add(self, reading: float) -> None:
        """Add a temperature reading."""
        self.readings.append(reading)

    def average(self) -> float | None:
        """Return the average temperature, or None if there are no readings."""
        if not self.readings:
            return None
        return sum(self.readings) / len(self.readings)

    def hottest(self) -> float | None:
        """Return the highest temperature, or None if there are no readings."""
        if not self.readings:
            return None
        return max(self.readings)

    # Q2: Readable object representation
    def __repr__(self) -> str:
        """Return a useful representation of the thermometer."""
        return (
            f"Thermometer(location={self.location!r}, "
            f"n_readings={len(self.readings)}, "
            f"average={self.average()})"
        )


# Q1: Create a thermometer and add four readings
my_thermometer = Thermometer("Raleigh")
my_thermometer.add(18.0)
my_thermometer.add(20.0)
my_thermometer.add(17.0)
my_thermometer.add(19.0)

print("--- Question 1 ---")
print("Average:", my_thermometer.average())
print("Hottest:", my_thermometer.hottest())

empty_thermometer = Thermometer("Empty location")

print("Empty average:", empty_thermometer.average())
print("Empty hottest:", empty_thermometer.hottest())

# average() must handle an empty list because dividing by zero
# raises ZeroDivisionError. Without the check, it would crash.


# Q2: Print the object and a list containing two thermometers
another_thermometer = Thermometer("Charlotte", [21.0, 23.0])

print("\n--- Question 2 ---")
print(my_thermometer)
print([my_thermometer, another_thermometer])

# Without __repr__, Python displays a default representation
# containing the class name and a memory address, which is
# not useful for understanding the object's data while debugging.


# Q3: TemperatureAlert class

class TemperatureAlert:
    def __init__(self, threshold: float = 30.0):
        """Store the temperature threshold for an alert."""
        self.threshold = threshold

    def breaches(self, thermometer: Thermometer) -> list[float]:
        """Return readings strictly above the threshold."""
        return [
            reading
            for reading in thermometer.readings
            if reading > self.threshold
        ]


# Check the same thermometer using two different thresholds
alert_30 = TemperatureAlert(30.0)
alert_19 = TemperatureAlert(19.0)

print("\n--- Question 3 ---")
print("Above 30°C:", alert_30.breaches(my_thermometer))
print("Above 19°C:", alert_19.breaches(my_thermometer))

# The threshold belongs to TemperatureAlert because it defines
# the alert's rule. We can reuse the same alert for twenty
# thermometers without passing the threshold every time.



# --- Dataclasses, Type Hints, and Docstrings ---

from dataclasses import dataclass, field, FrozenInstanceError


# Q1: Rewrite Station as a dataclass

@dataclass(frozen=True)
class Station:
    """Represent a weather station and its geographic information."""

    station_id: str
    name: str
    latitude: float
    longitude: float
    elevation: float


station_a = Station("NC001", "Raleigh", 35.78, -78.64, 96.0)
station_b = Station("NC001", "Raleigh", 35.78, -78.64, 96.0)

print("\n--- Dataclass Question 1 ---")
print("Station A:", station_a)
print("Station B:", station_b)
print("Stations equal:", station_a == station_b)

# A dataclass automatically generates __eq__, so objects with
# identical field values compare as equal. The original hand-written
# class would return False because the objects are different instances.


# Q2: Frozen dataclass

print("\n--- Dataclass Question 2 ---")

# Attempt to modify a frozen Station
try:
    station_a.name = "Charlotte"
except FrozenInstanceError as error:
    print("FrozenInstanceError:", error)

# Create a third station with different field values
station_c = Station("NC002", "Durham", 35.99, -78.90, 123.0)

# Two stations are identical, so the set contains only two unique stations
stations = {station_a, station_b, station_c}

print("Unique stations:", len(stations))

# frozen=True prevents field reassignment and allows the dataclass
# to be hashable when all its fields are hashable. This lets us use
# Station objects in a set and automatically eliminate duplicates.


# Q3: StationBatch dataclass

# Intentionally incorrect version:
#
# @dataclass
# class StationBatch:
#     region: str
#     stations: list[Station] = []
#
# Error:
# ValueError: mutable default <class 'list'> for field stations
# is not allowed: use default_factory
#
# Python rejects this because a mutable default list could be shared
# between instances. Use field(default_factory=list) instead.


@dataclass
class StationBatch:
    """Represent a collection of weather stations in a region."""

    region: str
    stations: list[Station] = field(default_factory=list)

    def add(self, station: Station) -> None:
        """Add a station to the batch."""
        self.stations.append(station)

    def highest(self) -> Station | None:
        """Return the station with the highest elevation, or None if empty."""
        if not self.stations:
            return None

        return max(self.stations, key=lambda station: station.elevation)


print("\n--- Dataclass Question 3 ---")

batch = StationBatch("North Carolina")
batch.add(station_a)
batch.add(station_c)

print("Batch:", batch)
print("Highest station:", batch.highest())

empty_batch = StationBatch("Empty Region")
print("Highest in empty batch:", empty_batch.highest())


# --- Pydantic ---

from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    ConfigDict,
    model_validator,
)


# Q1: Define the Reading model

class Reading(BaseModel):
    """Represent a validated weather reading from a station."""

    station_id: str = Field(min_length=3)
    timestamp: str
    temperature_c: float = Field(ge=-90, le=60)
    humidity: float = Field(ge=0, le=100)

    # Q4: Validate a combination of fields
    @model_validator(mode="after")
    def check_sensor_failure(self) -> "Reading":
        """Reject a combination of values that indicates sensor failure."""
        if self.humidity == 0.0 and self.temperature_c < -40:
            raise ValueError(
                "Humidity of 0% with temperature below -40°C "
                "indicates a possible sensor failure."
            )
        return self


print("\n--- Pydantic Question 1 ---")

valid_reading = Reading(
    station_id="NC001",
    timestamp="2026-10-13T12:00:00",
    temperature_c=21.5,
    humidity=45.0,
)

print(valid_reading)


# Q2: Show three separate validation failures

print("\n--- Pydantic Question 2 ---")

# Failure 1: Missing required field
try:
    Reading(
        station_id="NC001",
        temperature_c=21.5,
        humidity=45.0,
    )
except ValidationError as error:
    print("\nFailure 1: Missing timestamp")
    print(error)


# Failure 2: Temperature outside the allowed range
try:
    Reading(
        station_id="NC001",
        timestamp="2026-10-13T12:00:00",
        temperature_c=150.0,
        humidity=45.0,
    )
except ValidationError as error:
    print("\nFailure 2: Invalid temperature")
    print(error)


# Failure 3: Humidity cannot be converted to a number
try:
    Reading(
        station_id="NC001",
        timestamp="2026-10-13T12:00:00",
        temperature_c=21.5,
        humidity="very humid",
    )
except ValidationError as error:
    print("\nFailure 3: Invalid humidity")
    print(error)


# Pydantic can convert a numeric string such as "21.5" to float.
# It rejects "very humid" because that text cannot be converted
# to a number. Coercion works only when conversion is possible.

coerced_reading = Reading(
    station_id="NC001",
    timestamp="2026-10-13T12:00:00",
    temperature_c="21.5",
    humidity=40,
)

print("\nCoerced reading:", coerced_reading)
print("Temperature type:", type(coerced_reading.temperature_c))
print("Humidity type:", type(coerced_reading.humidity))


# Q3: Report several errors in one validation attempt

print("\n--- Pydantic Question 3 ---")

try:
    Reading(
        station_id="A1",
        # timestamp is intentionally missing
        temperature_c="not a number",
        humidity=50.0,
    )
except ValidationError as error:
    print("Validation errors:")
    for item in error.errors():
        print(f"Location: {item['loc']}, Message: {item['msg']}")

    print("Total errors:", len(error.errors()))

# Pydantic reports all independent field errors in one attempt.
# This helps us fix several problems at once instead of rerunning
# the program repeatedly after discovering each error.


# Q4: Validate a combination of fields

print("\n--- Pydantic Question 4 ---")

# A valid reading still constructs successfully.
print("Valid:", valid_reading)

# This combination passes the individual Field constraints,
# but the model-level rule rejects it.
try:
    Reading(
        station_id="NC001",
        timestamp="2026-10-13T12:00:00",
        temperature_c=-45.0,
        humidity=0.0,
    )
except ValidationError as error:
    print("Sensor failure detected:")
    print(error)

# Field constraints validate individual values independently.
# model_validator is needed because this rule depends on the
# relationship between humidity and temperature_c.

# --- pytest ---

import pytest


# Q1: Celsius to Fahrenheit

def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert a temperature from Celsius to Fahrenheit."""
    return (celsius * 9 / 5) + 32


def test_celsius_to_fahrenheit():
    """Test Celsius-to-Fahrenheit conversion."""
    assert celsius_to_fahrenheit(0) == 32
    assert celsius_to_fahrenheit(100) == 212
    assert celsius_to_fahrenheit(37) == pytest.approx(98.6)


# pytest.approx is needed because floating-point arithmetic can
# produce a tiny precision difference from the expected decimal.

#================================================================== FAILURES ==================================================================
#_________________________________________________________ test_celsius_to_fahrenheit _________________________________________________________
#
#   def test_celsius_to_fahrenheit():
#        """Test Celsius-to-Fahrenheit conversion."""
#       assert celsius_to_fahrenheit(0) == 32
#       assert celsius_to_fahrenheit(100) == 212
#E       assert 257.0 == 212
#E        +  where 257.0 = celsius_to_fahrenheit(100)

#warmup_01.py:372: AssertionError
#========================================================== short test summary info ===========================================================
#FAILED warmup_01.py::test_celsius_to_fahrenheit - assert 257.0 == 212
#======================================================== 1 failed, 5 passed in 0.25s =========================================================

# Q2: Mean and error handling

def mean(values: list[float]) -> float:
    """Return the arithmetic mean of a non-empty list."""
    if not values:
        raise ValueError("Cannot calculate mean of an empty list")

    return sum(values) / len(values)


def test_mean_of_empty_raises():
    """Verify that an empty list raises the expected error."""
    with pytest.raises(ValueError, match="empty"):
        mean([])


# pytest.raises(ValueError) alone checks only the exception type.
# match="empty" also checks that the message contains the expected word.

# Q3: Parametrized tests

@pytest.mark.parametrize(
    "values, expected",
    [
        ([1.0, 2.0, 3.0], 2.0),
        ([5.0], 5.0),
        ([-2.0, -4.0], -3.0),
        ([0.0, 10.0, 20.0], 10.0),
    ],
)
def test_mean_values(values, expected):
    """Test mean with multiple input/output pairs."""
    assert mean(values) == pytest.approx(expected)


# Parametrization avoids duplicated test code, makes each case
# independently visible in pytest output, and simplifies maintenance.