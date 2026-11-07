# Week 2 Mini-Project: Predicting the Daily High

# Task 1: Fetch the weather data.

import requests
import pandas as pd

url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": 35.78,
    "longitude": -78.64,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "America/New_York",
}

response = requests.get(url, params=params, timeout=30)
response.raise_for_status()

df = pd.DataFrame(response.json()["daily"])

df["date"] = pd.to_datetime(df["time"])
df = df.drop("time", axis=1)

# City chosen: Raleigh, North Carolina.

print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())



# Task 2: Explore the weather data.

print("\nDescriptive statistics:")
print(df.describe())

print("\nMissing values in each column:")
print(df.isnull().sum())

correlations = (
    df.select_dtypes(include="number")
    .corr()["temperature_2m_max"]
    .sort_values(ascending=False)
)

print("\nCorrelation with daily high:")
print(correlations)

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

plt.scatter(
    df["temperature_2m_min"],
    df["temperature_2m_max"]
)

plt.title("Daily High vs Low Temperature — Raleigh, 2023")
plt.xlabel("Minimum Temperature (°C)")
plt.ylabel("Maximum Temperature (°C)")

plt.tight_layout()
plt.savefig("outputs/high_vs_low_project.png")
plt.show()

# The minimum temperature has the strongest correlation with the daily high.
# This makes sense because warmer nights are generally associated
# with warmer daytime temperatures.

# Task 3: Engineer a Feature

# Mark summer days: June, July, and August.
df["is_summer"] = df["date"].dt.month.isin([6, 7, 8]).astype(int)

print("\nSummer and non-summer days:")
print(df["is_summer"].value_counts().sort_index())

# I expect is_summer to help predict the daily high because
# summer days are generally warmer than days in other seasons.
# However, the effect may be smaller when minimum temperature
# is already included in the model.


# Task 4: Baseline Model

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Use minimum temperature as the only feature.
X = df[["temperature_2m_min"]].values
y = df["temperature_2m_max"].values

# Split the data into training and test sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the model.
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set.
y_pred = model.predict(X_test)

# Evaluate the model.
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = model.score(X_test, y_test)

print("\nBaseline Model Results:")
print("Slope:", model.coef_[0])
print("RMSE:", rmse)
print("R² on test set:", r2)

# RMSE represents the typical size of prediction errors in degrees Celsius.

# The RMSE is measured in degrees Celsius.
# A baseline RMSE of about 3.16 means that the model's predictions
# typically differ from the actual daily highs by a few degrees Celsius.


# Task 5: Full Model

feature_cols = [
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
    "is_summer",
]

X_full = df[feature_cols].values
y_full = df["temperature_2m_max"].values

X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42
)

model_full = LinearRegression()
model_full.fit(X_train_full, y_train_full)

y_pred_full = model_full.predict(X_test_full)

rmse_full = np.sqrt(mean_squared_error(y_test_full, y_pred_full))
r2_train_full = model_full.score(X_train_full, y_train_full)
r2_test_full = model_full.score(X_test_full, y_test_full)

print("\nFull Model Results:")
print("Train R²:", r2_train_full)
print("Test R²:", r2_test_full)

# The train R² is about 0.866, while the test R² is about 0.891.
# The test score is slightly higher than the training score, so there is no
# concerning train/test gap suggesting overfitting. The model generalizes
# well to the test data, although the small difference may be due to the
# particular train/test split.


print("Test RMSE:", rmse_full)

print("\nFeature coefficients:")
for name, coef in zip(feature_cols, model_full.coef_):
    print(f"{name:22s}: {coef:+.3f}")

# Compare the full model's test R² with the baseline model.
print("\nBaseline test R²:", r2)
print("Full model test R²:", r2_test_full)
print("Difference in test R²:", r2_test_full - r2)

# The minimum temperature has a strong positive relationship
# with the predicted daily high.
# The summer indicator also has a positive coefficient.
# Precipitation and wind speed have negative coefficients.
# These coefficients describe relationships in the model
# and do not necessarily represent causal effects.


# Task 6: Evaluate and Summarize

# Compare predicted and actual daily high temperatures.
plt.figure(figsize=(8, 6))
plt.scatter(y_pred_full, y_test_full, alpha=0.7)

# Draw the diagonal reference line.
plot_min = min(y_test_full.min(), y_pred_full.min())
plot_max = max(y_test_full.max(), y_pred_full.max())
plt.plot(
    [plot_min, plot_max],
    [plot_min, plot_max],
    linestyle="--",
)

plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted Maximum Temperature (°C)")
plt.ylabel("Actual Maximum Temperature (°C)")
plt.tight_layout()
plt.savefig("outputs/predicted_vs_actual_high.png")
plt.show()

# One result that surprised me was that the full model performed slightly
# better on the test data than on the training data. I expected the training
# score to be higher, but the results suggest that the model generalizes well
# to this particular test set.
# Points above the diagonal mean actual temperatures were higher
# than predicted, so the model underestimated the temperature.
# Points below the diagonal mean actual temperatures were lower
# than predicted, so the model overestimated the temperature.

# Summary:
# The dataset contains 365 daily observations, with 73 observations
# in the 20% test set.
# The full model has a test RMSE of about 2.77 degrees Celsius,
# meaning prediction errors are typically a few degrees.
# Its test R² is about 0.891, indicating a good fit on the test data.
# Minimum temperature has the largest coefficient in absolute value
# among the continuous weather features and is strongly related
# to the predicted daily high.
# The summer indicator has a positive coefficient of about 1.045,
# suggesting higher predicted highs during summer when other features
# are held constant.
# Adding the extra features improved test R² and reduced RMSE
# compared with the baseline model.