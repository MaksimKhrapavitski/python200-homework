
import requests
import pandas as pd

# --- Task 1: Fetch the Data ---

# I chose Raleigh, North Carolina, USA.
url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": 35.7796,
    "longitude": -78.6382,
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

response = requests.get(url, params=params)
response.raise_for_status()

df = pd.DataFrame(response.json()["daily"])

df["date"] = pd.to_datetime(df["time"])
df = df.drop("time", axis=1)

print("Dataset shape:", df.shape)
print("First five rows:")
print(df.head())



# --- Task 2: Engineer the Label ---

df["good_for_running"] = (
    (df["temperature_2m_max"] >= 7) &
    (df["temperature_2m_max"] <= 26) &
    (df["temperature_2m_min"] >= 0) &
    (df["precipitation_sum"] < 3) &
    (df["wind_speed_10m_max"] < 30)
).astype(int)

good_days = (df["good_for_running"] == 1).sum()
skip_days = (df["good_for_running"] == 0).sum()

total_days = len(df)
good_fraction = good_days / total_days

print("\nClass balance:")
print("Good days:", good_days)
print("Skip days:", skip_days)
print("Fraction of good days:", round(good_fraction, 3))

print("\nFirst five rows with labels:")
print(df.head())

# Raleigh has a relatively mild climate, so I expect a reasonable
# number of good running days throughout the year.
# About 39.7% of days are good for running.
# This seems reasonable for Raleigh because the criteria are relatively
# strict and exclude days with heavy rain, strong winds, or extreme temperatures.
# Since the classes are moderately balanced, accuracy is useful,
# but precision, recall, and F1-score are also important.


# --- Task 3: Build and Evaluate a KNN Classifier ---

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

numeric = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]

X = df[numeric]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

k_values = [3, 5, 7, 9, 11, 15, 21]
cv_results = {}

for k in k_values:
    knn_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k)),
        ]
    )

    scores = cross_val_score(
        knn_pipeline, X_train, y_train, cv=5, scoring="accuracy"
    )

    cv_results[k] = scores.mean()
    print(f"k={k}: Mean CV Accuracy = {scores.mean():.4f}")

best_k = max(cv_results, key=cv_results.get)

print("Best k:", best_k)

best_knn = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=best_k)),
    ]
)

best_knn.fit(X_train, y_train)

y_pred_knn = best_knn.predict(X_test)

print("KNN Test Accuracy:", accuracy_score(y_test, y_pred_knn))
print("KNN Classification Report:")
print(classification_report(y_test, y_pred_knn))



# --- Task 4: Build and Evaluate a Logistic Regression Classifier ---

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

logreg_pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=1000)),
    ]
)

logreg_pipeline.fit(X_train, y_train)

y_pred_logreg = logreg_pipeline.predict(X_test)

logreg_accuracy = accuracy_score(y_test, y_pred_logreg)
logreg_precision = precision_score(y_test, y_pred_logreg, pos_label=1)
logreg_recall = recall_score(y_test, y_pred_logreg, pos_label=1)
logreg_f1 = f1_score(y_test, y_pred_logreg, pos_label=1)

print("\nLogistic Regression Results:")
print("Accuracy:", logreg_accuracy)
print("Precision:", logreg_precision)
print("Recall:", logreg_recall)
print("F1-score:", logreg_f1)

print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, y_pred_logreg))

# Print feature coefficients
coefficients = logreg_pipeline.named_steps["logreg"].coef_[0]

print("Logistic Regression Feature Coefficients:")

for feature, coefficient in zip(numeric, coefficients):
    print(f"{feature}: {coefficient:.4f}")

# Negative coefficients push predictions toward skip (class 0).
# Positive coefficients push predictions toward good (class 1).
# The largest negative coefficients show which weather conditions
# most strongly make a day unsuitable for running.



# --- Task 5: Compare and Choose ---

knn_accuracy = accuracy_score(y_test, y_pred_knn)
knn_precision = precision_score(y_test, y_pred_knn, pos_label=1)
knn_recall = recall_score(y_test, y_pred_knn, pos_label=1)
knn_f1 = f1_score(y_test, y_pred_knn, pos_label=1)

print("\nModel Comparison:")
print(f"KNN Accuracy: {knn_accuracy:.4f}")
print(f"Logistic Regression Accuracy: {logreg_accuracy:.4f}")

# KNN performed better than Logistic Regression on this dataset.
# For a running app, precision is especially important because false
# positives may recommend running in unsuitable weather conditions.
# Recall also matters because false negatives mean missing good running days.
# I chose KNN because it achieved higher accuracy, precision, recall,
# and F1-score. Although Logistic Regression is smaller and easier to
# interpret, KNN performed better on our test data and supports probabilities.


# --- Task 6: Save the Model ---

import json
import sys
import sklearn
import joblib

# Calculate the F1-score for the selected KNN model
knn_f1 = f1_score(y_test, y_pred_knn, pos_label=1)

# Save the complete fitted KNN pipeline
joblib.dump(best_knn, "models/weather_classifier.pkl")

# Create metadata
metadata = {
    "python_version": sys.version.split()[0],
    "scikit_learn_version": sklearn.__version__,
    "feature_names": numeric,
    "label_rule": (
        "Good day if maximum temperature is between 7 and 26 C, "
        "minimum temperature is at least 0 C, precipitation is under "
        "3 mm, and maximum wind speed is under 30 km/h."
    ),
    "city": "Raleigh, North Carolina, USA",
    "test_accuracy": float(knn_accuracy),
    "test_f1": float(knn_f1),
    "best_k": int(best_k),
}

with open("models/weather_classifier_metadata.json", "w") as file:
    json.dump(metadata, file, indent=4)

print("\nModel and metadata saved successfully!")
print("Model: models/weather_classifier.pkl")
print("Metadata: models/weather_classifier_metadata.json")
