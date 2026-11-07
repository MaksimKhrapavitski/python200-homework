
import json
import joblib
import pandas as pd

# --- Task 7: Load and Predict ---

# Load the saved model
model = joblib.load("models/weather_classifier.pkl")

# Load model metadata
with open("models/weather_classifier_metadata.json", "r") as file:
    metadata = json.load(file)

# Print model metadata
print("Model Information:")
print("City:", metadata["city"])
print("Features:", metadata["feature_names"])
print("Test Accuracy:", round(metadata["test_accuracy"], 4))

# Create hypothetical weather days
feature_names = metadata["feature_names"]

new_days = pd.DataFrame(
    [
        [22.0, 14.0, 0.0, 10.0],
        [34.0, 25.0, 8.0, 40.0],
        [27.0, 15.0, 2.5, 20.0],
        [12.0, 2.0, 0.5, 28.0],
    ],
    columns=feature_names,
    index=["Clearly good", "Clearly bad", "Borderline", "Cold day"],
)

# Make predictions
predictions = model.predict(new_days)
probabilities = model.predict_proba(new_days)

good_class_index = list(model.classes_).index(1)

print("\nWeather Predictions:")

for i, day_name in enumerate(new_days.index):
    label = "good" if predictions[i] == 1 else "skip"
    confidence = probabilities[i, good_class_index]

    print(f"\n{day_name}:")
    print(new_days.iloc[i].to_dict())
    print("Prediction:", label)
    print("Probability of good:", round(confidence, 2))



# --- Task 8: Reflection ---

# 1. The borderline day received a probability of 0.33 for good.
# The model predicts skip, but it is not completely certain.
# If the model predicts 0.52, I would treat the result as uncertain
# and avoid making a strong recommendation based only on this prediction.

# 2. If predict.py runs before train_classifier.py, the saved model
# and metadata files will not exist, causing a FileNotFoundError.
# A more helpful error message could tell the user to run
# train_classifier.py first to create the required files.

# 3. In Week 10, predict.py would need to receive weather data
# automatically from a cloud pipeline instead of using
# hypothetical weather data entered manually.
