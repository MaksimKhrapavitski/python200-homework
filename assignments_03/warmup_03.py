import joblib
import pandas as pd  # pyright: ignore[reportMissingModuleSource]
import matplotlib.pyplot as plt  # pyright: ignore[reportMissingModuleSource]

from sklearn.model_selection import train_test_split, cross_val_score  # pyright: ignore[reportMissingModuleSource]
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # pyright: ignore[reportMissingModuleSource]
from sklearn.compose import ColumnTransformer  # pyright: ignore[reportMissingModuleSource]
from sklearn.pipeline import Pipeline  # pyright: ignore[reportMissingModuleSource]
from sklearn.neighbors import KNeighborsClassifier  # pyright: ignore[reportMissingModuleSource]
from sklearn.linear_model import LogisticRegression  # pyright: ignore[reportMissingModuleSource]
from sklearn.metrics import (  # pyright: ignore[reportMissingModuleSource]
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
)
df = pd.read_csv("weather_classification.csv")
numeric = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]
X = df[numeric]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# --- Preprocessing ---
# Q1

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Mean of scaled training data:")
print(X_train_scaled.mean(axis=0))

# We fit the scaler on X_train only to avoid data leakage from the test set.



# Q2

encoder = OneHotEncoder(sparse_output=False)

season_encoded = encoder.fit_transform(df[["season"]])

print("Encoded season shape:", season_encoded.shape)
print("Category names:", encoder.get_feature_names_out(["season"]))

# One categorical column becomes four binary columns because the dataset has four seasons.



# Q3

X_all = df[numeric + ["season"]]

X_train_q3, X_test_q3, y_train_q3, y_test_q3 = train_test_split(
    X_all, y, test_size=0.2, random_state=42, stratify=y
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(), ["season"]),
    ]
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression()),
    ]
)

pipeline.fit(X_train_q3, y_train_q3)

y_pred_q3 = pipeline.predict(X_test_q3)
accuracy_q3 = accuracy_score(y_test_q3, y_pred_q3)

print("Q3 test accuracy:", accuracy_q3)

# The pipeline automatically preprocesses numeric and categorical features
# before fitting the classifier and applying the same transformations to new data.



# --- KNN ---
# Q1

knn_pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5)),
    ]
)

knn_pipeline.fit(X_train, y_train)

y_pred_knn = knn_pipeline.predict(X_test)

print("KNN Test Accuracy (k=5):", accuracy_score(y_test, y_pred_knn))
print("Classification Report:")
print(classification_report(y_test, y_pred_knn))



# Q2

k_values = [1, 3, 5, 7, 9, 11, 15, 21]
cv_scores = {}

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

    mean_score = scores.mean()
    cv_scores[k] = mean_score

    print(f"k={k}: Mean CV Accuracy = {mean_score:.4f}")

best_k = max(cv_scores, key=cv_scores.get)
print("Best k:", best_k)
print("Best mean CV accuracy:", cv_scores[best_k])

# I would choose the k with the highest mean cross-validation accuracy
# because it performed best on average across the five validation folds.



# Q3

knn_raw = KNeighborsClassifier(n_neighbors=5)
knn_raw.fit(X_train, y_train)
y_pred_raw = knn_raw.predict(X_test)

accuracy_raw = accuracy_score(y_test, y_pred_raw)

knn_scaled = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5)),
    ]
)
knn_scaled.fit(X_train, y_train)
y_pred_scaled = knn_scaled.predict(X_test)

accuracy_scaled = accuracy_score(y_test, y_pred_scaled)

print("KNN accuracy without scaling:", accuracy_raw)
print("KNN accuracy with scaling:", accuracy_scaled)

# Scaling helps when features have different ranges because KNN uses distances.
# Compare the two accuracies to determine whether scaling helped on this dataset.

# Scaling improved accuracy from 0.925 to about 0.9583.
# This is important because KNN uses distances, and features with
# larger numerical ranges can otherwise dominate the distance calculation.



# --- Logistic Regression ---
# Q1

logreg_pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=1000)),
    ]
)

logreg_pipeline.fit(X_train, y_train)

y_pred_logreg = logreg_pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred_logreg)
precision = precision_score(y_test, y_pred_logreg, pos_label=1)
recall = recall_score(y_test, y_pred_logreg, pos_label=1)
f1 = f1_score(y_test, y_pred_logreg, pos_label=1)

print("Logistic Regression Test Results:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)



# Q2

coefficients = logreg_pipeline.named_steps["logreg"].coef_[0]

print("Feature coefficients:")

for feature, coefficient in zip(numeric, coefficients):
    print(f"{feature}: {coefficient:.4f}")

# Features with the largest negative coefficients push predictions
# toward the "skip" class (good_for_running = 0).



# Q3

probabilities = logreg_pipeline.predict_proba(X_test)

good_class_index = list(logreg_pipeline.classes_).index(1)

good_probabilities = probabilities[:5, good_class_index]

print("Probability of good for running (first 5 test days):")

for i, probability in enumerate(good_probabilities, start=1):
    print(f"Day {i}: {probability:.2f}")

# A probability near 0.5 means the model is uncertain
# because both classes have similar predicted probabilities.


# Wind speed and precipitation have the largest negative coefficients,
# meaning they push predictions most strongly toward the "skip" class.
# This matches my intuition because strong winds and rain make running
# conditions less comfortable.



# --- Evaluation ---
# Q1

cm = confusion_matrix(y_test, y_pred_logreg)

print("Confusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["skip", "good"]
)

disp.plot(cmap="Blues")
plt.title("Logistic Regression Confusion Matrix")
plt.savefig("outputs/logreg_confusion_matrix.png")
plt.close()

false_positives = cm[0, 1]
false_negatives = cm[1, 0]

print("False Positives:", false_positives)
print("False Negatives:", false_negatives)

# False positives may be more dangerous for a running app because
# recommending a run on unsuitable weather conditions could create safety risks.
# The model made 9 false positives and 7 false negatives.
# False positives matter more for a running app because recommending
# running in unsuitable weather could create safety risks.


# --- joblib ---
# Q1

joblib.dump(logreg_pipeline, "models/warmup_model.pkl")

loaded = joblib.load("models/warmup_model.pkl")

assert (loaded.predict(X_test) == logreg_pipeline.predict(X_test)).all()

print("Predictions match.")

# Saving only the LogisticRegression step would lose the fitted scaler.
# Raw data would not be scaled before prediction, which could produce
# incorrect predictions because the model was trained on scaled features.
