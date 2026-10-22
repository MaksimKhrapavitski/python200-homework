# --- scikit-learn API ---
# Q1

import numpy as np
from sklearn.linear_model import LinearRegression

temp_min = np.array([1, 4, 8, 12, 16, 20]).reshape(-1, 1)
temp_max = np.array([8, 10, 15, 18, 23, 27])

model = LinearRegression()
model.fit(temp_min, temp_max)

new_temperatures = np.array([6, 18]).reshape(-1, 1)
predictions = model.predict(new_temperatures)

print("Predicted high temperature for 6 degrees:", predictions[0])
print("Predicted high temperature for 18 degrees:", predictions[1])

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)


#Q2
import numpy as np

x = np.array([10, 20, 30, 40, 50])

print("Original shape:", x.shape)

x = x.reshape(-1, 1)

print("New shape:", x.shape)

# scikit-learn expects X to be 2D, with rows representing samples
# and columns representing features.


# --- Linear Regression ---

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

rng = np.random.default_rng(42)
num_days = 120

temp_min = rng.uniform(-5, 22, num_days)
is_rainy = rng.integers(0, 2, num_days).astype(float)
temp_max = (
    1.0 * temp_min
    + 6
    - 4 * is_rainy
    + rng.normal(0, 2, num_days)
)
# Question 1: Visualize the relationship between daily low and high temperatures.

plt.figure(figsize=(8, 6))

plt.scatter(
    temp_min,
    temp_max,
    c=is_rainy,
    cmap="coolwarm"
)

plt.title("Daily High vs Low")
plt.xlabel("Minimum Temperature")
plt.ylabel("Maximum Temperature")

plt.savefig("assignments_02/outputs/high_vs_low.png")
plt.show()

# The points appear to form two groups because rainy and non-rainy days
# have different expected daily high temperatures.


# Question 2: Split the data into training and test sets.

X = temp_min.reshape(-1, 1)
y = temp_max

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Question 3: Train and evaluate the model.

model = LinearRegression()
model.fit(X_train, y_train)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = model.score(X_test, y_test)

print("RMSE:", rmse)
print("R² on test set:", r2)

# The slope estimates how much the predicted daily high changes
# when the daily low increases by one degree.


# Question 4: Train a model using both temperature and rain status.

X_full = np.column_stack([temp_min, is_rainy])

X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full,
    y,
    test_size=0.2,
    random_state=42
)

model_full = LinearRegression()
model_full.fit(X_train_full, y_train_full)

y_pred_full = model_full.predict(X_test_full)

r2_full = model_full.score(X_test_full, y_test_full)

print("R² with temp_min only:", r2)
print("R² with temp_min and is_rainy:", r2_full)

print("temp_min coefficient:", model_full.coef_[0])
print("is_rainy coefficient:", model_full.coef_[1])

# The is_rainy coefficient estimates how much the predicted daily high
# changes on rainy days compared with non-rainy days,
# holding minimum temperature constant.


# Question 5: Plot predicted vs actual temperatures.

plt.figure(figsize=(8, 6))

plt.scatter(y_pred_full, y_test_full)

min_value = min(y_pred_full.min(), y_test_full.min())
max_value = max(y_pred_full.max(), y_test_full.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.title("Predicted vs Actual")
plt.xlabel("Predicted Maximum Temperature")
plt.ylabel("Actual Maximum Temperature")

plt.savefig(
    "assignments_02/outputs/predicted_vs_actual_warmup.png"
)

plt.show()

# Points above the diagonal represent actual temperatures higher than predictions.
# Points below the diagonal represent actual temperatures lower than predictions.