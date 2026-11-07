
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd


# Features used by the trained weather model
FEATURES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]


@dataclass
class Prediction:
    """The model's verdict for a single day."""

    label: str
    probability: float



class WeatherClassifier:
    """Loads a saved weather pipeline once and predicts on many days."""

    def __init__(self, model_path: str | Path):
        """Load the saved pipeline from disk.

        Args:
            model_path: Path to a .pkl file saved with joblib.

        Raises:
            FileNotFoundError: If no file exists at model_path.
        """
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"No model file at {model_path}. Train and save a model first."
            )

        self.model = joblib.load(model_path)
        self.features = FEATURES



    def predict(self, days: list[dict]) -> list[Prediction]:
        """Predict whether each day is good for running.

        Args:
            days: A list of dictionaries, each with the four feature keys.

        Returns:
            One Prediction per input day, in the same order.

        Raises:
            ValueError: If any required feature is missing.
        """
        if not days:
            return []

        frame = pd.DataFrame(days)

        missing = [
            f for f in self.features
            if f not in frame.columns
        ]

        if missing:
            raise ValueError(f"missing required features: {missing}")

        X = frame[self.features]

        labels = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]

        return [
            Prediction(
                "good" if int(label) == 1 else "skip",
                float(prob)
            )
            for label, prob in zip(labels, probabilities)
        ]
