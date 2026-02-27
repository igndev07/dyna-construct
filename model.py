import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def generate_synthetic_data(n=800):
    np.random.seed(42)

    weather = np.random.randint(0, 3, n)  # 0: Sunny, 1: Rainy, 2: Humid
    labor = np.random.randint(30, 100, n)
    material_delay = np.random.randint(0, 2, n)
    complexity = np.random.randint(1, 4, n)
    progress = np.random.randint(10, 90, n)

    delay_days = (
        (weather * 2.5) +
        (100 - labor) * 0.12 +
        (material_delay * 12) +
        (complexity * 4) -
        (progress * 0.06)
    ) + np.random.normal(0, 1.5, n)

    data = pd.DataFrame({
        "weather": weather,
        "labor": labor,
        "material_delay": material_delay,
        "complexity": complexity,
        "progress": progress,
        "delay_days": delay_days
    })

    return data

def train_model():
    data = generate_synthetic_data()
    X = data.drop("delay_days", axis=1)
    y = data["delay_days"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = r2_score(y_test, y_pred)

    return model, accuracy, X.columns