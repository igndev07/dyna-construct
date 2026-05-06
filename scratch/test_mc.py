import pandas as pd
import numpy as np
from xgboost import XGBRegressor

def monte_carlo_simulation(model, base_input: dict, budget_cr: float, n_simulations: int = 1000):
    noise_cfg = {
        "weather": (0, 0), "labor": (0, 15), "material_delay": (0, 0),
        "complexity": (0, 0), "progress": (0, 10), "soil_risk": (0, 0),
        "equipment": (0, 15), "rework_rate": (0, 0.08),
    }
    rows = []
    for _ in range(n_simulations):
        row = {}
        for feat, (mn, sd) in noise_cfg.items():
            val = base_input[feat]
            row[feat] = val + np.random.normal(mn, sd) if sd > 0 else val
        rows.append(row)
    df = pd.DataFrame(rows)
    delays = np.clip(model.predict(df), 0, None)
    costs = budget_cr + (delays * budget_cr * 0.008) + np.random.normal(0, budget_cr*0.02, n_simulations)
    return delays, costs

model = XGBRegressor()
X = pd.DataFrame(np.random.rand(10, 8), columns=["weather", "labor", "material_delay", "complexity", "progress", "soil_risk", "equipment", "rework_rate"])
y = np.random.rand(10)
model.fit(X, y)

base_input = {
    "weather": 0, "labor": 70, "material_delay": 0, "complexity": 2,
    "progress": 0, "soil_risk": 0, "equipment": 80, "rework_rate": 0.05
}

try:
    mc = monte_carlo_simulation(model, base_input, 500.0, n_simulations=1000)
    print("Success")
except Exception as e:
    print(f"Error: {e}")
