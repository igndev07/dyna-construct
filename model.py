import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import networkx as nx

def generate_synthetic_data(n=1500):
    np.random.seed(42)
    weather     = np.random.randint(0, 3, n)
    labor       = np.random.randint(20, 100, n)
    mat_delay   = np.random.randint(0, 2, n)
    complexity  = np.random.randint(1, 4, n)
    progress    = np.random.randint(5, 95, n)
    soil_risk   = np.random.randint(0, 3, n)
    equipment   = np.random.randint(40, 100, n)
    rework_rate = np.random.uniform(0, 0.25, n)

    delay_days = (
        (weather * 2.8) +
        (100 - labor) * 0.14 +
        (mat_delay * 13.5) +
        (complexity * 4.5) +
        (soil_risk * 3.0) +
        (100 - equipment) * 0.08 +
        (rework_rate * 20) -
        (progress * 0.07)
    ) + np.random.normal(0, 1.8, n)

    return pd.DataFrame({
        "weather": weather, "labor": labor,
        "material_delay": mat_delay, "complexity": complexity,
        "progress": progress, "soil_risk": soil_risk,
        "equipment": equipment, "rework_rate": rework_rate,
        "delay_days": np.clip(delay_days, 0, None)
    })

def train_model():
    data = generate_synthetic_data()
    X = data.drop("delay_days", axis=1)
    y = data["delay_days"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)
    model.fit(X_tr, y_tr)
    acc = r2_score(y_te, model.predict(X_te))
    return model, acc, list(X.columns)

def monte_carlo_simulation(model, base_input: dict, n_simulations: int = 1000):
    noise_cfg = {
        "weather": (0, 0), "labor": (0, 10), "material_delay": (0, 0),
        "complexity": (0, 0), "progress": (0, 8), "soil_risk": (0, 0),
        "equipment": (0, 10), "rework_rate": (0, 0.05),
    }
    rows = []
    for _ in range(n_simulations):
        row = {}
        for feat, (mn, sd) in noise_cfg.items():
            val = base_input[feat]
            row[feat] = val + np.random.normal(mn, sd) if sd > 0 else val
        rows.append(row)
    df = pd.DataFrame(rows)
    df["labor"] = df["labor"].clip(10, 100)
    df["progress"] = df["progress"].clip(0, 100)
    df["equipment"] = df["equipment"].clip(10, 100)
    df["rework_rate"] = df["rework_rate"].clip(0, 0.4)
    results = np.clip(model.predict(df), 0, None)
    p10, p50, p90 = np.percentile(results, [10, 50, 90])
    return {
        "simulations": results, "mean": float(np.mean(results)),
        "p10": float(p10), "p50": float(p50), "p90": float(p90),
        "std": float(np.std(results)),
        "prob_overrun_15": float(np.mean(results > 15) * 100),
        "prob_overrun_30": float(np.mean(results > 30) * 100),
    }

def compute_critical_path(predicted_delay: float, complexity: int):
    scale = max(0.5, predicted_delay / 8)
    c_mul = 1 + (complexity - 1) * 0.25
    tasks = {
        "Site Survey":        {"duration": round(2  * scale * c_mul, 1), "deps": []},
        "Soil Testing":       {"duration": round(3  * scale * c_mul, 1), "deps": ["Site Survey"]},
        "Foundation Design":  {"duration": round(5  * scale * c_mul, 1), "deps": ["Soil Testing"]},
        "Procurement":        {"duration": round(7  * scale,         1), "deps": ["Site Survey"]},
        "Foundation Work":    {"duration": round(10 * scale * c_mul, 1), "deps": ["Foundation Design", "Procurement"]},
        "Structural Framing": {"duration": round(14 * scale * c_mul, 1), "deps": ["Foundation Work"]},
        "MEP Rough-in":       {"duration": round(8  * scale,         1), "deps": ["Structural Framing"]},
        "Concrete Pours":     {"duration": round(6  * scale * c_mul, 1), "deps": ["Structural Framing"]},
        "Finishing Works":    {"duration": round(9  * scale,         1), "deps": ["MEP Rough-in", "Concrete Pours"]},
        "QC Inspection":      {"duration": round(3  * scale,         1), "deps": ["Finishing Works"]},
        "Handover":           {"duration": round(2  * scale,         1), "deps": ["QC Inspection"]},
    }
    G = nx.DiGraph()
    G.add_node("START", duration=0)
    G.add_node("END", duration=0)
    for task, info in tasks.items():
        G.add_node(task, duration=info["duration"])
        if not info["deps"]:
            G.add_edge("START", task)
        for dep in info["deps"]:
            G.add_edge(dep, task)
    for task in tasks:
        if G.out_degree(task) == 0:
            G.add_edge(task, "END")
    for node in nx.topological_sort(G):
        preds = list(G.predecessors(node))
        G.nodes[node]["ES"] = max((G.nodes[p]["EF"] for p in preds), default=0)
        G.nodes[node]["EF"] = G.nodes[node]["ES"] + G.nodes[node]["duration"]
    project_duration = G.nodes["END"]["EF"]
    for node in reversed(list(nx.topological_sort(G))):
        succs = list(G.successors(node))
        G.nodes[node]["LF"] = min((G.nodes[s]["LS"] for s in succs), default=project_duration)
        G.nodes[node]["LS"] = G.nodes[node]["LF"] - G.nodes[node]["duration"]
        G.nodes[node]["slack"] = round(G.nodes[node]["LF"] - G.nodes[node]["EF"], 2)
    critical = [n for n in G.nodes if G.nodes[n].get("slack", 1) == 0 and n not in ("START", "END")]
    return G, critical, tasks, round(project_duration, 1)