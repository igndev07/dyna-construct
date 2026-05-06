import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import networkx as nx

def generate_synthetic_data(n=2000):
    np.random.seed(42)
    weather     = np.random.randint(0, 3, n)
    labor       = np.random.randint(20, 100, n)
    mat_delay   = np.random.randint(0, 2, n)
    complexity  = np.random.randint(1, 4, n)
    progress    = np.random.randint(5, 95, n)
    soil_risk   = np.random.randint(0, 3, n)
    equipment   = np.random.randint(40, 100, n)
    rework_rate = np.random.uniform(0, 0.25, n)

    # Nonlinear interactions (e.g., Rainy + Low Labor = Exponential Delay)
    interaction_term = (weather == 1).astype(int) * (100 - labor) * 0.25
    
    delay_days = (
        (weather * 3.5) +
        (100 - labor) * 0.18 +
        (mat_delay * 15.0) +
        (complexity * 5.5) +
        (soil_risk * 4.0) +
        (100 - equipment) * 0.12 +
        (rework_rate * 25) +
        interaction_term -
        (progress * 0.1)
    ) + np.random.normal(0, 3.8, n) # Increased noise for more realistic accuracy

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
    
    # Tuned XGBoost for competition-grade performance
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.06,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42,
        verbosity=0
    )
    
    model.fit(X_tr, y_tr)
    acc = r2_score(y_te, model.predict(X_te))
    return model, acc, list(X.columns)

def monte_carlo_simulation(model, base_input: dict, budget_cr: float, n_simulations: int = 1000):
    # Ensure budget_cr is a float
    budget_cr = float(budget_cr)
    n_simulations = int(n_simulations)
    
    noise_cfg = {
        "weather": (0, 0), "labor": (0, 15), "material_delay": (0, 0),
        "complexity": (0, 0), "progress": (0, 10), "soil_risk": (0, 0),
        "equipment": (0, 15), "rework_rate": (0, 0.08),
    }
    rows = []
    for _ in range(n_simulations):
        row = {}
        for feat, (mn, sd) in noise_cfg.items():
            # Explicitly cast base value to float to avoid type errors during addition
            val = float(base_input[feat])
            row[feat] = val + np.random.normal(mn, sd) if sd > 0 else val
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df["labor"] = df["labor"].clip(10, 100)
    df["progress"] = df["progress"].clip(0, 100)
    df["equipment"] = df["equipment"].clip(10, 100)
    df["rework_rate"] = df["rework_rate"].clip(0, 0.4)
    
    # Predict using the XGBoost model
    delays = np.clip(model.predict(df), 0, None)
    
    # Probabilistic cost impact
    # Base cost + (delay * 0.8% of budget daily) + random cost overruns
    costs = budget_cr + (delays * budget_cr * 0.008) + np.random.normal(0, budget_cr*0.02, n_simulations)
    
    # Calculate percentiles
    p10, p50, p90 = np.percentile(delays, [10, 50, 90])
    cp10, cp50, cp90 = np.percentile(costs, [10, 50, 90])
    
    return {
        "simulations": delays, 
        "cost_simulations": costs,
        "mean": float(np.mean(delays)),
        "p10": float(p10), "p50": float(p50), "p90": float(p90),
        "cp10": float(cp10), "cp50": float(cp50), "cp90": float(cp90),
        "std": float(np.std(delays)),
        "prob_overrun_15": float(np.mean(delays > 15) * 100),
        "prob_overrun_30": float(np.mean(delays > 30) * 100),
        "prob_budget_overrun": float(np.mean(costs > budget_cr * 1.1) * 100),
    }

def compute_critical_path(predicted_delay: float, complexity: int, budget_cr: float, task_delays: dict = None, speed_multiplier: float = 1.0):
    # --- SYNCHRONIZED SCIENCE: BROMILOW'S LAW ---
    # We use the same baseline as the Generative Engine for consistency.
    target_duration = (55 * (budget_cr ** 0.35)) / speed_multiplier
    
    # Scaling factor for individual tasks (Base tasks sum to approx 60 units)
    scale = (target_duration / 60.0)
    c_mul = 1 + (complexity - 1) * 0.2
    
    if task_delays is None:
        task_delays = {}
        
    base_tasks = {
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
    
    # Add the AI Predicted delay to the critical path (spread across execution phases)
    # This ensures the predicted delay from Tab 1 matches the total CPM duration
    base_tasks["Foundation Work"]["duration"] += predicted_delay * 0.4
    base_tasks["Structural Framing"]["duration"] += predicted_delay * 0.4
    base_tasks["Finishing Works"]["duration"]    += predicted_delay * 0.2
    
    # Apply specific task delays if any (from IoT)
    for t, delay_val in task_delays.items():
        if t in base_tasks:
            base_tasks[t]["duration"] += delay_val
            
    G = nx.DiGraph()
    G.add_node("START", duration=0)
    G.add_node("END", duration=0)
    for task, info in base_tasks.items():
        G.add_node(task, duration=info["duration"])
        if not info["deps"]:
            G.add_edge("START", task)
        for dep in info["deps"]:
            G.add_edge(dep, task)
    for task in base_tasks:
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
        
    critical = [n for n in G.nodes if G.nodes[n].get("slack", 1) <= 0.01 and n not in ("START", "END")]
    return G, critical, base_tasks, round(project_duration, 1)