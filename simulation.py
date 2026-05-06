import numpy as np

def generate_recommendation(weather, labor, material_delay, complexity, predicted_delay,
                             soil_risk=0, equipment=80, rework_rate=0.05):
    recs = []
    cost_impact = 0

    # --- Delay severity
    if predicted_delay > 30:
        recs.append(("CRITICAL", "🚨 Severe delay risk (>30 days): Activate emergency response protocol. Escalate to project director immediately.", 25))
        cost_impact += predicted_delay * 1.8
    elif predicted_delay > 15:
        recs.append(("HIGH", "⚠️ High delay risk (15–30 days): Re-sequence critical path tasks. Consider parallel execution where possible.", 15))
        cost_impact += predicted_delay * 1.2
    elif predicted_delay > 8:
        recs.append(("MEDIUM", "📋 Moderate delay detected: Review resource allocation and tighten schedule buffers.", 8))
        cost_impact += predicted_delay * 0.8

    # --- Labor
    if labor < 40:
        recs.append(("HIGH", f"👷 Critical labor shortage ({labor}%): Deploy subcontractors immediately. Activate shift scheduling (2 shifts/day).", 12))
        cost_impact += 8
    elif labor < 60:
        recs.append(("MEDIUM", f"👷 Low labor availability ({labor}%): Pre-qualify backup workforce. Monitor daily attendance.", 6))
        cost_impact += 4

    # --- Material
    if material_delay == 1:
        recs.append(("HIGH", "📦 Material supply disruption: Switch to pre-approved alternate suppliers. Fast-track procurement for critical items.", 10))
        cost_impact += 6

    # --- Weather
    if weather == 1:
        recs.append(("MEDIUM", "🌧️ Rain forecast: Waterproof formwork, halt concrete pours, protect excavations. Revise outdoor schedule.", 5))
        cost_impact += 3
    elif weather == 2:
        recs.append(("MEDIUM", "💧 High humidity: Extend concrete curing time by 20%. Monitor steel corrosion risk.", 4))
        cost_impact += 2

    # --- Site complexity
    if complexity == 3:
        recs.append(("HIGH", "🏗️ High site complexity: Deploy real-time monitoring on all critical structural nodes. Daily engineering review.", 8))
        cost_impact += 5

    # --- Soil risk
    if soil_risk == 2:
        recs.append(("CRITICAL", "⛰️ Unstable soil detected: Halt heavy equipment operations. Conduct emergency geotechnical review.", 20))
        cost_impact += 15
    elif soil_risk == 1:
        recs.append(("MEDIUM", "⛰️ Moderate soil risk: Increase foundation inspection frequency. Deploy settlement monitors.", 8))
        cost_impact += 5

    # --- Equipment
    if equipment < 50:
        recs.append(("HIGH", f"🔧 Low equipment availability ({equipment}%): Rent additional machinery. Prioritise equipment for critical path tasks.", 10))
        cost_impact += 6
    elif equipment < 70:
        recs.append(("MEDIUM", f"🔧 Equipment utilisation suboptimal ({equipment}%): Schedule preventive maintenance off-shift.", 5))

    # --- Rework rate
    rework_pct = round(rework_rate * 100, 1)
    if rework_rate > 0.15:
        recs.append(("HIGH", f"🔁 High rework rate ({rework_pct}%): Mandatory QC checkpoint before each phase. Root cause analysis required.", 12))
        cost_impact += rework_rate * 30
    elif rework_rate > 0.08:
        recs.append(("MEDIUM", f"🔁 Elevated rework ({rework_pct}%): Increase inspection frequency at pour stages.", 6))
        cost_impact += rework_rate * 15

    # --- Stable
    if not recs:
        recs.append(("OK", "✅ System Stable: All parameters within optimal range. Continue with current construction plan.", 0))

    return recs, round(cost_impact, 1)


def generate_design_variants(complexity: int, budget_cr: float, site_area_sqm: float):
    """
    Generative design: produce 3 ranked structural layout variants.
    Each variant trades off cost, time, and risk differently.
    """
    base_cost = budget_cr
    variants = [
        {
            "name": "Variant A — Lean Execution",
            "description": "Optimised for minimum cost. Sequential task scheduling, standard materials, conventional formwork.",
            "est_cost_cr":      round(base_cost * 0.88, 2),
            "est_duration_days": round(complexity * 42 * 0.95, 0),
            "rework_risk":      "Medium (12%)",
            "carbon_t":         round(site_area_sqm * 0.18, 1),
            "score":            72,
            "recommended":      False,
        },
        {
            "name": "Variant B — Balanced Optimum ⭐",
            "description": "AI-recommended balance of cost, time, and safety. Parallel critical path execution, prefab elements for repetitive sections.",
            "est_cost_cr":      round(base_cost * 0.96, 2),
            "est_duration_days": round(complexity * 42 * 0.80, 0),
            "rework_risk":      "Low (5%)",
            "carbon_t":         round(site_area_sqm * 0.14, 1),
            "score":            91,
            "recommended":      True,
        },
        {
            "name": "Variant C — Speed-First",
            "description": "Maximum speed using modular construction and 3-shift scheduling. Higher upfront cost, earliest handover.",
            "est_cost_cr":      round(base_cost * 1.12, 2),
            "est_duration_days": round(complexity * 42 * 0.65, 0),
            "rework_risk":      "Low (4%)",
            "carbon_t":         round(site_area_sqm * 0.16, 1),
            "score":            84,
            "recommended":      False,
        },
    ]
    return variants


def simulate_iot_tick(prev_state: dict) -> dict:
    """
    Simulate one IoT sensor tick. Called every few seconds for live feed.
    Returns updated sensor readings with small random drift.
    """
    def drift(val, lo, hi, sigma):
        return float(np.clip(val + np.random.normal(0, sigma), lo, hi))

    concrete_strength = drift(prev_state.get("concrete_strength", 28.0), 20, 45, 0.4)
    ambient_temp      = drift(prev_state.get("ambient_temp", 32.0), 15, 48, 0.6)
    humidity          = drift(prev_state.get("humidity", 65.0), 30, 95, 1.0)
    structural_load   = drift(prev_state.get("structural_load", 72.0), 40, 110, 1.5)
    dust_ppm          = drift(prev_state.get("dust_ppm", 180.0), 50, 500, 8.0)
    vibration_mmps    = drift(prev_state.get("vibration_mmps", 3.2), 0, 15, 0.3)

    # Anomaly flags
    alerts = []
    if concrete_strength < 22:
        alerts.append("⚠️ Concrete strength below threshold — delay curing phase")
    if structural_load > 100:
        alerts.append("🚨 Structural load exceeding safe limit — halt heavy operations")
    if ambient_temp > 43:
        alerts.append("🌡️ Extreme heat — protect fresh concrete, hydrate workers")
    if humidity > 88:
        alerts.append("💧 High humidity — extend curing, check steel corrosion")
    if dust_ppm > 400:
        alerts.append("😷 Dust levels critical — enforce PPE, consider halting")

    return {
        "concrete_strength": round(concrete_strength, 2),
        "ambient_temp":      round(ambient_temp, 1),
        "humidity":          round(humidity, 1),
        "structural_load":   round(structural_load, 1),
        "dust_ppm":          round(dust_ppm, 1),
        "vibration_mmps":    round(vibration_mmps, 2),
        "alerts":            alerts,
    }


# ─────────────────────────────────────────
#  FINANCIAL IMPACT  (clear, auditable math)
# ─────────────────────────────────────────
def compute_financials(predicted_delay: float, budget_cr: float, rework_rate: float):
    """
    All numbers in ₹ Crore.

    LOSSES (what the project is bleeding):
      delay_loss    = predicted_delay × budget × 0.008
                      (0.8% of budget per day: idle labour, equipment standby,
                       financing cost, penalty clauses — McKinsey benchmark)
      rework_loss   = budget × rework_rate
                      (fraction of budget spent redoing already-done work)
      material_waste= budget × rework_rate × 0.18
                      (18% of rework cost becomes landfill — industry average)

    SAVINGS (what DynaConstructa recovers via early warning):
      delay_saving  = delay_loss × 0.55
                      (55% of delay cost avoided when corrective action is
                       taken 48–72 hrs early vs after the fact)
      rework_saving = rework_loss × 0.60
                      (60% of rework eliminated by catching design-reality
                       mismatches before execution, not after)
      waste_saving  = material_waste × 0.50
                      (50% of waste avoided through better pre-planning)
    """
    # ── LOSSES ──────────────────────────────────────────────
    delay_loss     = round(predicted_delay * budget_cr * 0.008, 1)
    rework_loss    = round(budget_cr * rework_rate, 1)
    material_waste = round(budget_cr * rework_rate * 0.18, 1)
    total_loss     = round(delay_loss + rework_loss + material_waste, 1)

    # ── SAVINGS ─────────────────────────────────────────────
    delay_saving   = round(delay_loss  * 0.55, 1)
    rework_saving  = round(rework_loss * 0.60, 1)
    waste_saving   = round(material_waste * 0.50, 1)
    total_saving   = round(delay_saving + rework_saving + waste_saving, 1)

    # ── ROI ─────────────────────────────────────────────────
    tool_cost_est  = round(budget_cr * 0.002, 1)   # ~0.2% of budget = platform cost
    net_benefit    = round(total_saving - tool_cost_est, 1)
    roi_pct        = round((net_benefit / max(tool_cost_est, 0.1)) * 100, 0)

    return {
        # Losses
        "delay_loss":     delay_loss,
        "rework_loss":    rework_loss,
        "material_waste": material_waste,
        "total_loss":     total_loss,
        # Savings
        "delay_saving":   delay_saving,
        "rework_saving":  rework_saving,
        "waste_saving":   waste_saving,
        "total_saving":   total_saving,
        # ROI
        "tool_cost":      tool_cost_est,
        "net_benefit":    net_benefit,
        "roi_pct":        int(roi_pct),
    }