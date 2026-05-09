import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
import streamlit as st

from model import train_model, monte_carlo_simulation, compute_critical_path, run_cnn_inference
from simulation import generate_recommendation, generate_design_variants, simulate_iot_tick, compute_financials
from weather import fetch_weather, CITY_COORDS

st.set_page_config(
    page_title="DynaConstructa.AI",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Outfit', sans-serif;
        background-color: #020617;
    }

    /* RESET: Removed custom header/toolbar CSS to restore defaults */
    
    .block-container {






        padding-top: 2.5rem !important;
        padding-bottom: 0rem !important;
    }

    /* Main Title & Branding */
    .main-title { 
        font-size:3rem; 
        font-weight:800; 
        background: linear-gradient(90deg, #F59E0B 0%, #38BDF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing:-1.5px;
        margin-bottom: 0.2rem;
    }
    .sub-title { 
        font-size:1.1rem; 
        color:#94A3B8; 
        font-weight:300;
        margin-bottom:2rem;
        letter-spacing: 1px;
    }

    /* Glassmorphic Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: rgba(56, 189, 248, 0.3);
    }

    /* Section Headers */
    .section-header { 
        font-size:1.1rem; 
        font-weight:700; 
        color:#F8FAFC; 
        display: flex; 
        align-items: center; 
        margin: 2rem 0 1rem;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(245, 158, 11, 0.3);
    }

    /* Tooltips */
    .tooltip { position: relative; display: inline-block; cursor: help; margin-left: 8px; font-size: 0.9rem; color: #38BDF8; }
    .tooltip .tooltiptext { visibility: hidden; width: 260px; background-color: #1E293B; color: #F8FAFC; text-align: left; border-radius: 12px; padding: 12px; position: absolute; z-index: 100; bottom: 130%; left: 0; opacity: 0; transition: opacity 0.3s; font-weight: 400; font-size: 0.8rem; line-height: 1.5; border: 1px solid rgba(56, 189, 248, 0.2); box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.2); }
    .tooltip:hover .tooltiptext { visibility: visible; opacity: 1; }

    /* Variant Cards */
    .variant-card { 
        background: rgba(15, 23, 42, 0.6); 
        border: 1px solid rgba(255, 255, 255, 0.05); 
        border-radius: 12px; 
        padding: 1.25rem; 
        margin-bottom: 1rem; 
    }
    .variant-card-recommended { 
        background: rgba(15, 23, 42, 0.8); 
        border: 2px solid #F59E0B; 
        border-radius: 12px; 
        padding: 1.25rem; 
        margin-bottom: 1rem;
        box-shadow: 0 0 20px rgba(245, 158, 11, 0.15);
    }

    /* IoT Metrics */
    .iot-value { font-size: 1.8rem; font-weight: 700; color: #38BDF8; }
    .iot-label { font-size: 0.85rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.5px; }

    .crisis-banner { 
        background: linear-gradient(90deg, #7F1D1D 0%, #450A0A 100%);
        border: 1px solid #EF4444; 
        border-radius: 12px; 
        padding: 15px 20px; 
        margin-bottom: 1.5rem; 
        color: #FCA5A5; 
        font-weight: 600;
        text-align: center;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 0.9; }
        50% { opacity: 1; box-shadow: 0 0 15px rgba(239, 68, 68, 0.3); }
        100% { opacity: 0.9; }
    }

    div[data-testid="stTabs"] button { 
        font-size:1.05rem; 
        font-weight:600; 
        color: #94A3B8;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] { 
        color: #F59E0B !important;
        border-bottom-color: #F59E0B !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid rgba(56, 189, 248, 0.1);
    }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #F8FAFC;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Training AI model...")
def load_model():
    return train_model()

model, accuracy, feature_names, explainer, cv_scores = load_model()

if "outcome_history" not in st.session_state:
    # Pre-seed with 15 historical projects to show GA learning
    history = []
    np.random.seed(42)
    for i in range(15):
        method = np.random.randint(0, 3)
        pred = np.random.uniform(10, 40)
        # Modular (1) overperforms, Traditional (0) underperforms
        if method == 1: act = pred * np.random.uniform(1.05, 1.20)
        elif method == 0: act = pred * np.random.uniform(0.80, 0.95)
        else: act = pred * np.random.uniform(0.95, 1.05)
        history.append({"project_id": i+1, "method": method, "predicted_saving": pred, "actual_saving": act})
    st.session_state["outcome_history"] = history

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧩 Digital Twin Inputs")

    # ── CRISIS DEMO BUTTON ───────────────────────────────────────────────────
    st.markdown("---")
    if st.button("🚨 CRISIS DEMO — Trigger Site Emergency", use_container_width=True, type="primary"):
        st.session_state["crisis_mode"] = True
        st.session_state["crisis_weather"]  = "Rainy"
        st.session_state["crisis_labor"]    = 22
        st.session_state["crisis_mat"]      = "Yes"
        st.session_state["crisis_complex"]  = "High"
        st.session_state["crisis_soil"]     = "Unstable"
        st.session_state["crisis_equip"]    = 35
        st.session_state["crisis_rework"]   = 22
        st.rerun()

    if st.button("✅ Reset to Normal", use_container_width=True):
        for k in list(st.session_state.keys()):
            if k.startswith("crisis"):
                del st.session_state[k]
        st.rerun()

    st.markdown("---")

    # ── LIVE WEATHER ─────────────────────────────────────────────────────────
    st.markdown("### 🌐 Live Weather (Auto-fetch)")
    city = st.selectbox("Project City", list(CITY_COORDS.keys()), index=0)
    if st.button("📡 Fetch Live Weather", use_container_width=True):
        with st.spinner(f"Fetching weather for {city}..."):
            wx = fetch_weather(city)
            if wx["success"]:
                st.session_state["live_weather"]    = wx["condition"]
                st.session_state["live_wx_data"]    = wx
                st.success(f"✅ {city}: {wx['condition']} | {wx['temp_c']}°C | {wx['humidity']}% RH")
            else:
                st.warning(f"Could not fetch: {wx['error']}")

    if "live_wx_data" in st.session_state:
        wx = st.session_state["live_wx_data"]
        st.markdown(f"""
        <div class="metric-card" style="padding: 1rem; border-color: rgba(56, 189, 248, 0.2);">
            <div style="font-size:0.75rem; color:#94A3B8; margin-bottom:8px; display:flex; align-items:center; gap:6px;">
                <span style="color:#38BDF8">●</span> LIVE WEATHER — {wx['city']}
            </div>
            <div style="display:grid; grid-template-columns: 1fr 1fr; gap:12px;">
                <div style="background:rgba(255,255,255,0.03); padding:8px; border-radius:8px;">
                    <div style="font-size:0.7rem; color:#64748B;">TEMP</div>
                    <div style="color:#F59E0B; font-weight:700; font-size:1rem;">{wx['temp_c']}°C</div>
                </div>
                <div style="background:rgba(255,255,255,0.03); padding:8px; border-radius:8px;">
                    <div style="font-size:0.7rem; color:#64748B;">HUMIDITY</div>
                    <div style="color:#38BDF8; font-weight:700; font-size:1rem;">{wx['humidity']}%</div>
                </div>
                <div style="background:rgba(255,255,255,0.03); padding:8px; border-radius:8px;">
                    <div style="font-size:0.7rem; color:#64748B;">RAIN</div>
                    <div style="color:#60A5FA; font-weight:700; font-size:1rem;">{wx['precip_mm']}mm</div>
                </div>
                <div style="background:rgba(255,255,255,0.03); padding:8px; border-radius:8px;">
                    <div style="font-size:0.7rem; color:#64748B;">WIND</div>
                    <div style="color:#94A3B8; font-weight:700; font-size:1rem;">{wx['wind_kmh']}kph</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── SITE INPUTS ───────────────────────────────────────────────────────────
    # Use crisis values if set, else live weather if available, else defaults
    default_weather  = st.session_state.get("crisis_weather",  st.session_state.get("live_weather", "Sunny"))
    default_labor    = st.session_state.get("crisis_labor",    70)
    default_mat      = st.session_state.get("crisis_mat",      "No")
    default_complex  = st.session_state.get("crisis_complex",  "Medium")
    default_soil     = st.session_state.get("crisis_soil",     "Stable")
    default_equip    = st.session_state.get("crisis_equip",    80)
    default_rework   = st.session_state.get("crisis_rework",   5)

    weather    = st.selectbox("🌤️ Weather Condition", ["Sunny","Rainy","Humid"],
                              index=["Sunny","Rainy","Humid"].index(default_weather),
                              help="Impacts concrete curing and labor efficiency. Rain adds 15-30% delay to site work.")
    labor      = st.slider("👷 Labor Availability (%)", 10, 100, default_labor,
                           help="Percentage of required crew present. Below 60% triggers AI shift-optimization.")
    mat_delay  = st.selectbox("📦 Material Supply Delay", ["No","Yes"],
                              index=["No","Yes"].index(default_mat),
                              help="Global supply chain status. 'Yes' forces a recalculation of the critical path.")
    complexity = st.selectbox("🏗️ Site Complexity", ["Low","Medium","High"],
                              index=["Low","Medium","High"].index(default_complex),
                              help="Technical difficulty of the build (e.g., High-rise vs. Warehouse).")
    progress   = st.slider("📊 Current Project Progress (%)", 0, 100, 0,
                           help="Current state of execution. AI switches to 'Rescue Mode' after 10% progress.")
    st.info(f"📅 Simulation Mode: Day {int(5.5 * progress)} of project execution.")
    
    st.markdown("### 🔬 Advanced Parameters")
    soil_risk  = st.selectbox("⛰️ Soil Risk Level", ["Stable","Moderate","Unstable"],
                              index=["Stable","Moderate","Unstable"].index(default_soil),
                              help="Geological stability. Unstable soil increases Foundation duration by 25%.")
    equipment  = st.slider("🔧 Equipment Availability (%)", 10, 100, default_equip,
                           help="Serviceability of cranes, batching plants, and heavy machinery.")
    rework_rate = st.slider("🔁 Historical Rework Rate (%)", 0, 30, default_rework,
                            help="Percentage of tasks requiring redo. Directly impacts the 'Financial Loss' metrics.") / 100.0

    st.markdown("---")
    st.markdown("### 🔄 Strategy Execution")
    sync_ai = st.toggle("🛰️ Sync AI Strategy to Digital Twin", value=False,
                        help="LINK the AI's optimized 'Magic Cards' directly to your Gantt Chart. Turn this on to see the project shrink!")
    
    st.markdown("---")
    st.markdown("### 💰 Project Parameters")
    budget_cr    = st.number_input("Budget (₹ Crore)", min_value=10.0, max_value=100000.0, value=500.0, step=10.0,
                                   help="The total estimated cost of the project. This helps the AI calculate the standard timeline for projects of this scale.")
    site_area    = st.number_input("Site Area (sqm)", min_value=500, max_value=5000000, value=25000, step=500,
                                   help="The physical size of your construction site. Larger areas increase logistical complexity and the overall carbon footprint.")
    project_name = st.text_input("Project Name", value="L&T Infrastructure Project Alpha")

# ── ENCODINGS ─────────────────────────────────────────────────────────────────
w_map = {"Sunny":0,"Rainy":1,"Humid":2}
c_map = {"Low":1,"Medium":2,"High":3}
m_map = {"No":0,"Yes":1}
s_map = {"Stable":0,"Moderate":1,"Unstable":2}

base_input = {
    "weather":        w_map[weather],
    "labor":          labor,
    "material_delay": m_map[mat_delay],
    "complexity":     c_map[complexity],
    "progress":       progress,
    "soil_risk":      s_map[soil_risk],
    "equipment":      equipment,
    "rework_rate":    rework_rate,
}

input_df        = pd.DataFrame([base_input])
predicted_delay = float(np.clip(model.predict(input_df)[0], 0, None))
# Make confidence more dynamic based on model accuracy and site complexity
confidence      = accuracy * (1.0 - (c_map[complexity]/50.0)) - (np.random.uniform(0.01, 0.03))
risk            = "CRITICAL" if predicted_delay > 30 else "HIGH" if predicted_delay > 15 else "MEDIUM" if predicted_delay > 8 else "LOW"
risk_color      = {"CRITICAL":"#EF4444","HIGH":"#F97316","MEDIUM":"#FBBF24","LOW":"#10B981"}[risk]

# ── HEADER ────────────────────────────────────────────────────────────────────
crisis_mode = st.session_state.get("crisis_mode", False)

if crisis_mode:
    st.markdown('<div class="crisis-banner">🚨 CRISIS MODE ACTIVE — Site emergency scenario triggered. All panels reflect worst-case conditions.</div>', unsafe_allow_html=True)

st.markdown('<div class="main-title">🏗️ DynaConstructa.AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Adaptive Digital Twin · Generative Design · Real-Time IoT Recalibration</div>', unsafe_allow_html=True)

# ── KPI BAR ───────────────────────────────────────────────────────────────────
# Calculate Finish Metrics (Sync with simulation.py K=85)
baseline_dur = 85 * (budget_cr ** 0.35)
total_expected_dur = baseline_dur + predicted_delay

remaining_days = total_expected_dur * (1 - progress/100.0)

# Calculate speed multiplier from AI recommendation if synced
speed_mult = 1.0
if sync_ai and remaining_days > 0:
    # Get the top recommended variant
    v_opt = generate_design_variants(c_map[complexity], budget_cr, site_area, current_progress_pct=progress, outcome_history=st.session_state.get("outcome_history"))[0]
    # Ratio of original remaining days vs optimized remaining days
    speed_mult = remaining_days / max(1.0, v_opt['est_duration_days'])

st.markdown("""
    <div style="display: flex; gap: 1rem; margin-bottom: 2rem;">
        <div class="metric-card" style="flex: 1; border-left: 4px solid #EF4444;">
            <div style="font-size: 0.8rem; color: #94A3B8;">TOTAL DELAY</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: #F8FAFC;">{delay:.1f} <span style="font-size: 1rem; color: #EF4444;">days</span></div>
        </div>
        <div class="metric-card" style="flex: 1; border-left: 4px solid #38BDF8;">
            <div style="font-size: 0.8rem; color: #94A3B8;">AI CONFIDENCE</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: #F8FAFC;">{conf:.1f}%</div>
        </div>
        <div class="metric-card" style="flex: 1; border-left: 4px solid #F59E0B;">
            <div style="font-size: 0.8rem; color: #94A3B8;">BUDGET CONSUMED</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: #F8FAFC;">₹{budget:.1f} <span style="font-size: 1rem; color: #94A3B8;">Cr</span></div>
        </div>
        <div class="metric-card" style="flex: 1; border-left: 4px solid #10B981;">
            <div style="font-size: 0.8rem; color: #94A3B8;">REMAINING TIME</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: #F8FAFC;">{rem} <span style="font-size: 1rem; color: #94A3B8;">days</span></div>
        </div>
    </div>
""".format(
    delay=predicted_delay, 
    conf=confidence*100, 
    budget=budget_cr * (progress/100.0), 
    rem=int(remaining_days)
), unsafe_allow_html=True)
st.markdown(
    f"<div style='text-align:right;margin-top:-10px'>"
    f"<span style='background:{risk_color}22;color:{risk_color};padding:4px 14px;"
    f"border-radius:20px;font-weight:600;font-size:0.9rem'>⬤ RISK: {risk}</span></div>",
    unsafe_allow_html=True
)
st.markdown("---")

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📊 Command Centre",
    "🎲 Monte Carlo & CPM",
    "🧬 Generative Design",
    "📡 Live IoT Sensor Feed",
    "🛰️ BIM & Vision Twin",
    "🔬 AI Science & Provenance",
    "🌍 Supply Chain & ESG",
    "⚡ Autonomous Execution",
])

# ── CSS FOR HORIZONTAL TABS (PREMIUM SCROLL) ──
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        overflow-x: auto;
        overflow-y: hidden;
        white-space: nowrap;
        flex-wrap: nowrap !important;
        scrollbar-width: thin;
        scrollbar-color: #38BDF8 #020617;
    }
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
        height: 4px;
    }
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar-thumb {
        background: #38BDF8;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — COMMAND CENTRE
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    left, right = st.columns([1.3, 1])

    with left:
        st.markdown('''
            <div class="section-header">
                📈 Delay Benchmark Analysis
                <div class="tooltip">ⓘ<span class="tooltiptext">Compares your project's predicted delay against global standards and L&T's own high-efficiency benchmarks.</span></div>
            </div>
        ''', unsafe_allow_html=True)
        cats   = ["World-Class\nTarget","L&T\nBenchmark","Industry\nAverage","Your\nPrediction"]
        vals   = [3.0, 5.0, 11.0, predicted_delay]
        colors = ["#10B981","#3B82F6","#FBBF24", risk_color]
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(x=cats, y=vals, marker_color=colors,
                                  text=[f"{v:.1f}d" for v in vals], textposition="outside", width=0.5))
        fig_bar.update_layout(paper_bgcolor="#0B0F19", plot_bgcolor="#111827",
                               font=dict(color="#F9FAFB"),
                               yaxis=dict(title="Delay (Days)", gridcolor="#1F2937"),
                               xaxis=dict(gridcolor="#1F2937"),
                               showlegend=False, height=300, margin=dict(t=20,b=10))
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown('''
            <div class="section-header">
                🧠 SHAP Analysis (Why THIS project?)
                <div class="tooltip">ⓘ<span class="tooltiptext">Unlike global feature importance, this shows exactly how each factor on your specific site pushes the delay up or down.</span></div>
            </div>
        ''', unsafe_allow_html=True)
        # Calculate SHAP values for current input
        shap_values = explainer(input_df)
        base_value = explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = base_value[0]
            
        sv = shap_values.values[0]
        
        # Sort by absolute SHAP value
        sorted_idx = np.argsort(np.abs(sv))
        sorted_features = np.array(feature_names)[sorted_idx]
        sorted_sv = sv[sorted_idx]
        
        # Create Waterfall
        measures = ["relative"] * len(sorted_sv) + ["total"]
        y_labels = list(sorted_features) + ["Total Delay Prediction"]
        x_values = list(sorted_sv) + [0] # total doesn't need a relative value
        
        fig_shap = go.Figure(go.Waterfall(
            name="SHAP", orientation="h",
            measure=measures,
            y=y_labels,
            x=x_values,
            connector={"line":{"color":"#374151"}},
            decreasing={"marker":{"color":"#10B981"}},
            increasing={"marker":{"color":"#EF4444"}},
            totals={"marker":{"color":"#F59E0B"}},
            text=[f"{v:+.1f}d" if m=="relative" else f"{base_value+sum(sorted_sv):.1f}d" for m, v in zip(measures, x_values)],
            textposition="outside"
        ))
        fig_shap.update_layout(
            paper_bgcolor="#0B0F19", plot_bgcolor="#111827",
            font=dict(color="#F9FAFB"),
            xaxis=dict(title="Delay Contribution (Days)", gridcolor="#1F2937"),
            yaxis=dict(gridcolor="#1F2937"),
            height=320, margin=dict(t=10,b=10),
            showlegend=False
        )
        st.plotly_chart(fig_shap, use_container_width=True)

    with right:
        st.markdown('''
            <div class="section-header">
                🤖 Adaptive Engineering Recommendations
                <div class="tooltip">ⓘ<span class="tooltiptext">AI-generated action plans to mitigate risks and speed up the project based on current site conditions.</span></div>
            </div>
        ''', unsafe_allow_html=True)
        recs, cost_impact = generate_recommendation(
            w_map[weather], labor, m_map[mat_delay], c_map[complexity],
            predicted_delay, s_map[soil_risk], equipment, rework_rate
        )
        tag = {"CRITICAL":"tag-critical","HIGH":"tag-high","MEDIUM":"tag-medium","OK":"tag-ok"}
        for severity, msg, _ in recs:
            color = {"CRITICAL":"#EF4444","HIGH":"#F97316","MEDIUM":"#FBBF24","OK":"#10B981"}[severity]
            st.markdown(f"<span style='background:{color}22;color:{color};padding:2px 8px;border-radius:4px;font-size:0.75rem;font-weight:600'>{severity}</span> &nbsp;{msg}<br>", unsafe_allow_html=True)

        st.markdown('''
            <div class="section-header">
                💰 Financial Impact (Auditable)
                <div class="tooltip">ⓘ<span class="tooltiptext">A detailed breakdown of how delays and reworks translate into real money lost, and how much the AI can help you recover.</span></div>
            </div>
        ''', unsafe_allow_html=True)
        fin = compute_financials(predicted_delay, budget_cr, rework_rate)

        st.markdown("**📉 What this project is losing:**")
        loss_df = pd.DataFrame({
            "Loss Category": [
                "Delay Cost (idle labour, equipment, penalties)",
                "Rework Cost (redoing completed work)",
                "Material Waste (from rework-driven over-ordering)",
                "🔴 TOTAL PROJECT LOSS",
            ],
            "Amount": [
                f"₹{fin['delay_loss']} Cr",
                f"₹{fin['rework_loss']} Cr",
                f"₹{fin['material_waste']} Cr",
                f"₹{fin['total_loss']} Cr",
            ],
            "Basis": [
                f"{predicted_delay:.1f}d × ₹{budget_cr:.0f}Cr × 0.8%/day",
                f"₹{budget_cr:.0f}Cr × {rework_rate*100:.0f}% rework rate",
                f"Rework cost × 18% waste factor",
                "",
            ]
        })
        st.dataframe(loss_df, use_container_width=True, hide_index=True)

        st.markdown("**📈 What DynaConstructa.AI recovers:**")
        save_df = pd.DataFrame({
            "Saving Category": [
                "Delay avoided (48–72hr early warning)",
                "Rework eliminated (design-reality gap closed)",
                "Material waste avoided",
                "🟢 TOTAL SAVINGS",
            ],
            "Amount": [
                f"₹{fin['delay_saving']} Cr",
                f"₹{fin['rework_saving']} Cr",
                f"₹{fin['waste_saving']} Cr",
                f"₹{fin['total_saving']} Cr",
            ],
            "Basis": [
                "55% of delay cost avoided with early action",
                "60% of rework eliminated pre-execution",
                "50% of waste avoided through better planning",
                f"Net of ₹{fin['tool_cost']} Cr platform cost → ROI: {fin['roi_pct']}x",
            ]
        })
        st.dataframe(save_df, use_container_width=True, hide_index=True)
        st.success(f"💡 On this ₹{budget_cr:.0f} Cr project: DynaConstructa saves ₹{fin['total_saving']} Cr vs a platform cost of ₹{fin['tool_cost']} Cr — a {fin['roi_pct']}x ROI.")

        st.markdown('''
            <div class="section-header">
                📊 Project Health Radar
                <div class="tooltip">ⓘ<span class="tooltiptext">A quick visual check of the project across 5 key areas: Schedule, Cost, Quality, Safety, and Resources.</span></div>
            </div>
        ''', unsafe_allow_html=True)
        radar_cats  = ["Schedule","Cost","Quality","Safety","Resources"]
        scores = [
            max(0, 10 - predicted_delay/5),
            max(0, 10 - cost_impact/10),
            max(0, 10 - rework_rate*50),
            max(0, 10 - s_map[soil_risk]*3 - (1 if weather != "Sunny" else 0)),
            (labor + equipment) / 20,
        ]
        fig_radar = go.Figure(go.Scatterpolar(
            r=scores+[scores[0]], theta=radar_cats+[radar_cats[0]],
            fill="toself", fillcolor="rgba(245,158,11,0.15)",
            line=dict(color="#F59E0B", width=2),
        ))
        fig_radar.update_layout(
            polar=dict(bgcolor="#111827",
                       radialaxis=dict(visible=True, range=[0,10], gridcolor="#1F2937", color="#9CA3AF"),
                       angularaxis=dict(gridcolor="#1F2937", color="#F9FAFB")),
            paper_bgcolor="#0B0F19", font=dict(color="#F9FAFB"),
            height=280, margin=dict(t=20,b=10), showlegend=False,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — MONTE CARLO & CPM
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('''
        <div class="section-header">
            🎲 Monte Carlo Simulation (1,000 Scenarios)
            <div class="tooltip">ⓘ<span class="tooltiptext">Runs 1,000 'what-if' scenarios to calculate the probability of different outcomes, giving you a range of best and worst-case dates.</span></div>
        </div>
    ''', unsafe_allow_html=True)
    with st.spinner("Running 1,000 probabilistic simulations..."):
        mc = monte_carlo_simulation(model, base_input, budget_cr, n_simulations=1000)

    mc1,mc2,mc3,mc4 = st.columns(4)
    mc1.metric("P10 — Best Case",   f"{mc['p10']:.1f} days")
    mc2.metric("P50 — Most Likely", f"{mc['p50']:.1f} days")
    mc3.metric("P90 — Worst Case",  f"{mc['p90']:.1f} days")
    mc4.metric("Prob >15d Overrun", f"{mc['prob_overrun_15']:.1f}%")

    col_hist1, col_hist2 = st.columns(2)
    with col_hist1:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=mc["simulations"], nbinsx=50,
                                         marker_color="#3B82F6", opacity=0.8))
        for x, label, color in [
            (mc["p10"],"P10","#10B981"), (mc["p50"],"P50","#FBBF24"),
            (mc["p90"],"P90","#EF4444"), (predicted_delay,f"AI: {predicted_delay:.1f}d","#F59E0B")
        ]:
            fig_hist.add_vline(x=x, line_dash="dash", line_color=color, annotation_text=label)
        fig_hist.update_layout(title="Probabilistic Delay Distribution", paper_bgcolor="#0B0F19", plot_bgcolor="#111827",
                                font=dict(color="#F9FAFB"),
                                xaxis=dict(title="Delay (Days)", gridcolor="#1F2937"),
                                yaxis=dict(title="Frequency", gridcolor="#1F2937"),
                                height=350, margin=dict(t=40,b=20), bargap=0.05)
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with col_hist2:
        fig_cost = go.Figure()
        fig_cost.add_trace(go.Histogram(x=mc["cost_simulations"], nbinsx=50,
                                         marker_color="#10B981", opacity=0.8))
        for x, label, color in [
            (mc["cp10"],"P10","#3B82F6"), (mc["cp50"],"P50","#FBBF24"),
            (mc["cp90"],"P90","#EF4444")
        ]:
            fig_cost.add_vline(x=x, line_dash="dash", line_color=color, annotation_text=label)
        fig_cost.update_layout(title="Probabilistic Cost Impact (₹ Cr)", paper_bgcolor="#0B0F19", plot_bgcolor="#111827",
                                font=dict(color="#F9FAFB"),
                                xaxis=dict(title="Total Cost (₹ Cr)", gridcolor="#1F2937"),
                                yaxis=dict(title="Frequency", gridcolor="#1F2937"),
                                height=350, margin=dict(t=40,b=20), bargap=0.05)
        st.plotly_chart(fig_cost, use_container_width=True)

    st.markdown('''
        <div class="section-header">
            🗺️ Critical Path Method (CPM)
            <div class="tooltip">ⓘ<span class="tooltiptext">Identifies the most important sequence of tasks that determines the project's finish date. Any delay in these 'Critical' tasks delays the whole project.</span></div>
        </div>
    ''', unsafe_allow_html=True)
    
    # Using global speed_mult calculated above

    G, critical, tasks, proj_duration = compute_critical_path(predicted_delay, c_map[complexity], budget_cr, speed_multiplier=speed_mult)

    cc1,cc2 = st.columns(2)
    cc1.metric("Total Project Duration", f"{proj_duration} days")
    cc2.metric("Critical Tasks", f"{len(critical)} of {len(tasks)}")

    gantt_data = []
    for task, info in tasks.items():
        node = G.nodes[task]
        # Calculate Year/Month/Day for Gantt (Start May 1, 2025)
        total_days = int(node["ES"])
        year = 2025 + (total_days // 365)
        month = 5 + ((total_days % 365) // 30)
        if month > 12:
            year += (month - 1) // 12
            month = (month - 1) % 12 + 1
        day = (total_days % 30) + 1
        
        total_days_f = int(node["EF"])
        year_f = 2025 + (total_days_f // 365)
        month_f = 5 + ((total_days_f % 365) // 30)
        if month_f > 12:
            year_f += (month_f - 1) // 12
            month_f = (month_f - 1) % 12 + 1
        day_f = (total_days_f % 30) + 1
        
        gantt_data.append(dict(
            Task=task,
            Start=f"{year}-{month:02d}-{day:02d}",
            Finish=f"{year_f}-{month_f:02d}-{day_f:02d}",
            Resource="Critical" if task in critical else "Non-Critical"
        ))
    fig_gantt = ff.create_gantt(gantt_data, colors={"Critical":"#EF4444","Non-Critical":"#3B82F6"},
                                 index_col="Resource", show_colorbar=True,
                                 group_tasks=True, showgrid_x=True, showgrid_y=True)
    fig_gantt.update_layout(paper_bgcolor="#0B0F19", plot_bgcolor="#111827",
                             font=dict(color="#F9FAFB"), height=420, margin=dict(t=20,b=20))
    st.plotly_chart(fig_gantt, use_container_width=True)
    st.info(f"🔴 **Critical Path:** {' → '.join(critical)}")

    # Calculate baseline tasks to compare for "Saved" days
    _, _, baseline_tasks, _ = compute_critical_path(predicted_delay, c_map[complexity], budget_cr, speed_multiplier=1.0)

    slack_rows = []
    action_plan = {}
    if sync_ai and 'v_opt' in locals():
        action_plan = v_opt.get("action_plan", {})
        
    for task in tasks:
        n = G.nodes[task]
        # Calculate reduction if any
        base_dur = baseline_tasks[task]["duration"]
        reduction = base_dur - n['duration']
        
        row = {
            "Task": task, 
            "Duration": f"{n['duration']:.1f}d",
            "ES": f"Day {n['ES']:.0f}", 
            "EF": f"Day {n['EF']:.0f}",
            "Slack": f"{n['slack']:.1f}d",
            "Critical": "🔴 YES" if task in critical else "🟢 No"
        }
        
        if sync_ai and reduction > 0.1:
            row["⚡ Days Reduced"] = f"-{reduction:.1f}d"
            # Add prescriptive action
            action = action_plan.get(task, "AI optimized resource scheduling")
            row["🛠️ AI Action Taken"] = action
        
        slack_rows.append(row)
    
    df_slack = pd.DataFrame(slack_rows)
    
    # Eye-catching styling for the reduction column
    if "⚡ Days Reduced" in df_slack.columns:
        # Reorder columns for visibility
        cols = list(df_slack.columns)
        if "⚡ Days Reduced" in cols:
            cols.insert(cols.index("Duration") + 1, cols.pop(cols.index("⚡ Days Reduced")))
        if "🛠️ AI Action Taken" in cols:
            cols.insert(cols.index("⚡ Days Reduced") + 1, cols.pop(cols.index("🛠️ AI Action Taken")))
            
        df_slack = df_slack[cols]
            
        st.dataframe(
            df_slack.style.apply(lambda x: ['background-color: rgba(16, 185, 129, 0.2); color: #10B981; font-weight: bold' 
                                          if x.name == '⚡ Days Reduced' else '' for _ in x], axis=0),
            use_container_width=True, 
            hide_index=True
        )
    else:
        st.dataframe(df_slack, use_container_width=True, hide_index=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — GENERATIVE DESIGN
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    # 🎊 Celebration State for 100% Progress
    if progress >= 100:
        st.balloons()
        st.markdown("""
        <div style="background: linear-gradient(135deg, #10B981 0%, #059669 100%); padding: 3rem; border-radius: 20px; text-align: center; margin: 2rem 0; box-shadow: 0 20px 40px rgba(16, 185, 129, 0.2);">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🏆</div>
            <h1 style="color: white; margin: 0; font-size: 2.5rem; font-weight: 800;">MISSION ACCOMPLISHED</h1>
            <p style="color: #ECFDF5; font-size: 1.2rem; margin-top: 1rem; opacity: 0.9;">
                Project successfully delivered. All structural nodes verified. <br>
                <b>Final Status:</b> 100% Operational · 0 Safety Incidents · AI-Optimized Handover Ready.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    if progress == 0:
        st.markdown('''
            <div class="section-header">
                🧬 Initial Generative Design — 3 Baseline Variants
                <div class="tooltip">ⓘ<span class="tooltiptext">Uses AI to create multiple alternative project plans (variants) that optimize for speed, cost, or sustainability.</span></div>
            </div>
        ''', unsafe_allow_html=True)
        st.caption("Select the optimal project DNA before execution begins.")
    else:
        st.markdown(f'''
            <div class="section-header">
                🔄 Mid-Project Rescue Optimization — {100-progress}% Work Remaining
                <div class="tooltip">ⓘ<span class="tooltiptext">Sensors detected deviation. AI is now generating 'Rescue Strategies' to optimize the remaining scope.</span></div>
            </div>
        ''', unsafe_allow_html=True)
        st.caption(f"Sensors detected deviation at {progress}% progress. AI is now generating 'Rescue Strategies' to optimize the remaining scope.")

    variants = generate_design_variants(c_map[complexity], budget_cr, site_area, current_progress_pct=progress, outcome_history=st.session_state.get("outcome_history"))

    for v in variants:
        card_cls = "variant-card-recommended" if v["recommended"] else "variant-card"
        badge    = " &nbsp;<span style='background:#F59E0B;color:#000;padding:2px 8px;border-radius:4px;font-size:0.75rem;font-weight:700'>AI RECOMMENDED PIVOT</span>" if v["recommended"] and progress > 0 else " &nbsp;<span style='background:#F59E0B;color:#000;padding:2px 8px;border-radius:4px;font-size:0.75rem;font-weight:700'>AI RECOMMENDED</span>" if v["recommended"] else ""
        st.markdown(f"""
        <div class="{card_cls}">
            <div style="font-size:1.05rem;font-weight:600;color:#F9FAFB">{v['name']}{badge}</div>
            <div style="font-size:0.85rem;color:#9CA3AF;margin:6px 0">{v['description']}</div>
            <div style="display:flex;gap:2rem;margin-top:10px;flex-wrap:wrap">
                <div><div style="font-size:0.75rem;color:#6B7280">Est. Remaining Cost</div><div style="font-size:1.1rem;font-weight:600;color:#F59E0B">₹{v['est_cost_cr']} Cr</div></div>
                <div><div style="font-size:0.75rem;color:#6B7280">Remaining Duration</div><div style="font-size:1.1rem;font-weight:600;color:#38BDF8">{v['est_duration_days']:.0f} days</div></div>
                <div><div style="font-size:0.75rem;color:#10B981;font-weight:700">⚡ Time Saved</div><div style="font-size:1.1rem;font-weight:600;color:#10B981">{max(0, int(remaining_days - v['est_duration_days']))} days</div></div>
                <div><div style="font-size:0.75rem;color:#6B7280">Rework Risk</div><div style="font-size:1.1rem;font-weight:600;color:#F87171">{v['rework_risk']}</div></div>
                <div><div style="font-size:0.75rem;color:#6B7280">Est. Carbon Footprint</div><div style="font-size:1.1rem;font-weight:600;color:#94A3B8">{v['carbon_t']} T CO₂</div></div>
                <div><div style="font-size:0.75rem;color:#6B7280">Strategy Score</div><div style="font-size:1.1rem;font-weight:600;color:#{'10B981' if v['score']>85 else 'FBBF24'}">{v['score']}/100</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('''
        <div class="section-header">
            📊 Variant Comparison Radar
            <div class="tooltip">ⓘ<span class="tooltiptext">Compares the different AI-generated plans side-by-side to help you pick the best strategy.</span></div>
        </div>
    ''', unsafe_allow_html=True)
    cats_r   = ["Cost Efficiency","Speed","Quality","Sustainability","AI Score"]
    v_colors = ["#3B82F6","#F59E0B","#10B981"]
    fig_cmp  = go.Figure()
    for i, v in enumerate(variants):
        # Extract percentage from string like "5.0%"
        risk_val = float(v["rework_risk"].replace("%",""))
        
        # Recalibrate radar scores for professional demo impact
        time_saved = max(0, baseline_dur - v["est_duration_days"])
        # 10/10 speed = 40% time saved (our modeled max efficiency)
        dur_score = round(np.clip((time_saved / (baseline_dur * 0.4)) * 10, 0, 10), 1)
        
        # 10/10 cost = at or below budget. Penalty for overruns.
        cost_overrun = v["est_cost_cr"] - budget_cr
        cost_score = round(np.clip(10 - (max(0, cost_overrun)/(budget_cr*0.2))*10, 0, 10), 1)
        
        sc = [
            cost_score,
            dur_score,
            round(np.clip(10-(risk_val/2), 0, 10), 1),
            round(np.clip(10-(v["carbon_t"]/(site_area*0.12*(budget_cr/100.0)))*10, 0, 10), 1) if site_area > 0 else 10.0,
            round(v["score"]/10,1),
        ]


        fig_cmp.add_trace(go.Scatterpolar(
            r=sc+[sc[0]], theta=cats_r+[cats_r[0]],
            fill="toself", name=v["name"].split("—")[0].strip(),
            line=dict(color=v_colors[i]),
            fillcolor=f"rgba({int(v_colors[i][1:3],16)},{int(v_colors[i][3:5],16)},{int(v_colors[i][5:7],16)},0.1)"
        ))
    fig_cmp.update_layout(
        polar=dict(bgcolor="#111827",
                   radialaxis=dict(visible=True, range=[0,10], gridcolor="#1F2937", color="#9CA3AF"),
                   angularaxis=dict(gridcolor="#1F2937", color="#F9FAFB")),
        paper_bgcolor="#0B0F19", font=dict(color="#F9FAFB"),
        legend=dict(bgcolor="#111827", bordercolor="#1F2937"),
        height=380, margin=dict(t=20,b=20),
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — LIVE IOT SENSOR FEED
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    # 📡 Delivery State for 100% Progress
    if progress >= 100:
        st.markdown("""
        <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid #10B981; padding: 2rem; border-radius: 15px; text-align: center; margin-bottom: 2rem;">
            <h3 style="color: #10B981; margin: 0;">📡 Final Structural Verification: SUCCESS</h3>
            <p style="color: #9CA3AF; margin-top: 0.5rem;">All IoT sensors have been decommissioned or transitioned to facility management mode.</p>
        </div>
        """, unsafe_allow_html=True)
        # Show static final readings
        c1, c2, c3 = st.columns(3)
        c1.metric("Final Concrete Strength", "42.5 MPa", "Verified")
        c2.metric("Structural Load", "82.0%", "Stable")
        c3.metric("System Health", "100%", "Optimal")
        st.stop()

    st.markdown('''
        <div class="section-header">
            📡 Real-Time IoT Sensor Feed — Construction Site
            <div class="tooltip">ⓘ<span class="tooltiptext">Live data from site sensors monitoring concrete strength, temperature, and structural safety.</span></div>
        </div>
    ''', unsafe_allow_html=True)


    # If crisis mode — spike the sensors dramatically
    if crisis_mode and "iot_state" not in st.session_state:
        st.session_state.iot_state = {
            "concrete_strength": 19.5, "ambient_temp": 44.2,
            "humidity": 91.0, "structural_load": 107.0,
            "dust_ppm": 445.0, "vibration_mmps": 13.1,
            "alerts": [], "history": [],
        }

    if "iot_state" not in st.session_state:
        st.session_state.iot_state = {
            "concrete_strength": 28.0, "ambient_temp": 32.0,
            "humidity": 65.0, "structural_load": 72.0,
            "dust_ppm": 180.0, "vibration_mmps": 3.2,
            "alerts": [], "history": [],
        }

    live_toggle = st.toggle("🔴 Live Feed Active", value=crisis_mode)

    sensor_state = simulate_iot_tick(st.session_state.iot_state)
    st.session_state.iot_state.update(sensor_state)
    history = st.session_state.iot_state.get("history", [])
    history.append({k:sensor_state[k] for k in sensor_state if k != "alerts"})
    if len(history) > 60: history = history[-60:]
    st.session_state.iot_state["history"] = history

    def sensor_card(col, label, value, unit, lo_warn, hi_warn):
        color = "#EF4444" if (value > hi_warn or value < lo_warn) else "#10B981"
        col.markdown(f"""
        <div class="metric-card" style="border-top: 3px solid {color}; padding: 1.2rem;">
            <div class="iot-label" style="font-size:0.65rem;">{label}</div>
            <div class="iot-value" style="color:{color}; margin: 4px 0;">{value}</div>
            <div style="font-size:0.7rem; color:#64748B; font-weight:600;">{unit}</div>
        </div>""", unsafe_allow_html=True)

    s1,s2,s3,s4,s5,s6 = st.columns(6)
    sensor_card(s1,"Concrete Strength", sensor_state["concrete_strength"],"MPa",    22, 42)
    sensor_card(s2,"Ambient Temp",       sensor_state["ambient_temp"],     "°C",    15, 43)
    sensor_card(s3,"Humidity",           sensor_state["humidity"],         "%",     30, 88)
    sensor_card(s4,"Structural Load",    sensor_state["structural_load"],  "kN/m²", 0, 100)
    sensor_card(s5,"Dust Level",         sensor_state["dust_ppm"],         "PPM",   0, 400)
    sensor_card(s6,"Vibration",          sensor_state["vibration_mmps"],   "mm/s",  0, 12)

    alerts = sensor_state.get("alerts", [])
    if alerts:
        st.error("**⚡ Sensor Alerts — Adaptive Recalibration Triggered**")
        for a in alerts: st.warning(a)
        
        # Pillar 4: Adaptive Recalibration visualization
        st.markdown('''
            <div class="section-header">
                🔄 Recalibrated CPM Gantt & Cost Delta
                <div class="tooltip">ⓘ<span class="tooltiptext">Automatically updates the project schedule and budget risk based on real-time sensor alerts (e.g., if concrete takes longer to dry).</span></div>
            </div>
        ''', unsafe_allow_html=True)
        # Calculate base and recalibrated
        _, _, _, base_duration = compute_critical_path(predicted_delay, c_map[complexity], budget_cr, task_delays={}, speed_multiplier=speed_mult)
        
        # Inject dynamic delays based on sensor state
        task_delays = {}
        if sensor_state["structural_load"] > 100: task_delays["Structural Framing"] = 12
        if sensor_state["concrete_strength"] < 22: task_delays["Concrete Pours"] = 8
        if sensor_state["dust_ppm"] > 400: task_delays["Site Survey"] = 3
        if not task_delays: task_delays["Foundation Work"] = 5 # Default if alert but no exact match
        
        G_rec, crit_rec, tasks_rec, rec_duration = compute_critical_path(predicted_delay, c_map[complexity], budget_cr, task_delays=task_delays, speed_multiplier=speed_mult)
        
        # 🎯 AREA AFFECTED IDENTIFICATION
        st.markdown('<div style="background:rgba(239,68,68,0.1); border:1px solid #EF4444; border-radius:12px; padding:1rem; margin-bottom:1.5rem;">', unsafe_allow_html=True)
        st.markdown('<div style="color:#EF4444; font-weight:700; margin-bottom:0.5rem;">🚨 AFFECTED AREAS DETECTED:</div>', unsafe_allow_html=True)
        for task, extra in task_delays.items():
             st.markdown(f"• **{task}**: +{extra} days delay (Recalibrated from sensors)")
        st.markdown('</div>', unsafe_allow_html=True)
        
        delay_delta = rec_duration - base_duration
        cost_delta = delay_delta * budget_cr * 0.008
        
        cd1, cd2, cd3 = st.columns(3)
        cd1.metric("Original Duration", f"{base_duration} days")
        cd2.metric("Recalibrated Duration", f"{rec_duration} days", delta=f"+{delay_delta:.1f} days", delta_color="inverse")
        cd3.metric("Cost Delta (Risk)", f"₹{cost_delta:.2f} Cr", delta=f"+₹{cost_delta:.2f} Cr", delta_color="inverse")
        
        # Draw Recalibrated Gantt
        gantt_data = []
        for task, info in tasks_rec.items():
            node = G_rec.nodes[task]
            # Calculate Year/Month/Day (Baseline May 1, 2026)
            total_days = int(node["ES"])
            year = 2026 + (total_days // 365)
            month = 5 + ((total_days % 365) // 30)
            if month > 12:
                year += (month - 1) // 12
                month = (month - 1) % 12 + 1
            day = (total_days % 30) + 1
            
            total_days_f = int(node["EF"])
            year_f = 2026 + (total_days_f // 365)
            month_f = 5 + ((total_days_f % 365) // 30)
            if month_f > 12:
                year_f += (month_f - 1) // 12
                month_f = (month_f - 1) % 12 + 1
            day_f = (total_days_f % 30) + 1

            gantt_data.append(dict(
                Task=task,
                Start=f"{year}-{month:02d}-{day:02d}",
                Finish=f"{year_f}-{month_f:02d}-{day_f:02d}",
                Resource="Critical Path" if task in crit_rec else "Delayed Task" if task in task_delays else "Standard"
            ))
        fig_gantt_rec = ff.create_gantt(gantt_data, colors={"Critical Path":"#EF4444", "Delayed Task":"#F59E0B", "Standard":"#3B82F6"},
                                     index_col="Resource", show_colorbar=True,
                                     group_tasks=True, showgrid_x=True, showgrid_y=True)
        fig_gantt_rec.update_layout(paper_bgcolor="#0B0F19", plot_bgcolor="#111827",
                                 font=dict(color="#F9FAFB"), height=350, margin=dict(t=20,b=20))
        st.plotly_chart(fig_gantt_rec, use_container_width=True)
        
    else:
        st.success("✅ All sensors nominal.")

    if len(history) > 2:
        st.markdown('''
            <div class="section-header">
                📈 Sensor History
                <div class="tooltip">ⓘ<span class="tooltiptext">A trend line showing how site conditions (concrete strength, load, humidity) have changed over time.</span></div>
            </div>
        ''', unsafe_allow_html=True)
        hist_df = pd.DataFrame(history)
        fig_live = go.Figure()
        fig_live.add_trace(go.Scatter(y=hist_df["concrete_strength"], name="Concrete (MPa)", line=dict(color="#3B82F6")))
        fig_live.add_trace(go.Scatter(y=hist_df["structural_load"],   name="Load (kN/m²)",  line=dict(color="#EF4444")))
        fig_live.add_trace(go.Scatter(y=hist_df["humidity"],          name="Humidity (%)",  line=dict(color="#10B981")))
        fig_live.add_hline(y=22,  line_dash="dash", line_color="#EF4444", annotation_text="Concrete Min")
        fig_live.add_hline(y=100, line_dash="dash", line_color="#F97316", annotation_text="Load Limit")
        fig_live.update_layout(paper_bgcolor="#0B0F19", plot_bgcolor="#111827",
                                font=dict(color="#F9FAFB"), height=320,
                                xaxis=dict(title="Readings", gridcolor="#1F2937"),
                                yaxis=dict(title="Value", gridcolor="#1F2937"),
                                legend=dict(bgcolor="#111827"), margin=dict(t=10,b=10))
        st.plotly_chart(fig_live, use_container_width=True)

    if live_toggle:
        time.sleep(2)
        st.rerun()
    else:
        if st.button("🔄 Refresh Sensor Data"):
            st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — AI SCIENCE & PROVENANCE
# ═════════════════════════════════════════════════════════════════════════════
# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — BIM & VISION TWIN
# ═════════════════════════════════════════════════════════════════════════════
with tab5:
    v_left, v_right = st.columns([1, 1])

    with v_left:
        st.markdown('''
            <div class="section-header">
                📸 Reality Capture Engine
                <div class="tooltip">ⓘ<span class="tooltiptext">Advanced Computer Vision system continuously compares drone/camera feeds against the BIM schedule to detect visual delays before they show up in paperwork.</span></div>
            </div>
        ''', unsafe_allow_html=True)
        
        # Use the generated image
        image_path = r"C:\Users\amank\.gemini\antigravity\brain\749443ad-57a5-4a35-bc9c-676cf7ad6085\construction_site_wide_shot_1778322758707.png"
        st.image(image_path, use_container_width=True, caption="Site Camera #04-B: West Elevation")
        
        # CNN Results
        detections = run_cnn_inference(progress=progress)
        st.markdown("### 👁️ As-Built vs BIM Verification")
        st.caption("How do we know the schedule is slipping? The AI literally 'sees' it.")
        for d in detections:
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; background:rgba(30,41,59,0.4); padding:8px 12px; border-radius:8px; margin-bottom:6px; border-left:4px solid {d['color']};">
                <span style="color:#F8FAFC; font-weight:600;">{d['label']}</span>
                <span style="color:#94A3B8;">{d['value']}</span>
                <span style="color:{d['color']}; font-weight:700;">{d['status']}</span>
            </div>
            """, unsafe_allow_html=True)

    with v_right:
        st.markdown('''
            <div class="section-header">
                🏗️ 4D Generative BIM Twin
                <div class="tooltip">ⓘ<span class="tooltiptext">Not just a static model. When the AI recommends a 'Modular Switch' or a 'Design Pivot', this Digital Twin updates automatically to reflect the new structure.</span></div>
            </div>
        ''', unsafe_allow_html=True)
        
        # Create a 3D Building Skeleton in Plotly
        # progress determines how many floors are "built"
        n_floors = 10
        built_floors = int((progress / 100.0) * n_floors)
        
        fig_bim = go.Figure()
        
        # Draw columns and slabs for each floor
        for f in range(n_floors):
            is_built = f < built_floors
            color = "#38BDF8" if is_built else "#334155"
            opacity = 0.8 if is_built else 0.2
            
            # Floor Slab
            x = [0, 10, 10, 0, 0]
            y = [0, 0, 10, 10, 0]
            z = [f*3, f*3, f*3, f*3, f*3]
            fig_bim.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color=color, width=4), opacity=opacity, showlegend=False))
            
            # Columns (at 4 corners)
            if f < n_floors - 1:
                for cx, cy in [(0,0), (10,0), (10,10), (0,10)]:
                    fig_bim.add_trace(go.Scatter3d(x=[cx, cx], y=[cy, cy], z=[f*3, (f+1)*3], mode='lines', line=dict(color=color, width=4), opacity=opacity, showlegend=False))

        fig_bim.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(title="Floors", backgroundcolor="rgba(0,0,0,0)"),
                bgcolor="#0B0F19"
            ),
            paper_bgcolor="#0B0F19",
            margin=dict(l=0, r=0, b=0, t=0),
            height=450
        )
        st.plotly_chart(fig_bim, use_container_width=True)
        
        st.info(f"📍 **Digital Twin Status:** Building Structure at {progress}% Verification Level.")

with tab6:
    st.markdown('''
        <div class="section-header">
            📚 Data Provenance & Research Calibration
            <div class="tooltip">ⓘ<span class="tooltiptext">The real-world data sources used to train our AI model, ensuring it reflects actual Indian construction realities.</span></div>
        </div>
    ''', unsafe_allow_html=True)
    
    prov_df = pd.DataFrame({
        "Feature": ["Material Delay", "Rework Rate", "Equipment Utilisation", "Weather Profile", "Global Delay Baselines"],
        "Calibrated Value": ["44.5% occurrence", "12-18% average (Beta dist)", "75% mean, 20% std", "Mumbai Monsoon Data", "98% megaprojects overrun"],
        "Source": ["MoSPI Flash Report (2023)", "KPMG India Infra Report (2022)", "L&T Annual Report (2023)", "IMD Climate Data", "McKinsey Global Institute (2017)"]
    })
    st.dataframe(prov_df, use_container_width=True, hide_index=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('''
            <div class="section-header">
                🎯 Model Performance (5-Fold Cross Validation)
            </div>
        ''', unsafe_allow_html=True)
        fig_cv = go.Figure(go.Box(y=cv_scores, name="R² Score", marker_color="#38BDF8", boxpoints='all', jitter=0.3, pointpos=-1.8))
        fig_cv.update_layout(title=f"Mean R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}",
                             paper_bgcolor="#0B0F19", plot_bgcolor="#111827", font=dict(color="#F9FAFB"),
                             height=250, margin=dict(t=40,b=10))
        st.plotly_chart(fig_cv, use_container_width=True)
        
    with c2:
        st.markdown('''
            <div class="section-header">
                🧠 GA Adaptive Learning Curve
            </div>
        ''', unsafe_allow_html=True)
        st.caption("How the AI's Genetic Algorithm improves its recommendations over time.")
        hist = st.session_state.get("outcome_history", [])
        if hist:
            hist_df2 = pd.DataFrame(hist)
            hist_df2["Accuracy"] = 100 - (abs(hist_df2["actual_saving"] - hist_df2["predicted_saving"]) / hist_df2["predicted_saving"] * 100)
            fig_learn = go.Figure(go.Scatter(x=hist_df2["project_id"], y=hist_df2["Accuracy"], mode="lines+markers", line=dict(color="#10B981")))
            fig_learn.update_layout(title="Prediction Accuracy vs Actual Outcomes",
                                    xaxis_title="Project Sequence", yaxis_title="Accuracy (%)",
                                    paper_bgcolor="#0B0F19", plot_bgcolor="#111827", font=dict(color="#F9FAFB"),
                                    height=250, margin=dict(t=40,b=10))
            st.plotly_chart(fig_learn, use_container_width=True)

    st.markdown('''
        <div class="section-header">
            📡 Production IoT Architecture Layer
        </div>
    ''', unsafe_allow_html=True)
    st.markdown("""
    <div style="display:flex; align-items:center; justify-content:center; gap:0; flex-wrap:wrap; padding:1.5rem 0;">
        <!-- Node 1: Sensors -->
        <div style="background:#1E293B; border:2px solid #38BDF8; border-radius:12px; padding:14px 18px; text-align:center; min-width:140px;">
            <div style="font-size:1.5rem;">📡</div>
            <div style="color:#38BDF8; font-weight:700; font-size:0.8rem; margin-top:4px;">Bosch / Trimble</div>
            <div style="color:#94A3B8; font-size:0.7rem;">Site Sensors</div>
        </div>
        <!-- Arrow -->
        <div style="display:flex; flex-direction:column; align-items:center; padding:0 6px;">
            <div style="color:#38BDF8; font-size:0.65rem; font-weight:600; margin-bottom:2px;">MQTT</div>
            <div style="color:#475569; font-size:1.4rem; line-height:1;">→</div>
        </div>
        <!-- Node 2: Edge Gateway -->
        <div style="background:#1E293B; border:2px solid #F59E0B; border-radius:12px; padding:14px 18px; text-align:center; min-width:140px;">
            <div style="font-size:1.5rem;">🌐</div>
            <div style="color:#F59E0B; font-weight:700; font-size:0.8rem; margin-top:4px;">Edge Gateway</div>
            <div style="color:#94A3B8; font-size:0.7rem;">DynaConstructa</div>
        </div>
        <!-- Arrow -->
        <div style="display:flex; flex-direction:column; align-items:center; padding:0 6px;">
            <div style="color:#F59E0B; font-size:0.65rem; font-weight:600; margin-bottom:2px;">REST / WS</div>
            <div style="color:#475569; font-size:1.4rem; line-height:1;">→</div>
        </div>
        <!-- Node 3: Anomaly Engine -->
        <div style="background:#450A0A; border:2px solid #EF4444; border-radius:12px; padding:14px 18px; text-align:center; min-width:140px; box-shadow: 0 0 15px rgba(239,68,68,0.2);">
            <div style="font-size:1.5rem;">⚡</div>
            <div style="color:#EF4444; font-weight:700; font-size:0.8rem; margin-top:4px;">Anomaly Engine</div>
            <div style="color:#FCA5A5; font-size:0.7rem;">Real-Time Detection</div>
        </div>
        <!-- Arrow -->
        <div style="display:flex; flex-direction:column; align-items:center; padding:0 6px;">
            <div style="color:#EF4444; font-size:0.65rem; font-weight:600; margin-bottom:2px;">Trigger</div>
            <div style="color:#475569; font-size:1.4rem; line-height:1;">→</div>
        </div>
        <!-- Node 4: CPM Recalibration -->
        <div style="background:#1E293B; border:2px solid #10B981; border-radius:12px; padding:14px 18px; text-align:center; min-width:140px;">
            <div style="font-size:1.5rem;">🔄</div>
            <div style="color:#10B981; font-weight:700; font-size:0.8rem; margin-top:4px;">CPM Recalibration</div>
            <div style="color:#94A3B8; font-size:0.7rem;">Auto-Update Gantt</div>
        </div>
        <!-- Arrow -->
        <div style="display:flex; flex-direction:column; align-items:center; padding:0 6px;">
            <div style="color:#10B981; font-size:0.65rem; font-weight:600; margin-bottom:2px;">Alert</div>
            <div style="color:#475569; font-size:1.4rem; line-height:1;">→</div>
        </div>
        <!-- Node 5: PM Dashboard -->
        <div style="background:#1E293B; border:2px solid #8B5CF6; border-radius:12px; padding:14px 18px; text-align:center; min-width:140px;">
            <div style="font-size:1.5rem;">📊</div>
            <div style="color:#8B5CF6; font-weight:700; font-size:0.8rem; margin-top:4px;">PM Dashboard</div>
            <div style="color:#94A3B8; font-size:0.7rem;">Decision & Action</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.info("The simulation runs on Gaussian drift for demonstration. The production architecture shown above is designed to ingest live MQTT payloads from industry-standard hardware.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 7 — SUPPLY CHAIN & ESG CONTROL TOWER
# ═════════════════════════════════════════════════════════════════════════════
with tab7:
    sc_left, sc_right = st.columns([1.2, 1])

    with sc_left:
        st.markdown('''
            <div class="section-header">
                🌍 Global Logistics & JIT Supply Chain
                <div class="tooltip">ⓘ<span class="tooltiptext">Real-time geospatial tracking of materials. AI dynamically reroutes shipments to prevent schedule delays.</span></div>
            </div>
        ''', unsafe_allow_html=True)
        
        # Determine status based on sidebar inputs
        delay_status = "CRITICAL DELAY" if mat_delay == "Yes" else "ON TIME"
        route_color = "#EF4444" if mat_delay == "Yes" and not sync_ai else "#10B981"
        if mat_delay == "Yes" and sync_ai:
            delay_status = "REROUTED BY AI"
            route_color = "#F59E0B"
            
        # Realistic Regional Supply Chain Hubs for Indian Cities
        SUPPLIER_HUBS = {
            "Mumbai": [("Pune Steel Hub", 18.5204, 73.8567), ("Nashik Silos", 19.9975, 73.7898), ("Surat Precast", 21.1702, 72.8311)],
            "Delhi": [("Panipat Steel", 29.3909, 76.9635), ("Alwar Cement", 27.5530, 76.6346), ("Ghaziabad Precast", 28.6692, 77.4538)],
            "Bangalore": [("Salem Steel", 11.6643, 78.1460), ("Mysore Cement", 12.2958, 76.6394), ("Hosur Precast", 12.7409, 77.8253)],
            "Chennai": [("Sriperumbudur Steel", 12.9675, 79.9466), ("Ariyalur Cement", 11.1401, 79.0786), ("Tiruvallur Precast", 13.1438, 79.9071)],
            "Hyderabad": [("Bolarum Steel", 17.5186, 78.5036), ("Nalgonda Cement", 17.0500, 79.2700), ("Patancheru Precast", 17.5287, 78.2667)],
            "Pune": [("Mumbai Port Steel", 18.9667, 72.8333), ("Solapur Cement", 17.6599, 75.9064), ("Chakan Precast", 18.7500, 73.8500)],
            "Ahmedabad": [("Hazira Steel", 21.1235, 72.6375), ("Ambuja Nagar", 20.8167, 70.8333), ("Sanand Precast", 22.9833, 72.3833)],
            "Kolkata": [("Durgapur Steel", 23.5204, 87.3119), ("Asansol Cement", 23.6739, 86.9524), ("Haldia Precast", 22.0625, 88.0673)],
            "Surat": [("Hazira Steel", 21.1235, 72.6375), ("Bharuch Cement", 21.7051, 72.9959), ("Navsari Precast", 20.9467, 72.9520)],
            "Jaipur": [("Bhiwadi Steel", 28.2104, 76.8407), ("Beawar Cement", 26.1039, 74.3160), ("Ajmer Precast", 26.4499, 74.6399)],
        }
        site_lat, site_lon = CITY_COORDS.get(city, (19.0760, 72.8777))
        hubs = SUPPLIER_HUBS.get(city, SUPPLIER_HUBS["Mumbai"])

        # Dynamic Rerouting Logic
        active_hubs = list(hubs)
        rerouted = False
        if mat_delay == "Yes" and sync_ai:
            active_hubs[0] = ("Secondary Steel Hub (Rerouted)", hubs[0][1] + 0.5, hubs[0][2] + 0.5)
            rerouted = True

        lats = [site_lat, active_hubs[0][1], active_hubs[1][1], active_hubs[2][1]]
        lons = [site_lon, active_hubs[0][2], active_hubs[1][2], active_hubs[2][2]]
        names = [f"Project Site ({city})", active_hubs[0][0], active_hubs[1][0], active_hubs[2][0]]

        # --- LEGEND ---
        st.markdown("""
        <div style="display:flex; gap:20px; margin-bottom:15px; background:rgba(30,41,59,0.5); padding:10px; border-radius:8px; border:1px solid rgba(56,189,248,0.2);">
            <div style="display:flex; align-items:center; gap:8px;">
                <div style="width:12px; height:12px; border-radius:50%; background:#38BDF8;"></div>
                <span style="color:#F9FAFB; font-size:0.8rem;">Project Site</span>
            </div>
            <div style="display:flex; align-items:center; gap:8px;">
                <div style="width:12px; height:12px; border-radius:50%; background:#F9FAFB;"></div>
                <span style="color:#F9FAFB; font-size:0.8rem;">Industrial Hub</span>
            </div>
            <div style="display:flex; align-items:center; gap:8px;">
                <div style="width:20px; height:2px; background:#10B981;"></div>
                <span style="color:#F9FAFB; font-size:0.8rem;">Healthy Route</span>
            </div>
            <div style="display:flex; align-items:center; gap:8px;">
                <div style="width:20px; height:2px; background:#EF4444;"></div>
                <span style="color:#F9FAFB; font-size:0.8rem;">Delayed Route</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        fig_map = go.Figure()
        
        # Add Nodes
        fig_map.add_trace(go.Scattermap(
            lat=lats, lon=lons, mode='markers+text',
            marker=go.scattermap.Marker(size=[20, 12, 12, 12], color=["#38BDF8", "#F9FAFB", "#F9FAFB", "#F9FAFB"]),
            text=names, textposition="bottom right",
            textfont=dict(color="white", size=12),
            name="Facilities"
        ))
        
        # Add Routes
        for i in range(1, 4):
            line_color = route_color if i == 1 else "#334155"
            line_width = 2
            
            if rerouted and i == 1: 
                line_color = "#10B981" # Rerouted to green
                line_width = 4         # Thicker line for "AI Optimized Path"
            
            fig_map.add_trace(go.Scattermap(
                lat=[lats[0], lats[i]], lon=[lons[0], lons[i]],
                mode='lines', line=dict(width=line_width, color=line_color),
                name=f"Route to {names[i]}"
            ))
            
            # Add a truck/shipment marker on the route
            progress_ratio = (progress % 100) / 100.0 if i != 1 else (0.2 if delay_status == "CRITICAL DELAY" and not sync_ai else 0.8)
            t_lat = lats[i] + (lats[0] - lats[i]) * progress_ratio
            t_lon = lons[i] + (lons[0] - lons[i]) * progress_ratio
            fig_map.add_trace(go.Scattermap(
                lat=[t_lat], lon=[t_lon], mode='markers',
                marker=go.scattermap.Marker(size=12, color=line_color),
                hoverinfo="text", hovertext=f"Shipment Status: {delay_status if i==1 and not sync_ai else 'ON TIME'}",
                showlegend=False
            ))

        fig_map.update_layout(
            margin={"r":0,"t":0,"l":0,"b":0},
            map=dict(
                style="carto-darkmatter",
                zoom=5.8, # Closer zoom for more detail
                center=dict(lat=site_lat + 0.5, lon=site_lon + 0.5), # Offset center to show context
            ),
            paper_bgcolor="#0B0F19",
            height=450, showlegend=False
        )
        st.plotly_chart(fig_map, use_container_width=True)
        
        # Add spacing to prevent Mapbox attribution overlap
        st.write("") 

        
        if mat_delay == "Yes" and not sync_ai:
            st.error("🚨 **SUPPLY CHAIN ALERT:** Primary steel shipment from Pune is delayed. Critical path at risk. **Turn on 'Sync AI Strategy' to resolve.**")
        elif mat_delay == "Yes" and sync_ai:
            st.warning("🔄 **AI ACTION TAKEN:** Order rerouted to secondary supplier in Nashik. Schedule integrity maintained.")
        else:
            st.success("✅ **SUPPLY CHAIN STATUS:** All JIT deliveries arriving on schedule.")

    with sc_right:
        st.markdown('''
            <div class="section-header">
                🌱 ESG & Carbon Intelligence
                <div class="tooltip">ⓘ<span class="tooltiptext">Monitors and optimizes the project's carbon footprint. L&T's commitment to Net-Zero starts here.</span></div>
            </div>
        ''', unsafe_allow_html=True)
        
        # Calculate Carbon Baseline vs Optimized
        # We must account for progress: savings only apply to the remaining work
        rem_factor = 1.0 - (progress / 100.0)
        baseline_carbon_total = (site_area * 0.12) * (budget_cr / 100.0)
        baseline_carbon_remaining = baseline_carbon_total * rem_factor
        
        current_carbon_remaining = baseline_carbon_remaining
        
        if sync_ai:
            v_opt = generate_design_variants(c_map[complexity], budget_cr, site_area, current_progress_pct=progress)[0]
            current_carbon_remaining = v_opt["carbon_t"]
            
        carbon_saved = baseline_carbon_remaining - current_carbon_remaining
        # Total projected carbon for the whole project lifecycle
        total_projected_carbon = (baseline_carbon_total * (1.0 - rem_factor)) + current_carbon_remaining

        
        # Gauge Chart for Carbon
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = total_projected_carbon,
            title = {'text': "Projected Total Carbon (Tons CO₂)", 'font': {'size': 16, 'color': '#F9FAFB'}},
            delta = {'reference': baseline_carbon_total, 'increasing': {'color': '#EF4444'}, 'decreasing': {'color': '#10B981'}},
            gauge = {
                'axis': {'range': [None, baseline_carbon_total * 1.2], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#10B981" if carbon_saved > 0 else "#38BDF8"},
                'bgcolor': "rgba(255,255,255,0.05)",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, baseline_carbon_total], 'color': "rgba(16, 185, 129, 0.1)"},
                    {'range': [baseline_carbon_total, baseline_carbon_total * 1.5], 'color': "rgba(239, 68, 68, 0.1)"}],
                'threshold': {
                    'line': {'color': "red", 'width': 2},
                    'thickness': 0.75,
                    'value': baseline_carbon_total}
            }
        ))
        fig_gauge.update_layout(paper_bgcolor="#0B0F19", font=dict(color="#F9FAFB"), height=250, margin=dict(t=40,b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        st.markdown(f"""
        <div style="background:rgba(16, 185, 129, 0.1); border:1px solid #10B981; border-radius:12px; padding:15px;">
            <div style="color:#10B981; font-weight:700; font-size:0.9rem; margin-bottom:5px;">🌿 SUSTAINABILITY IMPACT</div>
            <div style="font-size:2rem; font-weight:800; color:#F8FAFC; line-height:1;">{carbon_saved:,.0f} <span style="font-size:1rem; color:#94A3B8; font-weight:400;">Tons CO₂ Saved</span></div>
            <div style="color:#94A3B8; font-size:0.8rem; margin-top:8px;">
                Equivalent to removing <b>{int(carbon_saved * 0.22)} passenger vehicles</b> from the road for a year. 
                Achieved via generative material optimization and waste reduction.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 8 — AUTONOMOUS EXECUTION ENGINE
# ═════════════════════════════════════════════════════════════════════════════
with tab8:
    st.markdown('''
        <div class="section-header">
            ⚡ Autonomous Action Execution Log
            <div class="tooltip">ⓘ<span class="tooltiptext">Watch the AI Agent communicate with external APIs (ERP systems, Robotics, Drones) to physically execute the strategy.</span></div>
        </div>
    ''', unsafe_allow_html=True)
    
    st.markdown("<p style='color:#94A3B8; margin-bottom:20px;'>DynaConstructa doesn't just predict delays—it fixes them. Below is the live terminal log of the AI Agent dispatching commands to external hardware and enterprise software to execute the generative strategy.</p>", unsafe_allow_html=True)
    
    if not sync_ai:
        st.warning("⚠️ **Autonomous Agent Offline.** Enable 'Sync AI Strategy' in the sidebar to authorize the AI to deploy fixes.")
    else:
        # Get the specific actions from the AI
        action_plan = {}
        if 'v_opt' in locals():
            action_plan = v_opt.get("action_plan", {})
            
        import datetime
        now = datetime.datetime.now()
        
        terminal_html = f"""<section style="width:100%;"><div style="background-color:#0B0F19; border:1px solid #1E293B; border-radius:10px; overflow:hidden; box-shadow: 0 10px 25px rgba(0,0,0,0.5); margin-bottom:15px; width:100%; box-sizing:border-box;">
<div style="background-color:#1E293B; padding:8px 15px; display:flex; align-items:center;">
<div style="width:12px; height:12px; border-radius:50%; background-color:#EF4444; margin-right:8px;"></div>
<div style="width:12px; height:12px; border-radius:50%; background-color:#F59E0B; margin-right:8px;"></div>
<div style="width:12px; height:12px; border-radius:50%; background-color:#10B981; margin-right:15px;"></div>
<div style="color:#94A3B8; font-family:monospace; font-size:0.8rem;">root@dynaconstructa-agent: ~/execution_node</div>
</div>
<div style="padding:15px; font-family:'Courier New', monospace; font-size:0.85rem; line-height:1.6; max-height:500px; overflow-y:auto; color:#E2E8F0; width:100%; box-sizing:border-box;">"""
        
        def add_log(tag_color, tag_name, message, status_color, status_text, latency, t_offset):
            t = (now + datetime.timedelta(milliseconds=t_offset)).strftime("%H:%M:%S.%f")[:-3]
            return f"""<div style="display:grid; grid-template-columns: 100px 120px 1fr 100px; gap:10px; border-bottom:1px solid rgba(30,41,59,0.5); padding:8px 0; align-items:center; width:100%; box-sizing:border-box;">
<div style="color:#64748B; font-size:0.75rem;">[{t}]</div>
<div style="color:{tag_color}; font-weight:bold; font-size:0.75rem;">[{tag_name}]</div>
<div style="font-size:0.8rem; line-height:1.3; overflow:hidden; word-break:break-word;">{message}</div>
<div style="color:{status_color}; text-align:right; font-size:0.75rem; white-space:nowrap;">{status_text} <span style="color:#64748B; font-size:0.65rem;">({latency}ms)</span></div>
</div>"""

        t_off = 0
        terminal_html += add_log("#38BDF8", "SYSTEM", "Initiating Autonomous Execution Protocol...", "#38BDF8", "OK", 12, t_off)
        t_off += 45
        terminal_html += add_log("#38BDF8", "SYSTEM", "Establishing secure handshakes with L&T SAP ERP, Boston Dynamics API, and DJI Fleet Manager...", "#38BDF8", "OK", 89, t_off)
        t_off += 120
        terminal_html += add_log("#10B981", "SUCCESS", f"Handshakes verified. Executing {len(action_plan)} critical directives.", "#10B981", "READY", 4, t_off)
        t_off += 50
        
        for task, action in action_plan.items():
            t_off += int(np.random.randint(100, 800))
            lat1 = int(np.random.randint(20, 150))
            lat2 = int(np.random.randint(30, 300))
            
            if "LiDAR" in action or "Drone" in action:
                terminal_html += add_log("#F59E0B", "AGENT_DISPATCH", f"Task: {task} -> Sending payload to DJI Matrix Fleet...", "#94A3B8", "Sent", lat1, t_off)
                t_off += lat1 + 10
                terminal_html += add_log("#10B981", "EXECUTED", f"{action}", "#10B981", "In-Air", lat2, t_off)
            elif "JIT" in action or "Procurement" in task:
                terminal_html += add_log("#F59E0B", "API_CALL", f"Task: {task} -> POST /api/v2/sap_erp/logistics/reroute", "#94A3B8", "Sent", lat1, t_off)
                t_off += lat1 + 10
                terminal_html += add_log("#10B981", "EXECUTED", f"{action}", "#10B981", "PO #99812", lat2, t_off)
            elif "robot" in action.lower() or "autonomous" in action.lower():
                terminal_html += add_log("#F59E0B", "ROBOTICS_CMD", f"Task: {task} -> Uploading BIM trajectory to Boston Dynamics Spot...", "#94A3B8", "Sent", lat1, t_off)
                t_off += lat1 + 10
                terminal_html += add_log("#10B981", "EXECUTED", f"{action}", "#10B981", "Uploaded", lat2, t_off)
            elif "Concrete" in task:
                terminal_html += add_log("#F59E0B", "IOT_CMD", f"Task: {task} -> Modifying sensor polling frequency to 1Hz...", "#94A3B8", "Sent", lat1, t_off)
                t_off += lat1 + 10
                terminal_html += add_log("#10B981", "EXECUTED", f"{action}", "#10B981", "Calibrated", lat2, t_off)
            else:
                terminal_html += add_log("#F59E0B", "WORK_ORDER", f"Task: {task} -> Generating digital work order for site manager...", "#94A3B8", "Sent", lat1, t_off)
                t_off += lat1 + 10
                terminal_html += add_log("#10B981", "EXECUTED", f"{action}", "#10B981", "Delivered", lat2, t_off)
                
        t_off += 50
        terminal_html += add_log("#38BDF8", "SYSTEM", "All directives successfully dispatched. Continuous monitoring active.", "#38BDF8", "IDLE", 2, t_off)
        
        terminal_html += "</div></div></section>"
        st.markdown(terminal_html, unsafe_allow_html=True)
        
        st.success("✅ **Execution Complete.** Physical interventions are underway.")

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#4B5563;font-size:0.8rem'>"
    "DynaConstructa.AI · TeamTurbo · L&T CreaTech 2026 · Project Alpha · Site 04-B"
    "</div>",
    unsafe_allow_html=True
)