import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
import streamlit as st

from model import train_model, monte_carlo_simulation, compute_critical_path
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

    /* Remove top space and hide header */
    [data-testid="stHeader"] {
        display: none;
    }
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

model, accuracy, feature_names = load_model()

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
    budget_cr    = st.number_input("Budget (₹ Crore)", min_value=10.0, max_value=10000.0, value=500.0, step=10.0,
                                   help="The total estimated cost of the project. This helps the AI calculate the standard timeline for projects of this scale.")
    site_area    = st.number_input("Site Area (sqm)", min_value=500, max_value=500000, value=25000, step=500,
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
# Calculate Finish Metrics
baseline_dur = 55 * (budget_cr ** 0.35)
total_expected_dur = baseline_dur + predicted_delay
remaining_days = total_expected_dur * (1 - progress/100.0)

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
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Command Centre",
    "🎲 Monte Carlo & CPM",
    "🧬 Generative Design",
    "📡 Live IoT Sensor Feed",
])

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
                🧠 Feature Importance (Explainable AI)
                <div class="tooltip">ⓘ<span class="tooltiptext">Shows which factors (like weather or labor) are most responsible for the predicted delay, helping you know where to focus.</span></div>
            </div>
        ''', unsafe_allow_html=True)
        feat_df = pd.DataFrame({"Feature":feature_names,"Importance":model.feature_importances_}).sort_values("Importance")
        fig_feat = go.Figure(go.Bar(x=feat_df["Importance"], y=feat_df["Feature"], orientation="h",
                                    marker_color="#F59E0B",
                                    text=[f"{v:.3f}" for v in feat_df["Importance"]], textposition="outside"))
        fig_feat.update_layout(paper_bgcolor="#0B0F19", plot_bgcolor="#111827",
                                font=dict(color="#F9FAFB"),
                                xaxis=dict(title="Importance", gridcolor="#1F2937"),
                                yaxis=dict(gridcolor="#1F2937"),
                                height=280, margin=dict(t=10,b=10))
        st.plotly_chart(fig_feat, use_container_width=True)

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
    
    # Calculate speed multiplier from AI recommendation if synced
    speed_mult = 1.0
    if sync_ai:
        # Get the top recommended variant
        v_opt = generate_design_variants(c_map[complexity], budget_cr, site_area, current_progress_pct=progress)[0]
        # Ratio of original remaining days vs optimized remaining days
        speed_mult = remaining_days / max(1.0, v_opt['est_duration_days'])

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

    slack_rows = []
    for task in tasks:
        n = G.nodes[task]
        slack_rows.append({"Task":task, "Duration":f"{n['duration']}d",
                            "ES":f"Day {n['ES']:.0f}", "EF":f"Day {n['EF']:.0f}",
                            "Slack":f"{n['slack']:.1f}d",
                            "Critical":"🔴 YES" if task in critical else "🟢 No"})
    st.dataframe(pd.DataFrame(slack_rows), use_container_width=True, hide_index=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — GENERATIVE DESIGN
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
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

    variants = generate_design_variants(c_map[complexity], budget_cr, site_area, current_progress_pct=progress)

    for v in variants:
        card_cls = "variant-card-recommended" if v["recommended"] else "variant-card"
        badge    = " &nbsp;<span style='background:#F59E0B;color:#000;padding:2px 8px;border-radius:4px;font-size:0.75rem;font-weight:700'>AI RECOMMENDED PIVOT</span>" if v["recommended"] and progress > 0 else " &nbsp;<span style='background:#F59E0B;color:#000;padding:2px 8px;border-radius:4px;font-size:0.75rem;font-weight:700'>AI RECOMMENDED</span>" if v["recommended"] else ""
        st.markdown(f"""
        <div class="{card_cls}">
            <div style="font-size:1.05rem;font-weight:600;color:#F9FAFB">{v['name']}{badge}</div>
            <div style="font-size:0.85rem;color:#9CA3AF;margin:6px 0">{v['description']}</div>
            <div style="display:flex;gap:2rem;margin-top:10px;flex-wrap:wrap">
                <div><div style="font-size:0.75rem;color:#6B7280">Est. Remaining Cost</div><div style="font-size:1.1rem;font-weight:600;color:#F59E0B">₹{v['est_cost_cr']} Cr</div></div>
                <div><div style="font-size:0.75rem;color:#6B7280">Remaining Duration</div><div style="font-size:1.1rem;font-weight:600;color:#F59E0B">{v['est_duration_days']:.0f} days</div></div>
                <div><div style="font-size:0.75rem;color:#6B7280">⏳ Time Saved</div><div style="font-size:1.1rem;font-weight:600;color:#10B981">{max(0, int(remaining_days - v['est_duration_days']))} days</div></div>
                <div><div style="font-size:0.75rem;color:#6B7280">Rework Risk</div><div style="font-size:1.1rem;font-weight:600;color:#F59E0B">{v['rework_risk']}</div></div>
                <div><div style="font-size:0.75rem;color:#6B7280">Carbon</div><div style="font-size:1.1rem;font-weight:600;color:#F59E0B">{v['carbon_t']} T CO₂</div></div>
                <div><div style="font-size:0.75rem;color:#6B7280">Strategy Score</div><div style="font-size:1.1rem;font-weight:600;color:#{'10B981' if v['score']>88 else 'FBBF24'}">{v['score']}/100</div></div>
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
    max_d    = max(v["est_duration_days"] for v in variants)
    max_c    = max(v["est_cost_cr"] for v in variants)
    v_colors = ["#3B82F6","#F59E0B","#10B981"]
    fig_cmp  = go.Figure()
    for i, v in enumerate(variants):
        # Extract percentage from string like "5.0%"
        risk_val = float(v["rework_risk"].replace("%",""))
        # Safety check for division by zero (when progress is 100%)
        cost_score = round(10-(v["est_cost_cr"]/max_c)*10,1) if max_c > 0 else 10.0
        dur_score  = round(10-(v["est_duration_days"]/max_d)*10,1) if max_d > 0 else 10.0
        
        sc = [
            cost_score,
            dur_score,
            round(10-(risk_val/2),1),
            round(10-(v["carbon_t"]/(site_area*0.2*(budget_cr/100.0)))*10,1) if site_area > 0 else 10.0,
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

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#4B5563;font-size:0.8rem'>"
    "DynaConstructa.AI · TeamTurbo · L&T CreaTech 2026 · Project Alpha · Site 04-B"
    "</div>",
    unsafe_allow_html=True
)