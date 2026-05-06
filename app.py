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
    .main-title { font-size:2.2rem;font-weight:700;color:#F59E0B;letter-spacing:-0.5px;margin-bottom:0; }
    .sub-title  { font-size:0.95rem;color:#9CA3AF;margin-top:2px;margin-bottom:1.5rem; }
    .section-header { font-size:1rem;font-weight:600;color:#F59E0B;border-left:3px solid #F59E0B;padding-left:10px;margin:1.5rem 0 0.75rem; }
    .variant-card             { background:#111827;border:1px solid #1F2937;border-radius:10px;padding:1rem;margin-bottom:0.75rem; }
    .variant-card-recommended { background:#111827;border:2px solid #F59E0B;border-radius:10px;padding:1rem;margin-bottom:0.75rem; }
    .metric-card { background:#111827;border:1px solid #1F2937;border-radius:12px;padding:1rem 1.2rem;margin-bottom:0.5rem; }
    .iot-value { font-size:1.4rem;font-weight:700;color:#F59E0B; }
    .iot-label { font-size:0.75rem;color:#9CA3AF; }
    .crisis-banner { background:#7F1D1D;border:1px solid #EF4444;border-radius:10px;padding:12px 16px;margin-bottom:1rem;color:#FCA5A5;font-weight:500; }
    .weather-card { background:#111827;border:1px solid #1F2937;border-radius:10px;padding:10px 14px;margin-bottom:0.75rem; }
    div[data-testid="stTabs"] button { font-size:0.95rem;font-weight:500; }
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
        <div class="weather-card">
            <div style="font-size:0.8rem;color:#9CA3AF;margin-bottom:4px">Live — {wx['city']}</div>
            <div style="display:flex;gap:1rem;flex-wrap:wrap">
                <div><span style="color:#F59E0B;font-weight:600">{wx['temp_c']}°C</span><br><span style="font-size:0.7rem;color:#6B7280">Temp</span></div>
                <div><span style="color:#F59E0B;font-weight:600">{wx['humidity']}%</span><br><span style="font-size:0.7rem;color:#6B7280">Humidity</span></div>
                <div><span style="color:#F59E0B;font-weight:600">{wx['precip_mm']}mm</span><br><span style="font-size:0.7rem;color:#6B7280">Rain</span></div>
                <div><span style="color:#F59E0B;font-weight:600">{wx['wind_kmh']}km/h</span><br><span style="font-size:0.7rem;color:#6B7280">Wind</span></div>
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
                              index=["Sunny","Rainy","Humid"].index(default_weather))
    labor      = st.slider("👷 Labor Availability (%)", 10, 100, default_labor)
    mat_delay  = st.selectbox("📦 Material Supply Delay", ["No","Yes"],
                              index=["No","Yes"].index(default_mat))
    complexity = st.selectbox("🏗️ Site Complexity", ["Low","Medium","High"],
                              index=["Low","Medium","High"].index(default_complex))
    progress   = st.slider("📊 Project Progress (%)", 0, 100, 40)

    st.markdown("### 🔬 Advanced Parameters")
    soil_risk  = st.selectbox("⛰️ Soil Risk Level", ["Stable","Moderate","Unstable"],
                              index=["Stable","Moderate","Unstable"].index(default_soil))
    equipment  = st.slider("🔧 Equipment Availability (%)", 10, 100, default_equip)
    rework_rate = st.slider("🔁 Historical Rework Rate (%)", 0, 30, default_rework) / 100.0

    st.markdown("### 💰 Project Parameters")
    budget_cr    = st.number_input("Budget (₹ Crore)", min_value=10.0, max_value=10000.0, value=500.0, step=10.0)
    site_area    = st.number_input("Site Area (sqm)", min_value=500, max_value=500000, value=25000, step=500)
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
confidence      = max(0.72, min(0.98, accuracy + 0.08))
risk            = "CRITICAL" if predicted_delay > 30 else "HIGH" if predicted_delay > 15 else "MEDIUM" if predicted_delay > 8 else "LOW"
risk_color      = {"CRITICAL":"#EF4444","HIGH":"#F97316","MEDIUM":"#FBBF24","LOW":"#10B981"}[risk]

# ── HEADER ────────────────────────────────────────────────────────────────────
crisis_mode = st.session_state.get("crisis_mode", False)

if crisis_mode:
    st.markdown('<div class="crisis-banner">🚨 CRISIS MODE ACTIVE — Site emergency scenario triggered. All panels reflect worst-case conditions.</div>', unsafe_allow_html=True)

st.markdown('<div class="main-title">🏗️ DynaConstructa.AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Dynamic Engineering System · Generative Design · Predictive Simulation · Real-Time Sensor Fusion · Adaptive Recalibration</div>', unsafe_allow_html=True)
st.caption(f"📁 {project_name} &nbsp;|&nbsp; ₹{budget_cr:.0f} Cr &nbsp;|&nbsp; {site_area:,} sqm")

# ── KPI BAR ───────────────────────────────────────────────────────────────────
k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("⏱️ Predicted Delay",    f"{predicted_delay:.1f} days",
          delta=f"{predicted_delay-5:.1f} vs optimal", delta_color="inverse")
k2.metric("🎯 Model Accuracy (R²)", f"{accuracy:.3f}")
k3.metric("🤖 AI Confidence",       f"{confidence*100:.1f}%")
k4.metric("💸 Cost-at-Risk",        f"₹{predicted_delay*budget_cr*0.008:.1f} Cr")
rework_saved = round(budget_cr * rework_rate * 0.6, 1)
k5.metric("✅ Rework Savings Pot.", f"₹{rework_saved} Cr")
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
        st.markdown('<div class="section-header">📈 Delay Benchmark Analysis</div>', unsafe_allow_html=True)
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

        st.markdown('<div class="section-header">🧠 Feature Importance (Explainable AI)</div>', unsafe_allow_html=True)
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
        st.markdown('<div class="section-header">🤖 Adaptive Engineering Recommendations</div>', unsafe_allow_html=True)
        recs, cost_impact = generate_recommendation(
            w_map[weather], labor, m_map[mat_delay], c_map[complexity],
            predicted_delay, s_map[soil_risk], equipment, rework_rate
        )
        tag = {"CRITICAL":"tag-critical","HIGH":"tag-high","MEDIUM":"tag-medium","OK":"tag-ok"}
        for severity, msg, _ in recs:
            color = {"CRITICAL":"#EF4444","HIGH":"#F97316","MEDIUM":"#FBBF24","OK":"#10B981"}[severity]
            st.markdown(f"<span style='background:{color}22;color:{color};padding:2px 8px;border-radius:4px;font-size:0.75rem;font-weight:600'>{severity}</span> &nbsp;{msg}<br>", unsafe_allow_html=True)

        st.markdown('<div class="section-header">💰 Financial Impact (Auditable)</div>', unsafe_allow_html=True)
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

        st.markdown('<div class="section-header">📊 Project Health Radar</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="section-header">🎲 Monte Carlo Simulation (1,000 Scenarios)</div>', unsafe_allow_html=True)
    with st.spinner("Running 1,000 probabilistic simulations..."):
        mc = monte_carlo_simulation(model, base_input, n_simulations=1000)

    mc1,mc2,mc3,mc4 = st.columns(4)
    mc1.metric("P10 — Best Case",   f"{mc['p10']:.1f} days")
    mc2.metric("P50 — Most Likely", f"{mc['p50']:.1f} days")
    mc3.metric("P90 — Worst Case",  f"{mc['p90']:.1f} days")
    mc4.metric("Prob >15d Overrun", f"{mc['prob_overrun_15']:.1f}%")

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=mc["simulations"], nbinsx=50,
                                     marker_color="#3B82F6", opacity=0.8))
    for x, label, color in [
        (mc["p10"],"P10","#10B981"), (mc["p50"],"P50","#FBBF24"),
        (mc["p90"],"P90","#EF4444"), (predicted_delay,f"AI: {predicted_delay:.1f}d","#F59E0B")
    ]:
        fig_hist.add_vline(x=x, line_dash="dash", line_color=color, annotation_text=label)
    fig_hist.update_layout(paper_bgcolor="#0B0F19", plot_bgcolor="#111827",
                            font=dict(color="#F9FAFB"),
                            xaxis=dict(title="Delay (Days)", gridcolor="#1F2937"),
                            yaxis=dict(title="Frequency", gridcolor="#1F2937"),
                            height=350, margin=dict(t=20,b=20), bargap=0.05)
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown('<div class="section-header">🗺️ Critical Path Method (CPM)</div>', unsafe_allow_html=True)
    G, critical, tasks, proj_duration = compute_critical_path(predicted_delay, c_map[complexity])

    cc1,cc2 = st.columns(2)
    cc1.metric("Total Project Duration", f"{proj_duration} days")
    cc2.metric("Critical Tasks", f"{len(critical)} of {len(tasks)}")

    gantt_data = []
    for task, info in tasks.items():
        node = G.nodes[task]
        es = int(node["ES"]) % 28 + 1
        ef = int(node["EF"]) % 28 + 1
        if ef <= es: ef = es + 1
        gantt_data.append(dict(
            Task=task,
            Start=f"2025-05-{es:02d}",
            Finish=f"2025-05-{ef:02d}",
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
    st.markdown('<div class="section-header">🧬 AI Generative Design — 3 Structural Variants</div>', unsafe_allow_html=True)
    variants = generate_design_variants(c_map[complexity], budget_cr, site_area)

    for v in variants:
        card_cls = "variant-card-recommended" if v["recommended"] else "variant-card"
        badge    = " &nbsp;<span style='background:#F59E0B;color:#000;padding:2px 8px;border-radius:4px;font-size:0.75rem;font-weight:700'>AI RECOMMENDED</span>" if v["recommended"] else ""
        st.markdown(f"""
        <div class="{card_cls}">
            <div style="font-size:1.05rem;font-weight:600;color:#F9FAFB">{v['name']}{badge}</div>
            <div style="font-size:0.85rem;color:#9CA3AF;margin:6px 0">{v['description']}</div>
            <div style="display:flex;gap:2rem;margin-top:10px;flex-wrap:wrap">
                <div><div style="font-size:0.75rem;color:#6B7280">Est. Cost</div><div style="font-size:1.1rem;font-weight:600;color:#F59E0B">₹{v['est_cost_cr']} Cr</div></div>
                <div><div style="font-size:0.75rem;color:#6B7280">Duration</div><div style="font-size:1.1rem;font-weight:600;color:#F59E0B">{v['est_duration_days']:.0f} days</div></div>
                <div><div style="font-size:0.75rem;color:#6B7280">Rework Risk</div><div style="font-size:1.1rem;font-weight:600;color:#F59E0B">{v['rework_risk']}</div></div>
                <div><div style="font-size:0.75rem;color:#6B7280">Carbon</div><div style="font-size:1.1rem;font-weight:600;color:#F59E0B">{v['carbon_t']} T CO₂</div></div>
                <div><div style="font-size:0.75rem;color:#6B7280">AI Score</div><div style="font-size:1.1rem;font-weight:600;color:#{'10B981' if v['score']>88 else 'FBBF24'}">{v['score']}/100</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">📊 Variant Comparison Radar</div>', unsafe_allow_html=True)
    cats_r   = ["Cost Efficiency","Speed","Quality","Sustainability","AI Score"]
    max_d    = max(v["est_duration_days"] for v in variants)
    max_c    = max(v["est_cost_cr"] for v in variants)
    v_colors = ["#3B82F6","#F59E0B","#10B981"]
    fig_cmp  = go.Figure()
    for i, v in enumerate(variants):
        sc = [
            round(10-(v["est_cost_cr"]/max_c)*10,1),
            round(10-(v["est_duration_days"]/max_d)*10,1),
            round(10-float(v["rework_risk"].split("(")[1].replace("%)","")),1),
            round(10-(v["carbon_t"]/(site_area*0.2))*10,1),
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
    st.markdown('<div class="section-header">📡 Real-Time IoT Sensor Feed — Construction Site</div>', unsafe_allow_html=True)

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
        <div class="metric-card" style="border-color:{color}33">
            <div class="iot-label">{label}</div>
            <div class="iot-value" style="color:{color}">{value}</div>
            <div class="iot-label">{unit}</div>
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
    else:
        st.success("✅ All sensors nominal.")

    if len(history) > 2:
        st.markdown('<div class="section-header">📈 Sensor History</div>', unsafe_allow_html=True)
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
    "DynaConstructa.AI · TeamTurbo · BITS Pilani Goa · CreaTech 2026 · L&T Problem Statement 3"
    "</div>",
    unsafe_allow_html=True
)