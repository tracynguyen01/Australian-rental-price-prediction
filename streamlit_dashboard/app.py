import streamlit as st
import pandas as pd
import plotly.express as px
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Australian Rental Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# =========================
# CSS
# =========================
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.pipeline-box {
    display: inline-block;
    padding: 14px 18px;
    border-radius: 16px;
    font-weight: 700;
    margin: 6px 4px;
}

.arrow {
    display: inline-block;
    font-size: 26px;
    color: #64748b;
    margin: 0 4px;
}

.insight-card {
    background: linear-gradient(135deg, #eff6ff, #ffffff);
    border: 1px solid #bfdbfe;
    border-left: 6px solid #2563eb;
    border-radius: 18px;
    padding: 18px;
    margin-bottom: 14px;
    box-shadow: 0 4px 14px rgba(37,99,235,0.08);
}

.insight-title {
    font-weight: 700;
    font-size: 17px;
    color: #1e3a8a;
}

.insight-text {
    margin-top: 6px;
    color: #475569;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.title("🏠 Australian Rental Price Prediction")

st.markdown(
    "Machine learning dashboard for predicting Australian rental prices using property features, location, furnishing status and seasonal signals."
)

st.divider()

# =========================
# KPI
# =========================
components.html(
    """
    <div style="
        display:grid;
        grid-template-columns:repeat(4,1fr);
        gap:18px;
        margin:5px 0 10px 0;
        font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
    ">

        <div style="
            background:white;
            border:1px solid #e2e8f0;
            border-radius:20px;
            padding:18px 16px;
            text-align:center;
            box-shadow:0 4px 14px rgba(15,23,42,0.05);
        ">
            <div style="
                font-size:13px;
                font-weight:700;
                color:#64748b;
                letter-spacing:0.08em;
                display:flex;
                align-items:center;
                justify-content:center;
                gap:6px;
            ">
                🏆 BEST MODEL
            </div>

            <div style="
                font-size:34px;
                font-weight:800;
                color:#0f172a;
                margin-top:10px;
            ">
                KNN
            </div>
        </div>

        <div style="
            background:white;
            border:1px solid #e2e8f0;
            border-radius:20px;
            padding:18px 16px;
            text-align:center;
            box-shadow:0 4px 14px rgba(15,23,42,0.05);
        ">
            <div style="
                font-size:13px;
                font-weight:700;
                color:#64748b;
                letter-spacing:0.08em;
                display:flex;
                align-items:center;
                justify-content:center;
                gap:6px;
            ">
                📉 TEST RMSE
            </div>

            <div style="
                font-size:34px;
                font-weight:800;
                color:#0f172a;
                margin-top:10px;
            ">
                48.99
            </div>
        </div>

        <div style="
            background:white;
            border:1px solid #e2e8f0;
            border-radius:20px;
            padding:18px 16px;
            text-align:center;
            box-shadow:0 4px 14px rgba(15,23,42,0.05);
        ">
            <div style="
                font-size:13px;
                font-weight:700;
                color:#64748b;
                letter-spacing:0.08em;
                display:flex;
                align-items:center;
                justify-content:center;
                gap:6px;
            ">
                🎯 TARGET
            </div>

            <div style="
                font-size:34px;
                font-weight:800;
                color:#0f172a;
                margin-top:10px;
            ">
                Rent
            </div>
        </div>

        <div style="
            background:white;
            border:1px solid #e2e8f0;
            border-radius:20px;
            padding:18px 16px;
            text-align:center;
            box-shadow:0 4px 14px rgba(15,23,42,0.05);
        ">
            <div style="
                font-size:13px;
                font-weight:700;
                color:#64748b;
                letter-spacing:0.08em;
                display:flex;
                align-items:center;
                justify-content:center;
                gap:6px;
            ">
                💵 PRICE RANGE
            </div>

            <div style="
                font-size:30px;
                font-weight:800;
                color:#0f172a;
                margin-top:10px;
            ">
                $400–$800
            </div>
        </div>

    </div>
    """,
    height=155
)

st.divider()

# =========================
# MAIN
# =========================
left, space, right = st.columns([1.17, 0.06, 0.9])

with left:
    st.subheader("📊 Model Comparison")

    sort_order = st.radio(
        "Sort RMSE",
        ["Lowest to Highest", "Highest to Lowest"],
        horizontal=True
    )

    model_df = pd.DataFrame({
        "Model": ["Linear Regression", "ElasticNet", "KNN Regression"],
        "RMSE": [54.66, 54.65, 48.99]
    })

    model_df = model_df.sort_values(
        "RMSE",
        ascending=(sort_order == "Lowest to Highest")
    )

    fig = px.bar(
        model_df,
        x="RMSE",
        y="Model",
        orientation="h",
        text=model_df["RMSE"].map(lambda x: f"{x:.2f}"),
        color="Model",
        color_discrete_map={
            "KNN Regression": "#2563eb",
            "ElasticNet": "#7c3aed",
            "Linear Regression": "#14b8a6",
        }
    )

    fig.update_layout(
        height=420,
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        xaxis_title="RMSE",
        yaxis_title="",
        font=dict(size=15, color="#334155"),
        margin=dict(l=20, r=120, t=20, b=20)
    )

    fig.update_xaxes(
        range=[0, 60],
        gridcolor="#e2e8f0"
    )

    fig.update_traces(
        textposition="outside",
        opacity=0.95,
        cliponaxis=False
    )

    st.plotly_chart(fig, use_container_width=True)

    st.success(
        "KNN Regression achieved the strongest performance on unseen data, capturing non-linear rental patterns more effectively than linear models."
    )

with right:
    st.subheader("⚙️ ML Pipeline")

    pipeline_html = """
    <div>
        <span class="pipeline-box" style="background:#dbeafe;color:#1e40af;">Clean</span>
        <span class="arrow">→</span>
        <span class="pipeline-box" style="background:#ede9fe;color:#5b21b6;">Engineer</span>
        <span class="arrow">→</span>
        <span class="pipeline-box" style="background:#ccfbf1;color:#0f766e;">Encode</span>
        <span class="arrow">→</span>
        <span class="pipeline-box" style="background:#fef3c7;color:#92400e;">Scale</span>
        <span class="arrow">→</span>
        <span class="pipeline-box" style="background:#dcfce7;color:#166534;">Tune</span>
        <span class="arrow">→</span>
        <span class="pipeline-box" style="background:#fee2e2;color:#991b1b;">Evaluate</span>
    </div>
    """

    st.markdown(pipeline_html, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:60px;'></div>", unsafe_allow_html=True)
    st.markdown("## 📌 Key Insights")

    st.markdown("""
    <div class="insight-card">
        <div class="insight-title">🏡 Floor Area Impact</div>
        <div class="insight-text">Larger floor areas generally increase rental prices.</div>
    </div>

    <div class="insight-card" style="background:linear-gradient(135deg,#faf5ff,#ffffff);border-color:#ddd6fe;border-left-color:#7c3aed;">
        <div class="insight-title" style="color:#5b21b6;">🛏️ Room Correlation</div>
        <div class="insight-text">Bedrooms and bathrooms show positive correlation with rent.</div>
    </div>

    <div class="insight-card" style="background:linear-gradient(135deg,#ecfeff,#ffffff);border-color:#a5f3fc;border-left-color:#0891b2;">
        <div class="insight-title" style="color:#155e75;">📅 Seasonal Trend</div>
        <div class="insight-text">July showed stronger rental demand and pricing activity.</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# =========================
# KEY RENTAL DRIVERS
# =========================
st.subheader("🏙️ Key Rental Drivers")

feature_df = pd.DataFrame({
    "Feature": ["Bathrooms", "Floor Area", "Bedrooms", "Furnishing", "Seasonality"],
    "Importance": [0.39, 0.38, 0.32, 0.24, 0.19]
})

fig2 = px.line(
    feature_df,
    x="Feature",
    y="Importance",
    markers=True,
    text="Importance"
)

fig2.update_traces(
    line=dict(width=4, color="#2563eb"),
    marker=dict(size=12, color="#7c3aed"),
    texttemplate="%{text:.2f}",
    textposition="top center"
)

fig2.update_layout(
    height=420,
    plot_bgcolor="white",
    paper_bgcolor="white",
    yaxis_title="Relative Importance",
    xaxis_title="",
    font=dict(size=15, color="#334155"),
    margin=dict(l=20, r=20, t=20, b=20)
)

fig2.update_yaxes(
    range=[0, 0.45],
    gridcolor="#e2e8f0"
)

st.plotly_chart(fig2, use_container_width=True)

st.divider()

# =========================
# BUSINESS IMPACT
# =========================
st.subheader("💼 Business Impact")

colA, colB = st.columns(2)

with colA:
    st.markdown("""
    ### 🎯 Value for Landlords

    - More accurate rental pricing  
    - Reduce underpricing and overpricing  
    - Better understanding of market demand  
    - Data-driven pricing strategy  
    """)

with colB:
    st.markdown("""
    ### 🏢 Value for Property Agencies

    - Faster rental valuation process  
    - More consistent pricing decisions  
    - Scalable across multiple suburbs  
    - Supports market trend analysis  
    """)

st.divider()

st.caption("Built with Streamlit | Machine Learning Rental Price Prediction Dashboard")
