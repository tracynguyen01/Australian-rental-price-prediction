from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Australian Rental Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "knn_rental_model.pkl"
SCALER_PATH = MODEL_DIR / "rental_scaler.pkl"
FEATURE_COLUMNS_PATH = MODEL_DIR / "rental_feature_columns.pkl"


# ============================================================
# LOAD MODEL ARTIFACTS
# ============================================================

@st.cache_resource
def load_model_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

    return model, scaler, feature_columns


try:
    model, scaler, feature_columns = load_model_artifacts()

except FileNotFoundError:
    st.error(
        "Model files could not be found. "
        "Please make sure the three .pkl files are inside the models folder."
    )
    st.stop()

except Exception as e:
    st.error(f"Unable to load model files: {e}")
    st.stop()


# ============================================================
# PREDICTION FUNCTIONS
# ============================================================

def categorize_level(floor_level):
    """
    Convert actual floor number into the same level categories
    used during model training.
    """

    if floor_level <= 10:
        return "Low"

    elif floor_level <= 20:
        return "Medium"

    return "High"


def preprocess_user_input(
    user_input,
    feature_columns,
    scaler
):
    """
    Apply the same transformations used during model training.
    """

    input_processed = user_input.copy()

    # --------------------------------------------------------
    # Log-transform floor area
    # --------------------------------------------------------

    input_processed["log_floor_area"] = np.log(
        input_processed["floor_area"]
    )

    input_processed = input_processed.drop(
        columns=["floor_area"]
    )

    # --------------------------------------------------------
    # Save categorical values
    # --------------------------------------------------------

    suburb = input_processed.loc[0, "suburb"]
    furnished = input_processed.loc[0, "furnished"]
    tenancy = input_processed.loc[0, "tenancy_preference"]
    contact = input_processed.loc[0, "point_of_contact"]
    level = input_processed.loc[0, "level"]

    # --------------------------------------------------------
    # Remove original categorical columns
    # --------------------------------------------------------

    input_processed = input_processed.drop(
        columns=[
            "suburb",
            "furnished",
            "tenancy_preference",
            "point_of_contact",
            "level"
        ]
    )

    # --------------------------------------------------------
    # Create dataframe with exactly the same columns
    # used during training
    # --------------------------------------------------------

    final_input = pd.DataFrame(
        0.0,
        index=[0],
        columns=feature_columns
    )

    # Copy numeric values
    for col in input_processed.columns:
        if col in final_input.columns:
            final_input.loc[0, col] = input_processed.loc[0, col]

    # --------------------------------------------------------
    # Activate correct dummy variables
    #
    # If a category was the baseline category during
    # drop_first=True encoding, its column will not exist.
    # In that case it correctly remains all zeros.
    # --------------------------------------------------------

    dummy_values = {
        f"suburb_{suburb}": 1,
        f"furnished_{furnished}": 1,
        f"tenancy_preference_{tenancy}": 1,
        f"point_of_contact_{contact}": 1,
        f"level_{level}": 1
    }

    for col, value in dummy_values.items():
        if col in final_input.columns:
            final_input.loc[0, col] = value

    # --------------------------------------------------------
    # Apply original scaler
    # --------------------------------------------------------

    final_scaled = scaler.transform(final_input)

    return final_scaled


def predict_rent(
    number_of_bedrooms,
    floor_area,
    floor_level,
    suburb,
    furnished,
    tenancy_preference,
    number_of_bathrooms,
    point_of_contact,
    advertised_month
):
    """
    Build model input and return predicted weekly rent.
    """

    level = categorize_level(floor_level)

    user_input = pd.DataFrame({
        "number_of_bedrooms": [number_of_bedrooms],
        "floor_area": [floor_area],
        "level": [level],
        "suburb": [suburb],
        "furnished": [furnished],
        "tenancy_preference": [tenancy_preference],
        "number_of_bathrooms": [number_of_bathrooms],
        "point_of_contact": [point_of_contact],
        "advertised_month": [advertised_month]
    })

    input_scaled = preprocess_user_input(
        user_input,
        feature_columns,
        scaler
    )

    prediction = model.predict(input_scaled)[0]

    return float(prediction), level


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    /* -------------------------------------------------------
       Main page
    ------------------------------------------------------- */

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    h1 {
        letter-spacing: -0.03em;
    }

    h2, h3 {
        letter-spacing: -0.02em;
    }

    /* -------------------------------------------------------
       Hero subtitle
    ------------------------------------------------------- */

    .hero-subtitle {
        font-size: 18px;
        color: #64748b;
        max-width: 900px;
        line-height: 1.65;
        margin-top: -5px;
        margin-bottom: 12px;
    }

    /* -------------------------------------------------------
       Predictor section
    ------------------------------------------------------- */

    .predictor-header {
        padding: 4px 0 12px 0;
    }

    .predictor-description {
        color: #64748b;
        font-size: 16px;
        margin-bottom: 18px;
    }

    /* -------------------------------------------------------
       Result card
    ------------------------------------------------------- */

    .prediction-card {
        background:
            linear-gradient(
                135deg,
                #eff6ff 0%,
                #ffffff 52%,
                #f5f3ff 100%
            );
        border: 1px solid #bfdbfe;
        border-radius: 24px;
        padding: 32px 28px;
        text-align: center;
        margin-top: 22px;
        box-shadow: 0 8px 28px rgba(37,99,235,0.10);
    }

    .prediction-label {
        font-size: 13px;
        font-weight: 800;
        letter-spacing: 0.12em;
        color: #64748b;
    }

    .prediction-value {
        font-size: 56px;
        line-height: 1.1;
        font-weight: 850;
        color: #2563eb;
        margin-top: 12px;
    }

    .prediction-period {
        font-size: 17px;
        color: #64748b;
        margin-top: 5px;
    }

    /* -------------------------------------------------------
       Result supporting metrics
    ------------------------------------------------------- */

    .summary-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 18px;
        height: 100%;
        box-shadow: 0 3px 12px rgba(15,23,42,0.04);
    }

    .summary-title {
        font-size: 12px;
        font-weight: 800;
        color: #64748b;
        letter-spacing: 0.08em;
    }

    .summary-value {
        font-size: 25px;
        font-weight: 800;
        color: #0f172a;
        margin-top: 5px;
    }

    /* -------------------------------------------------------
       Insight cards
    ------------------------------------------------------- */

    .insight-card {
        background: linear-gradient(135deg, #eff6ff, #ffffff);
        border: 1px solid #bfdbfe;
        border-left: 6px solid #2563eb;
        border-radius: 18px;
        padding: 18px;
        margin-bottom: 14px;
        box-shadow: 0 4px 14px rgba(37,99,235,0.07);
    }

    .insight-title {
        font-weight: 750;
        font-size: 16px;
        color: #1e3a8a;
    }

    .insight-text {
        margin-top: 6px;
        color: #475569;
        line-height: 1.55;
    }

    /* -------------------------------------------------------
       Pipeline
    ------------------------------------------------------- */

    .pipeline-box {
        display: inline-block;
        padding: 13px 17px;
        border-radius: 14px;
        font-weight: 700;
        margin: 6px 3px;
    }

    .arrow {
        display: inline-block;
        font-size: 24px;
        color: #94a3b8;
        margin: 0 3px;
    }

    /* -------------------------------------------------------
       Tabs
    ------------------------------------------------------- */

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        padding-left: 20px;
        padding-right: 20px;
    }

    /* -------------------------------------------------------
       Button
    ------------------------------------------------------- */

    div.stButton > button {
        border-radius: 12px;
        min-height: 48px;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.title("🏠 Australian Rental Price Prediction")

st.markdown(
    """
    <div class="hero-subtitle">
        Explore Australian rental market patterns and estimate weekly
        rental prices using a machine learning model trained on property,
        location, furnishing, tenancy and seasonal characteristics.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# KPI CARDS
# ============================================================

components.html(
    """
    <div style="
        display:grid;
        grid-template-columns:repeat(4,1fr);
        gap:18px;
        margin:12px 0 6px 0;
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
                font-size:12px;
                font-weight:750;
                color:#64748b;
                letter-spacing:0.08em;
            ">
                🏆 BEST MODEL
            </div>

            <div style="
                font-size:32px;
                font-weight:850;
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
                font-size:12px;
                font-weight:750;
                color:#64748b;
                letter-spacing:0.08em;
            ">
                📉 TEST RMSE
            </div>

            <div style="
                font-size:32px;
                font-weight:850;
                color:#0f172a;
                margin-top:10px;
            ">
                $48.99
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
                font-size:12px;
                font-weight:750;
                color:#64748b;
                letter-spacing:0.08em;
            ">
                🧠 MODEL FEATURES
            </div>

            <div style="
                font-size:32px;
                font-weight:850;
                color:#0f172a;
                margin-top:10px;
            ">
                9
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
                font-size:12px;
                font-weight:750;
                color:#64748b;
                letter-spacing:0.08em;
            ">
                📍 LOCATIONS
            </div>

            <div style="
                font-size:32px;
                font-weight:850;
                color:#0f172a;
                margin-top:10px;
            ">
                6
            </div>
        </div>

    </div>
    """,
    height=145
)


st.divider()


# ============================================================
# TABS
# ============================================================

tab_predictor, tab_insights, tab_model, tab_business = st.tabs(
    [
        "🔮 Rent Predictor",
        "📊 Market Insights",
        "🧠 Model Performance",
        "💼 Business Value"
    ]
)


# ============================================================
# TAB 1 — RENT PREDICTOR
# ============================================================

with tab_predictor:

    st.markdown("## 🔮 Estimate Weekly Rental Price")

    st.markdown(
        """
        <div class="predictor-description">
            Enter the property characteristics below.
            The trained KNN regression model will estimate the expected
            weekly rental price.
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.form("rental_prediction_form"):

        st.markdown("### Property Details")

        col1, col2, col3 = st.columns(3)

        # ----------------------------------------------------
        # Column 1
        # ----------------------------------------------------

        with col1:

            suburb = st.selectbox(
                "📍 Location",
                [
                    "Sydney",
                    "Melbourne",
                    "Brisbane",
                    "Perth",
                    "Adelaide",
                    "Canberra"
                ],
                help="Location categories available in the training dataset."
            )

            number_of_bedrooms = st.number_input(
                "🛏️ Bedrooms",
                min_value=1,
                max_value=6,
                value=2,
                step=1
            )

            number_of_bathrooms = st.number_input(
                "🛁 Bathrooms",
                min_value=1,
                max_value=7,
                value=2,
                step=1
            )

        # ----------------------------------------------------
        # Column 2
        # ----------------------------------------------------

        with col2:

            floor_area = st.number_input(
                "📐 Floor Area",
                min_value=20,
                max_value=8000,
                value=800,
                step=10,
                help=(
                    "Use the same floor-area unit represented "
                    "in the original training dataset."
                )
            )

            floor_level = st.number_input(
                "🏢 Floor Level",
                min_value=0,
                max_value=50,
                value=1,
                step=1,
                help=(
                    "Ground or basement can be entered as 0. "
                    "The model internally groups levels into "
                    "Low, Medium and High."
                )
            )

            furnished = st.selectbox(
                "🛋️ Furnishing Status",
                [
                    "Unfurnished",
                    "Semi-Furnished",
                    "Furnished"
                ]
            )

        # ----------------------------------------------------
        # Column 3
        # ----------------------------------------------------

        with col3:

            tenancy_preference = st.selectbox(
                "👥 Tenancy Preference",
                [
                    "Bachelors/Family",
                    "Bachelors",
                    "Family"
                ]
            )

            point_of_contact = st.selectbox(
                "☎️ Point of Contact",
                [
                    "Contact Owner",
                    "Contact Agent"
                ]
            )

            month_name = st.selectbox(
                "📅 Advertised Month",
                [
                    "April",
                    "May",
                    "June"
                ],
                index=2,
                help=(
                    "The training set used for the deployed model "
                    "contains advertisements from April to June."
                )
            )

        month_mapping = {
            "April": 4,
            "May": 5,
            "June": 6
        }

        advertised_month = month_mapping[month_name]

        st.markdown("")

        submitted = st.form_submit_button(
            "✨ Predict Weekly Rent",
            use_container_width=True,
            type="primary"
        )


    # --------------------------------------------------------
    # PREDICTION RESULT
    # --------------------------------------------------------

    if submitted:

        try:

            prediction, level_group = predict_rent(
                number_of_bedrooms=number_of_bedrooms,
                floor_area=floor_area,
                floor_level=floor_level,
                suburb=suburb,
                furnished=furnished,
                tenancy_preference=tenancy_preference,
                number_of_bathrooms=number_of_bathrooms,
                point_of_contact=point_of_contact,
                advertised_month=advertised_month
            )

            monthly_estimate = prediction * 52 / 12
            annual_estimate = prediction * 52

            st.markdown(
                f"""
                <div class="prediction-card">

                    <div class="prediction-label">
                        ESTIMATED WEEKLY RENT
                    </div>

                    <div class="prediction-value">
                        ${prediction:,.0f}
                    </div>

                    <div class="prediction-period">
                        per week
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("")

            result1, result2, result3 = st.columns(3)

            with result1:

                st.markdown(
                    f"""
                    <div class="summary-card">
                        <div class="summary-title">
                            MONTHLY ESTIMATE
                        </div>

                        <div class="summary-value">
                            ${monthly_estimate:,.0f}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with result2:

                st.markdown(
                    f"""
                    <div class="summary-card">
                        <div class="summary-title">
                            ANNUAL ESTIMATE
                        </div>

                        <div class="summary-value">
                            ${annual_estimate:,.0f}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with result3:

                st.markdown(
                    f"""
                    <div class="summary-card">
                        <div class="summary-title">
                            FLOOR CATEGORY
                        </div>

                        <div class="summary-value">
                            {level_group}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            st.markdown("### 🏠 Property Summary")

            summary_col1, summary_col2 = st.columns(2)

            with summary_col1:

                st.markdown(
                    f"""
                    **Location:** {suburb}  
                    **Bedrooms:** {number_of_bedrooms}  
                    **Bathrooms:** {number_of_bathrooms}  
                    **Floor Area:** {floor_area:,}
                    """
                )

            with summary_col2:

                st.markdown(
                    f"""
                    **Floor Level:** {floor_level} ({level_group})  
                    **Furnishing:** {furnished}  
                    **Tenancy Preference:** {tenancy_preference}  
                    **Point of Contact:** {point_of_contact}  
                    **Advertised Month:** {month_name}
                    """
                )

            st.info(
                "This estimate is produced by the trained KNN regression "
                "model and should be interpreted as a model-based rental "
                "estimate rather than a formal property valuation."
            )

        except Exception as e:

            st.error(
                f"Prediction could not be generated: {e}"
            )


# ============================================================
# TAB 2 — MARKET INSIGHTS
# ============================================================

with tab_insights:

    st.markdown("## 📊 Rental Market Insights")

    st.markdown(
        """
        Exploratory analysis identified several property characteristics
        associated with rental prices.
        """
    )

    # --------------------------------------------------------
    # EDA correlation chart
    # --------------------------------------------------------

    correlation_df = pd.DataFrame({
        "Feature": [
            "Bathrooms",
            "Floor Area",
            "Bedrooms"
        ],
        "Correlation with Rent": [
            0.39,
            0.38,
            0.32
        ]
    })

    fig_corr = px.bar(
        correlation_df,
        x="Correlation with Rent",
        y="Feature",
        orientation="h",
        text=correlation_df[
            "Correlation with Rent"
        ].map(lambda x: f"{x:.2f}"),
        color="Feature",
        color_discrete_map={
            "Bathrooms": "#2563eb",
            "Floor Area": "#7c3aed",
            "Bedrooms": "#14b8a6"
        }
    )

    fig_corr.update_layout(
        height=380,
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis_title="Correlation with Weekly Rent",
        yaxis_title="",
        font=dict(
            size=14,
            color="#334155"
        ),
        margin=dict(
            l=20,
            r=80,
            t=20,
            b=20
        )
    )

    fig_corr.update_xaxes(
        range=[0, 0.45],
        gridcolor="#e2e8f0"
    )

    fig_corr.update_traces(
        textposition="outside",
        cliponaxis=False
    )

    st.plotly_chart(
        fig_corr,
        use_container_width=True
    )


    insight_left, insight_right = st.columns(2)

    with insight_left:

        st.markdown(
            """
            <div class="insight-card">
                <div class="insight-title">
                    🛁 Bathrooms
                </div>
                <div class="insight-text">
                    Bathrooms showed the strongest positive numerical
                    correlation with rent among the analysed numeric
                    property characteristics.
                </div>
            </div>

            <div class="insight-card"
                style="
                    background:linear-gradient(135deg,#faf5ff,#ffffff);
                    border-color:#ddd6fe;
                    border-left-color:#7c3aed;
                "
            >
                <div class="insight-title"
                    style="color:#5b21b6;">
                    📐 Floor Area
                </div>
                <div class="insight-text">
                    Larger floor areas were generally associated with
                    higher rental prices. A log transformation was applied
                    during model preparation to reduce skewness.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with insight_right:

        st.markdown(
            """
            <div class="insight-card"
                style="
                    background:linear-gradient(135deg,#ecfeff,#ffffff);
                    border-color:#a5f3fc;
                    border-left-color:#0891b2;
                "
            >
                <div class="insight-title"
                    style="color:#155e75;">
                    🛏️ Bedrooms
                </div>
                <div class="insight-text">
                    The number of bedrooms showed a positive relationship
                    with rental prices, reflecting the higher value of
                    larger properties.
                </div>
            </div>

            <div class="insight-card"
                style="
                    background:linear-gradient(135deg,#f0fdf4,#ffffff);
                    border-color:#bbf7d0;
                    border-left-color:#16a34a;
                "
            >
                <div class="insight-title"
                    style="color:#166534;">
                    📍 Location
                </div>
                <div class="insight-text">
                    Rental price distributions varied across the six
                    locations represented in the dataset, making location
                    an important categorical input to the prediction model.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.caption(
        "Correlation values shown above come from exploratory analysis "
        "and should not be interpreted as KNN feature importance."
    )


# ============================================================
# TAB 3 — MODEL PERFORMANCE
# ============================================================

with tab_model:

    st.markdown("## 🧠 Model Performance")

    left, right = st.columns([1.08, 0.92])

    # --------------------------------------------------------
    # Model comparison
    # --------------------------------------------------------

    with left:

        st.markdown("### 📉 Model Comparison")

        sort_order = st.radio(
            "Sort RMSE",
            [
                "Lowest to Highest",
                "Highest to Lowest"
            ],
            horizontal=True,
            key="model_sort"
        )

        model_df = pd.DataFrame({
            "Model": [
                "Linear Regression",
                "ElasticNet",
                "KNN Regression"
            ],
            "RMSE": [
                54.66,
                54.65,
                48.99
            ]
        })

        model_df = model_df.sort_values(
            "RMSE",
            ascending=(
                sort_order == "Lowest to Highest"
            )
        )

        fig_model = px.bar(
            model_df,
            x="RMSE",
            y="Model",
            orientation="h",
            text=model_df["RMSE"].map(
                lambda x: f"{x:.2f}"
            ),
            color="Model",
            color_discrete_map={
                "KNN Regression": "#2563eb",
                "ElasticNet": "#7c3aed",
                "Linear Regression": "#14b8a6"
            }
        )

        fig_model.update_layout(
            height=410,
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=False,
            xaxis_title="Test RMSE",
            yaxis_title="",
            font=dict(
                size=14,
                color="#334155"
            ),
            margin=dict(
                l=20,
                r=100,
                t=20,
                b=20
            )
        )

        fig_model.update_xaxes(
            range=[0, 60],
            gridcolor="#e2e8f0"
        )

        fig_model.update_traces(
            textposition="outside",
            cliponaxis=False
        )

        st.plotly_chart(
            fig_model,
            use_container_width=True
        )

        st.success(
            "KNN Regression achieved the lowest test RMSE "
            "(48.99), outperforming Linear Regression and "
            "ElasticNet in the model comparison."
        )


    # --------------------------------------------------------
    # Pipeline
    # --------------------------------------------------------

    with right:

        st.markdown("### ⚙️ Machine Learning Pipeline")

        pipeline_html = """
        <div style="line-height:3.2;">

            <span class="pipeline-box"
                style="background:#dbeafe;color:#1e40af;">
                Clean
            </span>

            <span class="arrow">→</span>

            <span class="pipeline-box"
                style="background:#ede9fe;color:#5b21b6;">
                Engineer
            </span>

            <span class="arrow">→</span>

            <span class="pipeline-box"
                style="background:#ccfbf1;color:#0f766e;">
                Encode
            </span>

            <span class="arrow">→</span>

            <span class="pipeline-box"
                style="background:#fef3c7;color:#92400e;">
                Transform
            </span>

            <span class="arrow">→</span>

            <span class="pipeline-box"
                style="background:#fce7f3;color:#9d174d;">
                Scale
            </span>

            <span class="arrow">→</span>

            <span class="pipeline-box"
                style="background:#dcfce7;color:#166534;">
                KNN
            </span>

            <span class="arrow">→</span>

            <span class="pipeline-box"
                style="background:#fee2e2;color:#991b1b;">
                Predict
            </span>

        </div>
        """

        st.markdown(
            pipeline_html,
            unsafe_allow_html=True
        )

        st.markdown("### Final KNN Configuration")

        metric1, metric2 = st.columns(2)

        with metric1:
            st.metric(
                "Neighbors",
                "40"
            )

        with metric2:
            st.metric(
                "Distance p",
                "1"
            )

        st.markdown(
            """
            **Model inputs include:**

            Property location, bedrooms, bathrooms, floor area,
            floor level, furnishing status, tenancy preference,
            point of contact and advertised month.
            """
        )

        st.info(
            "The deployment model uses the selected KNN "
            "configuration and is refitted on the cleaned "
            "training dataset for interactive prediction."
        )


# ============================================================
# TAB 4 — BUSINESS VALUE
# ============================================================

with tab_business:

    st.markdown("## 💼 Business Value")

    colA, colB = st.columns(2)

    with colA:

        st.markdown(
            """
            ### 🎯 Value for Landlords

            - Estimate an appropriate weekly rental price from
              property characteristics.
            - Reduce the risk of significant underpricing or
              overpricing.
            - Compare how property characteristics may influence
              model estimates.
            - Support more consistent, data-informed rental
              pricing decisions.
            """
        )

    with colB:

        st.markdown(
            """
            ### 🏢 Value for Property Agencies

            - Support faster initial rental price estimation.
            - Provide a consistent analytical starting point for
              property assessment.
            - Apply the same model across the locations represented
              in the training data.
            - Combine model estimates with professional market
              knowledge for pricing decisions.
            """
        )

    st.divider()

    st.markdown("### 🔎 How to Interpret the Prediction")

    st.markdown(
        """
        The prediction is an analytical estimate generated from patterns
        learned from the historical training dataset. It is best used as
        decision support rather than as a substitute for a professional
        valuation or current real-estate market assessment.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Built with Streamlit | Australian Rental Price Prediction | "
    "KNN Regression"
)
