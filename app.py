"""
AI Car Price Predictor — Streamlit App
---------------------------------------
Frontend/UI wraps the EXACT model pipeline trained in the notebook:
  Features (order matters): Brand, Body, Mileage, EngineV, Engine Type, Registration, Year
  Target:  Log_price = np.log(Price)  ->  Price = np.exp(prediction)
  Model:   sklearn LinearRegression, saved as linear_regression_model.pkl (joblib)

Categorical encoding reconstructs the same LabelEncoder mapping used in training
(alphabetical order per column), verified against the notebook's printed output.
"""

import numpy as np
import joblib
import streamlit as st

# ----------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Prediction-pipeline constants (do NOT touch — mirrors the notebook exactly)
# ----------------------------------------------------------------------------
MODEL_PATH = "linear_regression_model.pkl"

# LabelEncoder assigns integers in alphabetical order of the categories it saw.
BRAND_MAP = {
    "Audi": 0, "BMW": 1, "Mercedes-Benz": 2, "Mitsubishi": 3,
    "Renault": 4, "Toyota": 5, "Volkswagen": 6,
}
BODY_MAP = {
    "crossover": 0, "hatch": 1, "other": 2, "sedan": 3, "vagon": 4, "van": 5,
}
ENGINE_TYPE_MAP = {"Diesel": 0, "Gas": 1, "Other": 2, "Petrol": 3}
REGISTRATION_MAP = {"no": 0, "yes": 1}

FEATURE_ORDER = ["Brand", "Body", "Mileage", "EngineV", "Engine Type", "Registration", "Year"]


@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_PATH)


def predict_price(model, brand, body, mileage, enginev, engine_type, registration, year):
    """Runs the exact same encoding + prediction + inverse-log-transform as the notebook."""
    row = [
        BRAND_MAP[brand],
        BODY_MAP[body],
        mileage,
        enginev,
        ENGINE_TYPE_MAP[engine_type],
        REGISTRATION_MAP[registration],
        year,
    ]
    log_price_pred = model.predict([row])[0]
    price_pred = np.exp(log_price_pred)  # inverse of np.log used in training
    return price_pred


# ----------------------------------------------------------------------------
# Styling — dark, glassmorphism, premium SaaS look
# ----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

    .stApp {
        background: radial-gradient(circle at 15% 0%, #1a2035 0%, #0d1117 45%, #090b10 100%);
        color: #e6e9ef;
    }

    #MainMenu, footer, header { visibility: hidden; }

    .hero {
        text-align: center;
        padding: 2.6rem 1rem 2rem 1rem;
    }
    .hero-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 999px;
        background: rgba(99, 179, 237, 0.12);
        border: 1px solid rgba(99, 179, 237, 0.35);
        color: #7fd6ff;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.3px;
        margin-bottom: 1rem;
    }
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ffffff 0%, #9fd3ff 60%, #6c9eff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .hero-subtitle {
        color: #a3adc2;
        font-size: 1.05rem;
        margin-top: 0.6rem;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.045);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 1.8rem 1.9rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
        backdrop-filter: blur(6px);
        margin-bottom: 1.4rem;
    }
    .section-label {
        font-size: 0.95rem;
        font-weight: 700;
        color: #9fd3ff;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 0.6rem;
        margin-top: 0.4rem;
    }
    .card-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #f2f4f8;
        margin-bottom: 1.1rem;
    }

    div[data-testid="stButton"] > button {
        width: 100%;
        background: linear-gradient(90deg, #3b82f6 0%, #6c5ce7 100%);
        color: white;
        font-weight: 700;
        font-size: 1.05rem;
        padding: 0.75rem 0;
        border-radius: 12px;
        border: none;
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.35);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div[data-testid="stButton"] > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 26px rgba(108, 92, 231, 0.45);
        color: white;
        border: none;
    }

    .result-card {
        text-align: center;
        padding: 2rem 1.5rem;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(59,130,246,0.15) 0%, rgba(108,92,231,0.12) 100%);
        border: 1px solid rgba(124, 168, 255, 0.35);
        box-shadow: 0 10px 34px rgba(59, 130, 246, 0.18);
    }
    .result-label {
        color: #a9c7ff;
        font-weight: 600;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }
    .result-value {
        font-size: 2.8rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0.3rem 0;
    }
    .result-note {
        color: #93a1b8;
        font-size: 0.85rem;
    }

    .footer-text {
        text-align: center;
        color: #6b7385;
        font-size: 0.85rem;
        padding: 1.6rem 0 0.6rem 0;
    }

    section[data-testid="stSidebar"] {
        background: #0d1117;
        border-right: 1px solid rgba(255,255,255,0.06);
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧠 About the Model")
    st.markdown(
        "- Machine Learning based car price prediction\n"
        "- Built with **Python**\n"
        "- Built with **Streamlit**\n"
        "- Trained using Linear, Ridge & Lasso Regression — **Linear Regression** selected as the best performer"
    )
    st.markdown("---")
    st.markdown("### 🛠️ Project Technologies")
    st.markdown("- Python\n- Pandas\n- NumPy\n- Scikit-learn\n- Streamlit")
    st.markdown("---")
    st.caption("This tool provides an estimate only, based on patterns learned from historical listings.")

# ----------------------------------------------------------------------------
# Hero section
# ----------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <div class="hero-badge">⚡ Powered by Machine Learning</div>
    <div class="hero-title">AI Car Price Predictor</div>
    <div class="hero-subtitle">Predict your car's estimated market value using Machine Learning.</div>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Input form
# ----------------------------------------------------------------------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="card-title">🚗 Enter Car Details</div>', unsafe_allow_html=True)

st.markdown('<div class="section-label">Vehicle Information</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    brand = st.selectbox("Brand", list(BRAND_MAP.keys()), help="Manufacturer of the vehicle")
with c2:
    body = st.selectbox("Body Type", list(BODY_MAP.keys()), help="Body style of the car")
with c3:
    year = st.number_input("Year", min_value=1960, max_value=2026, value=2012, step=1,
                            help="Manufacturing year")

st.markdown('<div class="section-label">Engine & Registration</div>', unsafe_allow_html=True)
c4, c5, c6 = st.columns(3)
with c4:
    engine_type = st.selectbox("Engine Type", list(ENGINE_TYPE_MAP.keys()), help="Fuel/engine type")
with c5:
    enginev = st.number_input("Engine Volume (L)", min_value=0.0, max_value=10.0, value=2.0,
                               step=0.1, help="Engine displacement in liters (0–10)")
with c6:
    registration = st.selectbox("Registered", list(REGISTRATION_MAP.keys()),
                                 help="Is the car currently registered?")

st.markdown('<div class="section-label">Usage</div>', unsafe_allow_html=True)
mileage = st.number_input("Mileage (in thousands of km/miles)", min_value=0, max_value=1000,
                           value=150, step=1, help="Total distance the car has traveled")

st.markdown("<br>", unsafe_allow_html=True)
predict_clicked = st.button("🔮 Predict Car Price")
st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Prediction & result
# ----------------------------------------------------------------------------
if predict_clicked:
    try:
        model = load_model()
        with st.spinner("Running the model..."):
            predicted_price = predict_price(
                model, brand, body, mileage, enginev, engine_type, registration, year
            )

        st.success("Prediction complete!")
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Estimated Market Value</div>
            <div class="result-value">${predicted_price:,.2f}</div>
            <div class="result-note">Estimated value based on the information provided.</div>
        </div>
        """, unsafe_allow_html=True)

    except FileNotFoundError:
        st.error(
            f"Model file `{MODEL_PATH}` was not found. Place the `linear_regression_model.pkl` "
            "file (produced by the notebook's `joblib.dump`) in the same folder as this app."
        )

# ----------------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------------
st.markdown('<div class="footer-text">Built with Python & Streamlit | Machine Learning Project</div>',
            unsafe_allow_html=True)