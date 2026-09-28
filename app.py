import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="wide")

GOLD, BG, PANEL, TEXT = "#C9A96E", "#0B0B0D", "#141417", "#E8E2D5"
CSV_NAME = "1.04. Real-life example.csv"

# ---------- Styling ----------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Mono', monospace; }
h1, h2, h3 { font-family: 'Cormorant Garamond', serif !important; letter-spacing: .02em; }
h1 { font-size: 3rem !important; color: #C9A96E; margin-bottom: 0; }
.tagline { color: #8d877a; font-size: .85rem; letter-spacing: .18em; text-transform: uppercase; margin-bottom: 1.5rem; }
div[data-testid="stMetric"] { background: #141417; border: 1px solid #2a2a2e; border-radius: 6px; padding: 14px 18px; }
div[data-testid="stMetricValue"] { color: #C9A96E; font-family: 'Cormorant Garamond', serif; }
.price-card { background: linear-gradient(135deg,#141417,#1c1a15); border: 1px solid #C9A96E; border-radius: 8px;
  padding: 28px; text-align: center; margin-top: 12px; }
.price-card .label { color: #8d877a; letter-spacing: .2em; font-size: .75rem; text-transform: uppercase; }
.price-card .value { font-family: 'Cormorant Garamond', serif; font-size: 3.4rem; color: #C9A96E; }
button[data-baseweb="tab"] { letter-spacing: .08em; }
</style>
""",
    unsafe_allow_html=True,
)

plt.rcParams.update(
    {
        "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
        "axes.edgecolor": "#3a3a3f", "axes.labelcolor": TEXT, "text.color": TEXT,
        "xtick.color": TEXT, "ytick.color": TEXT, "grid.color": "#2a2a2e",
    }
)


# ---------- Data loading (same source as the notebook) ----------
@st.cache_data(show_spinner="Loading dataset from Kaggle…")
def load_from_kaggle():
    import kagglehub
    from kagglehub import KaggleDatasetAdapter

    return kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS, "smritisingh1997/car-salescsv", CSV_NAME
    )


def get_raw():
    if os.path.exists(CSV_NAME):  # optional: commit the CSV next to app.py
        return pd.read_csv(CSV_NAME)
    try:
        return load_from_kaggle()
    except Exception as e:
        st.warning(f"Could not load the dataset from Kaggle ({type(e).__name__}). Upload the CSV instead.")
        up = st.file_uploader(f"Upload “{CSV_NAME}”", type="csv")
        if up is None:
            st.stop()
        return pd.read_csv(up)


# ---------- Notebook pipeline (Steps 4-13, logic unchanged) ----------
def fit_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    ridge = Ridge()
    ridge.fit(X_train, y_train)
    lasso = Lasso(alpha=0.1)
    lasso.fit(X, y)  # fitted on full X, y exactly as in the notebook
    preds = {
        "Linear Regression": reg.predict(X_test),
        "Ridge Regression": ridge.predict(X_test),
        "Lasso Regression": lasso.predict(X_test),
    }
    rows = []
    for name, p in preds.items():
        rows.append(
            {
                "Model": name,
                "R-squared": r2_score(y_test, p),
                "RMSE": float(np.sqrt(mean_squared_error(y_test, p))),
                "MAE": mean_absolute_error(y_test, p),
            }
        )
    return {"reg": reg, "y_test": y_test, "preds": preds, "table": pd.DataFrame(rows), "columns": list(X.columns)}


@st.cache_resource(show_spinner="Training models…")
def run_pipeline(raw):
    df = raw.dropna(subset=["Price", "EngineV"])  # Step 4
    df_with_outliers = df.copy()
    df = df[df["EngineV"] <= 10].copy()  # Step 5
    df["Log_price"] = np.log(df["Price"])  # Step 6
    clean = df.copy()

    encoders = {}  # Step 8 (one LabelEncoder per column, kept for the Predict tab)
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop(["Log_price"], axis=1)  # Step 10
    y = df["Log_price"]
    return {
        "before": df_with_outliers, "clean": clean, "encoded": df, "encoders": encoders,
        "notebook": fit_models(X, y),
        # Price is the thing we predict, so it can't be an input in a live app:
        "deploy": fit_models(X.drop(columns=["Price"]), y),
    }


# ---------- App ----------
st.markdown("# Car Price Prediction")
st.markdown('<div class="tagline">Linear · Ridge · Lasso regression on used-car sales</div>', unsafe_allow_html=True)

raw = get_raw()
R = run_pipeline(raw)
clean, encoders = R["clean"], R["encoders"]

tab_pred, tab_models, tab_eda = st.tabs(["Predict", "Model comparison", "Data exploration"])

# ----- Predict -----
with tab_pred:
    st.subheader("Estimate a used car's price")
    c1, c2, c3 = st.columns(3)
    brand = c1.selectbox("Brand", sorted(clean["Brand"].unique()))
    models_for_brand = sorted(clean.loc[clean["Brand"] == brand, "Model"].unique())
    model_name = c2.selectbox("Model", models_for_brand)
    body = c3.selectbox("Body", sorted(clean["Body"].unique()))

    c4, c5, c6 = st.columns(3)
    engine_type = c4.selectbox("Engine type", sorted(clean["Engine Type"].unique()))
    registration = c5.selectbox("Registered", sorted(clean["Registration"].unique()))
    year = c6.number_input("Year", int(clean["Year"].min()), int(clean["Year"].max()), int(clean["Year"].median()))

    c7, c8 = st.columns(2)
    mileage = c7.number_input("Mileage (km)", 0, int(clean["Mileage"].max()), int(clean["Mileage"].median()), step=1000)
    engine_v = c8.slider("Engine volume (L)", float(clean["EngineV"].min()), 10.0, float(clean["EngineV"].median()), 0.1)

    if st.button("Predict price", type="primary"):
        row = {
            "Brand": encoders["Brand"].transform([brand])[0],
            "Body": encoders["Body"].transform([body])[0],
            "Mileage": mileage,
            "EngineV": engine_v,
            "Engine Type": encoders["Engine Type"].transform([engine_type])[0],
            "Registration": encoders["Registration"].transform([registration])[0],
            "Year": year,
            "Model": encoders["Model"].transform([model_name])[0],
        }
        d = R["deploy"]
        x = pd.DataFrame([row])[d["columns"]]
        price = float(np.exp(d["reg"].predict(x)[0]))
        st.markdown(
            f'<div class="price-card"><div class="label">Estimated price</div>'
            f'<div class="value">${price:,.0f}</div></div>',
            unsafe_allow_html=True,
        )
        st.caption("Linear Regression on log(price), converted back with exp(). Rough estimate only.")

# ----- Model comparison -----
with tab_models:
    st.subheader("Model comparison")
    st.warning(
        "The notebook's feature set still contains the raw **Price** column (it only drops *Log_price*), "
        "so those scores are inflated by target leakage. The second table drops Price, which is what the "
        "Predict tab uses."
    )
    for key, title in [("notebook", "Notebook features (includes Price)"), ("deploy", "Deployable features (no Price)")]:
        st.markdown(f"**{title}**")
        t = R[key]["table"]
        st.dataframe(t.style.format({"R-squared": "{:.4f}", "RMSE": "{:.4f}", "MAE": "{:.4f}"}), hide_index=True)

    st.markdown("---")
    cc1, cc2 = st.columns(2)
    which = cc1.selectbox("Feature set", ["Deployable features (no Price)", "Notebook features (includes Price)"])
    mname = cc2.selectbox("Model", list(R["notebook"]["preds"].keys()))
    d = R["deploy"] if which.startswith("Deploy") else R["notebook"]
    y_test, y_pred = d["y_test"], d["preds"][mname]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.scatterplot(x=y_test, y=y_pred, ax=ax, color=GOLD, alpha=.6, edgecolor=None)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    ax.set_xlabel("Actual Log Price")
    ax.set_ylabel("Predicted Log Price")
    ax.set_title(f"Actual vs. Predicted Log Price ({mname})")
    ax.grid(True)
    st.pyplot(fig)

# ----- EDA -----
with tab_eda:
    st.subheader("Data exploration")
    m1, m2, m3 = st.columns(3)
    m1.metric("Rows (raw)", f"{len(raw):,}")
    m2.metric("Rows after cleaning", f"{len(clean):,}")
    m3.metric("Columns", raw.shape[1])

    st.markdown("**First 5 records**")
    st.dataframe(raw.head())
    st.markdown("**Missing values (raw)**")
    st.dataframe(raw.isnull().sum().rename("missing").to_frame().T)

    e1, e2 = st.columns(2)
    with e1:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.boxplot(data=R["before"]["EngineV"], ax=ax, color=GOLD)
        ax.set_title("EngineV — before removing outliers")
        st.pyplot(fig)
    with e2:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.boxplot(data=clean["EngineV"], ax=ax, color=GOLD)
        ax.set_title("EngineV — after (≤ 10 L)")
        st.pyplot(fig)

    h1, h2 = st.columns(2)
    with h1:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(clean["Price"], bins=50, color=GOLD)
        ax.set_title("Price")
        st.pyplot(fig)
    with h2:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(clean["Log_price"], bins=50, color=GOLD)
        ax.set_title("Log price")
        st.pyplot(fig)

    st.markdown("**Correlation heatmap (label-encoded)**")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(R["encoded"].corr(), annot=True, fmt=".2f", cmap="YlOrBr", ax=ax)
    st.pyplot(fig)
