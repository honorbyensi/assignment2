# Honor Jespersen Byensi
# Assignment 2
# Importing necessary libraries
import json
import numpy as np
import pandas as pd
import streamlit as st

# Components
from streamlit_option_menu import option_menu
from streamlit_echarts import st_echarts
from streamlit_lottie import st_lottie
from st_aggrid import AgGrid, GridOptionsBuilder
import extra_streamlit_components as stx

# Authentication
import bcrypt
import streamlit_authenticator as stauth

# ML
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Configure page
st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺", layout="wide")

# Simple theme polish
st.markdown("""
<style>
.card {background: var(--secondary-background-color); border-radius: 16px; padding: 18px; border: 1px solid rgba(127,127,127,0.25);}
.stat {font-size: 30px; font-weight: 700; margin-top: 6px;}
.badge {display:inline-block; padding:2px 8px; border-radius: 999px; background: #1f2937; font-size: 12px; border:1px solid rgba(255,255,255,0.08);}
.small {opacity:.8; font-size: 12px;}
.hr {height:1px; background: rgba(127,127,127,0.25); margin: 16px 0;}
</style>
""", unsafe_allow_html=True)

# Dataset path and ML config
CSV_PATH = "saved_models/diabetes.csv" 
FEATURES = [
    "Pregnancies","Glucose","BloodPressure","SkinThickness",
    "Insulin","BMI","DiabetesPedigreeFunction","Age"
]
TARGET = "Outcome"
ZERO_AS_MISSING = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]

# Authentication setup
def _hash(p: str) -> str:
    return bcrypt.hashpw(p.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

# Login credentials
CREDENTIALS = {
    "usernames": {
        "honor": {
            "email": "honor@school.ca",
            "name": "Honor",
            "password": _hash("2002"),
        }
    }
}

authenticator = stauth.Authenticate(CREDENTIALS, "hbai_auth", "hbai_auth_key", 1)


fields = {"Form name": "Login", "Username": "Username", "Password": "Password"}
name, auth_status, username = authenticator.login(location="sidebar", fields=fields)
if auth_status is False:
    st.error("Username/password is incorrect")
    st.stop()
elif auth_status is None:
    st.warning("Please enter your credentials")
    st.stop()

# Helpers
def load_lottie(path: str):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def load_data_and_train(csv_path: str = CSV_PATH):
    df = pd.read_csv(csv_path)

    # Treating zeros as missing in clinical columns
    df = df.copy()
    for c in ZERO_AS_MISSING:
        df[c] = df[c].replace(0, np.nan)

    X = df[FEATURES]
    y = df[TARGET].astype(int)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    pipe.fit(Xtr, ytr)
    acc = pipe.score(Xte, yte)
    return df, pipe, acc

def to_nan_zeros(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    for c in ZERO_AS_MISSING:
        if c in df.columns:
            df[c] = df[c].replace(0, np.nan)
    return df

# Sidebar (logout + NAV + cookie)
with st.sidebar:
    # logout
    authenticator.logout("Logout", "sidebar")

    # Navigation (unique key; single instance)
    selected = st.radio(
        "Navigation",
        ("Home", "Predict", "Explore"),
        index=0,
        key="nav_radio", 
    )

    # Cookie manager (single instance)
    if "cookie_manager" not in st.session_state:
        st.session_state["cookie_manager"] = stx.CookieManager(key="cookies-v1")

    cookie_manager = st.session_state["cookie_manager"]
    if cookie_manager.get("visited") is None:
        cookie_manager.set("visited", "yes")
        st.info("👋 First visit detected (cookie set).")
    else:
        st.success("Welcome back!")

# Loading the data and training the model
try:
    df_raw, model, test_acc = load_data_and_train(CSV_PATH)
except FileNotFoundError:
    st.error(f"Could not find dataset at '{CSV_PATH}'. Make sure the file exists.")
    st.stop()

# Page
if selected == "Home":
    st.title("🩺 Diabetes Predictor — Home")

    st.write("""
    Welcome! This app uses a Logistic Regression model to predict diabetes,
    the page uses the same diabetes dataset from the previous assignment.
    
    **What you can do:**
    - Go to the **Predict** page to enter patient values and get a risk prediction.
    - Upload a CSV of patients for batch predictions.
    - Go to the **Explore** page to view the dataset and explore feature patterns.
    """)

    # Summary cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", len(df_raw))
    with col2:
        st.metric("Features", len(FEATURES))
    with col3:
        st.metric("Baseline Diabetes Rate", f"{df_raw['Outcome'].mean():.0%}")

# ---------------------------------------------------
elif selected == "Predict":
    st.title("⚡ Predict Diabetes Risk")

    # Tabs for single prediction vs batch upload
    tab1, tab2 = st.tabs(["Single Input", "Batch Upload"])

    # Tab 1: Single input
    with tab1:
        st.subheader("Enter Patient Information")

        # Empty form to group inputs
        with st.form("predict_form"):
            col1, col2, col3 = st.columns(3)

            Pregnancies = col1.number_input("Pregnancies", 0, 20, 2)
            Glucose = col2.number_input("Glucose", 0, 300, 120)
            BloodPressure = col3.number_input("BloodPressure", 0, 200, 70)
            SkinThickness = col1.number_input("SkinThickness", 0, 100, 20)
            Insulin = col2.number_input("Insulin", 0, 900, 79)
            BMI = col3.number_input("BMI", 0.0, 80.0, 32.0)
            DiabetesPedigreeFunction = col1.number_input("Diabetes Pedigree", 0.0, 3.0, 0.5)
            Age = col2.number_input("Age", 1, 120, 33)

            submitted = st.form_submit_button("Predict")

        if submitted:
            # Creating DataFrame for model
            input_data = pd.DataFrame([{
                "Pregnancies": Pregnancies,
                "Glucose": Glucose,
                "BloodPressure": BloodPressure,
                "SkinThickness": SkinThickness,
                "Insulin": Insulin,
                "BMI": BMI,
                "DiabetesPedigreeFunction": DiabetesPedigreeFunction,
                "Age": Age
            }])

            # Handling zeros as missing
            input_data = to_nan_zeros(input_data)

            # Model prediction
            pred = model.predict(input_data)[0]
            prob = model.predict_proba(input_data)[0][1]

            st.success(f"Prediction: **{'Diabetic' if pred == 1 else 'Not Diabetic'}**")
            st.metric("Risk Probability", f"{prob:.2%}")

    # Tab 2: Batch upload
    with tab2:
        st.subheader("Upload CSV File")

        uploaded_file = st.file_uploader("Upload a CSV with the same columns")

        if uploaded_file:
            df_input = pd.read_csv(uploaded_file)

            # Checking for missing columns
            missing_cols = [col for col in FEATURES if col not in df_input.columns]
            if missing_cols:
                st.error(f"Missing columns in uploaded file: {missing_cols}")
            else:
                df_clean = to_nan_zeros(df_input.copy())
                preds = model.predict(df_clean)
                probs = model.predict_proba(df_clean)[:, 1]

                df_input["Prediction"] = preds
                df_input["Probability"] = probs

                st.write("Prediction Results:")
                st.dataframe(df_input.head())

                st.download_button(
                    "Download full results as CSV",
                    df_input.to_csv(index=False),
                    file_name="batch_predictions.csv"
                )

# ---------------------------------------------------
elif selected == "Explore":
    st.title("📊 Explore the Dataset")

    st.write("Below is the full dataset used to train the model:")

    st.dataframe(df_raw.head())

    # Example chart - feature vs outcome
    numeric_cols = df_raw.select_dtypes(include=["float64", "int64"]).columns.tolist()
    feature_to_plot = st.selectbox("Select a feature to visualize", numeric_cols)

    st.write(f"Average {feature_to_plot} by Outcome")
    chart_data = df_raw.groupby("Outcome")[feature_to_plot].mean()
    st.bar_chart(chart_data)
