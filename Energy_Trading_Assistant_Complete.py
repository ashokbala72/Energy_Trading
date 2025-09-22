import pandas as pd
import os
import datetime
import random
import requests
import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-raj")

client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

# -----------------------------
# Helper: Safe CSV Reader
# -----------------------------
def safe_read_csv(uploaded_file):
    """Safely read uploaded CSV, return DataFrame or None if empty/invalid."""
    if uploaded_file is None:
        return None
    try:
        uploaded_file.seek(0)
        if hasattr(uploaded_file, "size") and uploaded_file.size == 0:
            st.warning("⚠️ Uploaded file is empty.")
            return None
        df = pd.read_csv(uploaded_file)
        if df.empty:
            st.warning("⚠️ Uploaded CSV has no rows.")
            return None
        return df
    except pd.errors.EmptyDataError:
        st.error("❌ Uploaded file has no data (Empty CSV).")
    except Exception as e:
        st.error(f"❌ Could not read CSV: {e}")
    return None

# -----------------------------
# Helper: GenAI Response
# -----------------------------
def get_genai_response(prompt, max_tokens=400, temperature=0.4):
    try:
        resp = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"❌ GenAI call failed: {str(e)}"

# -----------------------------
# Streamlit UI Setup
# -----------------------------
st.set_page_config(layout="wide")
st.title("⚡ AI-Powered Energy Trading Assistant")

st.sidebar.header("📂 Upload Input Files")
forecast_file = st.sidebar.file_uploader("📁 Forecast Demand CSV", type=["csv"])
actual_file = st.sidebar.file_uploader("📁 Actual Generation CSV", type=["csv"])
regulation_file = st.sidebar.file_uploader("📁 Regulatory Bulletin CSV", type=["csv"])
trade_file = st.sidebar.file_uploader("📁 Trade Log CSV", type=["csv"])
contract_file = st.sidebar.file_uploader("📁 Contract / PPA TXT", type=["txt"])

main_tabs = st.tabs([
    "📌 Overview", "📈 Market Summary", "📊 Forecast Deviation", "📜 Regulatory Advisory",
    "📒 Trade Log", "📑 Contract Clause Analysis", "📈 Price Forecast AI",
    "🚨 Emerging Risks", "📘 Trading Strategy"
])

# -----------------------------
# Tab 0: Overview
# -----------------------------
with main_tabs[0]:
    st.subheader("📌 Application Overview: Energy Trader")
    st.markdown("""
### 🌟 Purpose
The **Energy Trader Assistant** is a GenAI-powered platform designed to assist energy utilities and trading professionals.

### 📅 Inputs
- Forecasted vs Actual Generation (CSV)
- Trade Logs (CSV)
- Regulatory Bulletins (CSV)
- Power Contracts / PPAs (Text file)
- Market Summary Data (API or simulated)

### 🧠 Functional Areas
1. Market Summary
2. Forecast Deviation
3. Regulatory Advisory
4. Trade Log Insights
5. Contract Clause Analysis
6. Trading Strategy
7. Price Forecast AI
8. Emerging Risks
    """)

# -----------------------------
# Tab 1: Market Summary
# -----------------------------
with main_tabs[1]:
    st.subheader("📈 Market Summary")
    try:
        url = "https://data.nationalgrideso.com/api/3/action/datastore_search?resource_id=4f6ec4a3-dc81-4c25-b01c-cd15ea52b421&limit=10"
        response = requests.get(url, timeout=10)
        records = response.json()["result"]["records"]
        market_df = pd.DataFrame(records)
        st.dataframe(market_df.head())
        st.info(get_genai_response(f"Analyze market summary:\n{market_df.head(10).to_string(index=False)}"))
    except Exception:
        st.warning("⚠️ Live market API failed, using fallback.")
        fallback_df = pd.DataFrame({
            'Region': ['North', 'South', 'East', 'West'],
            'Price (£/kWh)': [0.12, 0.15, 0.14, 0.13],
            'Volume (MWh)': [500, 700, 600, 450]
        })
        st.dataframe(fallback_df)
        st.info(get_genai_response(f"Analyze fallback market:\n{fallback_df.to_string(index=False)}"))

# -----------------------------
# Tab 2: Forecast Deviation
# -----------------------------
with main_tabs[2]:
    st.subheader("📊 Forecast Deviation")
    df_forecast = safe_read_csv(forecast_file)
    df_actual = safe_read_csv(actual_file)
    if df_forecast is not None and df_actual is not None:
        df_combined = df_forecast.iloc[:, :2].copy()
        df_combined['Actual (MW)'] = df_actual.iloc[:, -1]
        st.dataframe(df_combined.head())
        st.info(get_genai_response(f"Compare forecast vs actual:\n{df_combined.head(10).to_string(index=False)}"))
    else:
        st.warning("Upload both Forecast and Actual CSVs.")

# -----------------------------
# Tab 3: Regulatory Advisory
# -----------------------------
with main_tabs[3]:
    st.subheader("📜 Regulatory Advisory")
    df_reg = safe_read_csv(regulation_file)
    if df_reg is not None:
        st.dataframe(df_reg.head())
        st.success(get_genai_response(f"Review regulatory data:\n{df_reg.head(10).to_string(index=False)}"))
    else:
        st.warning("Upload a Regulatory CSV.")

# -----------------------------
# Tab 4: Trade Log
# -----------------------------
with main_tabs[4]:
    st.subheader("📒 Trade Log Insights")
    df_trade = safe_read_csv(trade_file)
    if df_trade is not None:
        st.dataframe(df_trade.head())
        st.success(get_genai_response(f"Trade log insights:\n{df_trade.head(10).to_string(index=False)}"))
    else:
        st.warning("Upload Trade Log CSV.")

# -----------------------------
# Tab 5: Contract Clause Analysis
# -----------------------------
with main_tabs[5]:
    st.subheader("📑 Contract Clause Analysis")
    if contract_file:
        contract_file.seek(0)
        contract_text = contract_file.read().decode("utf-8")
        st.success(get_genai_response(f"Contract analysis:\n{contract_text[:1000]}"))
    else:
        st.warning("Upload Contract TXT.")

# -----------------------------
# Tab 6: Price Forecast AI
# -----------------------------
with main_tabs[6]:
    st.subheader("📈 Price Forecast AI")
    df_forecast = safe_read_csv(forecast_file)
    if df_forecast is not None:
        today = datetime.date.today()
        price_data = [(today + datetime.timedelta(days=i), round(random.uniform(0.11, 0.18), 3)) for i in range(30)]
        price_df = pd.DataFrame(price_data, columns=["Date", "Predicted Price (£/kWh)"])
        st.dataframe(price_df)
        st.info(get_genai_response(f"Price forecast strategy:\n{price_df.to_string(index=False)}"))
    else:
        st.warning("Upload Forecast CSV for price forecast.")

# -----------------------------
# Tab 7: Emerging Risks
# -----------------------------
with main_tabs[7]:
    st.subheader("🚨 Emerging Risks")
    try:
        url = "https://data.nationalgrideso.com/api/3/action/datastore_search?resource_id=4f6ec4a3-dc81-4c25-b01c-cd15ea52b421&limit=10"
        response = requests.get(url, timeout=10)
        df = pd.DataFrame(response.json()["result"]["records"])
        st.dataframe(df)
        st.success(get_genai_response(f"Emerging risks:\n{df.head(10).to_string(index=False)}"))
    except Exception:
        st.warning("⚠️ Risk API failed, using fallback.")
        fallback_df = pd.DataFrame({
            'Region': ['North', 'South', 'East', 'West'],
            'Warning Level': ['High', 'Medium', 'Low', 'Medium'],
            'Signal': ['Capacity risk', 'Price spike', 'Stable', 'Congestion']
        })
        st.dataframe(fallback_df)
        st.success(get_genai_response(f"Fallback risks:\n{fallback_df.to_string(index=False)}"))

# -----------------------------
# Tab 8: Trading Strategy
# -----------------------------
with main_tabs[8]:
    st.subheader("📘 Trading Strategy")
    df_forecast = safe_read_csv(forecast_file)
    df_actual = safe_read_csv(actual_file)
    df_trade = safe_read_csv(trade_file)
    if df_forecast is not None and df_actual is not None and df_trade is not None and contract_file:
        contract_file.seek(0)
        contract_text = contract_file.read().decode("utf-8")[:1000]
        combined = f"""FORECAST:\n{df_forecast.head(10).to_string(index=False)}

ACTUAL:\n{df_actual.head(10).to_string(index=False)}

TRADE LOG:\n{df_trade.head(10).to_string(index=False)}

CONTRACT:\n{contract_text}"""
        st.success(get_genai_response(f"Trading strategy based on:\n{combined}", max_tokens=1000))
    else:
        st.warning("Upload Forecast, Actual, Trade Log and Contract files.")
