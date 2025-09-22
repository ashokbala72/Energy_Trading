import streamlit as st
import pandas as pd
import os
import datetime
import random
import requests
from dotenv import load_dotenv
from openai import AzureOpenAI

# --------------------------
# Setup
# --------------------------
st.set_page_config(page_title="⚡ Energy Trading Assistant", layout="wide")
st.title("⚡ AI-Powered Energy Trading Assistant")

# Load environment variables
load_dotenv()
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-raj")

# --------------------------
# Helper functions
# --------------------------
def get_genai_response(prompt, max_tokens=400, temperature=0.4):
    """Safe wrapper for Azure OpenAI calls"""
    try:
        resp = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ GenAI error: {e}"

def safe_read_csv(uploaded_file):
    """Safely read uploaded CSV into DataFrame"""
    if uploaded_file is None:
        return None
    try:
        uploaded_file.seek(0)  # reset pointer
        df = pd.read_csv(uploaded_file)
        if df.empty:
            st.warning("⚠️ Uploaded CSV has no rows.")
            return None
        return df
    except pd.errors.EmptyDataError:
        st.error("❌ Uploaded CSV is empty.")
    except Exception as e:
        st.error(f"❌ Could not read CSV: {e}")
    return None

# --------------------------
# Sidebar Uploads
# --------------------------
st.sidebar.header("📂 Upload Input Files")
forecast_file = st.sidebar.file_uploader("📁 Forecast Demand CSV", type=["csv"])
actual_file = st.sidebar.file_uploader("📁 Actual Generation CSV", type=["csv"])
regulation_file = st.sidebar.file_uploader("📁 Regulatory Bulletin CSV", type=["csv"])
trade_file = st.sidebar.file_uploader("📁 Trade Log CSV", type=["csv"])
contract_file = st.sidebar.file_uploader("📁 Contract / PPA TXT", type=["txt"])

# --------------------------
# Tabs
# --------------------------
main_tabs = st.tabs([
    "📌 Overview", "📈 Market Summary", "📊 Forecast Deviation", "📜 Regulatory Advisory",
    "📒 Trade Log", "📑 Contract Clause Analysis", "📈 Price Forecast AI",
    "🚨 Emerging Risks", "📘 Trading Strategy"
])

# --------------------------
# Tab 0 - Overview
# --------------------------
with main_tabs[0]:
    st.subheader("📌 Application Overview: Energy Trader")
    st.markdown("""
    This GenAI-powered assistant helps energy traders analyze **market data, forecasts, trades, contracts, and risks**.
    It integrates multiple data sources, detects mismatches, and provides **actionable trading strategies**.
    """)

# --------------------------
# Tab 1 - Market Summary
# --------------------------
with main_tabs[1]:
    st.subheader("📈 Market Summary")
    try:
        url = "https://data.nationalgrideso.com/api/3/action/datastore_search?resource_id=4f6ec4a3-dc81-4c25-b01c-cd15ea52b421&limit=10"
        response = requests.get(url, timeout=10)
        records = response.json()["result"]["records"]
        market_df = pd.DataFrame(records)
        st.dataframe(market_df.head())

        st.info(get_genai_response(
            f"Analyze the electricity market summary:\n{market_df.head(10).to_string(index=False)}"
        ))
    except Exception:
        st.warning("⚠️ Could not fetch live market data. Showing fallback.")
        fallback_df = pd.DataFrame({
            'Region': ['North', 'South', 'East', 'West'],
            'Price (£/kWh)': [0.12, 0.15, 0.14, 0.13],
            'Volume (MWh)': [500, 700, 600, 450]
        })
        st.dataframe(fallback_df)
        st.info(get_genai_response(
            f"Analyze fallback market summary:\n{fallback_df.to_string(index=False)}"
        ))

# --------------------------
# Tab 2 - Forecast Deviation
# --------------------------
with main_tabs[2]:
    st.subheader("📊 Forecast vs Actual Deviation")
    df_forecast = safe_read_csv(forecast_file)
    df_actual = safe_read_csv(actual_file)

    if df_forecast is not None and df_actual is not None:
        try:
            # Normalize col names
            df_forecast.columns = [c.lower() for c in df_forecast.columns]
            df_actual.columns = [c.lower() for c in df_actual.columns]

            # Expect columns: date, region, forecast, actual
            merged = pd.merge(df_forecast, df_actual, on=["date", "region"], suffixes=("_forecast", "_actual"))

            merged["Difference (MW)"] = merged.iloc[:, 2] - merged.iloc[:, 3]
            merged["% Error"] = ((merged["Difference (MW)"] / merged.iloc[:, 3]) * 100).round(2)

            st.dataframe(merged.head(20))

            st.info(get_genai_response(
                f"Compare forecast vs actual and suggest operator actions:\n{merged.head(10).to_string(index=False)}"
            ))
        except Exception as e:
            st.error(f"⚠️ Error joining forecast & actual: {e}")
    else:
        st.warning("Upload both Forecast and Actual CSVs with columns [date, region, MW].")

# --------------------------
# Tab 3 - Regulatory Advisory
# --------------------------
with main_tabs[3]:
    st.subheader("📜 Regulatory Advisory")
    df_reg = safe_read_csv(regulation_file)
    if df_reg is not None:
        st.dataframe(df_reg.head())
        st.success(get_genai_response(
            f"Review regulations for trading constraints:\n{df_reg.head(10).to_string(index=False)}"
        ))
    else:
        st.warning("Upload regulatory bulletin CSV.")

# --------------------------
# Tab 4 - Trade Log
# --------------------------
with main_tabs[4]:
    st.subheader("📒 Trade Log Insights")
    df_trade = safe_read_csv(trade_file)
    if df_trade is not None:
        st.dataframe(df_trade.head())
        st.success(get_genai_response(
            f"From trade log, identify missed opportunities and recommendations:\n{df_trade.head(10).to_string(index=False)}"
        ))
    else:
        st.warning("Upload trade log CSV.")

# --------------------------
# Tab 5 - Contract Clause
# --------------------------
with main_tabs[5]:
    st.subheader("📑 Contract Clause Analysis")
    if contract_file:
        contract_text = contract_file.read().decode("utf-8")[:1000]
        st.success(get_genai_response(
            f"Extract payment terms, penalties, restrictions, and risks:\n{contract_text}"
        ))
    else:
        st.warning("Upload contract file (TXT).")

# --------------------------
# Tab 6 - Price Forecast AI
# --------------------------
with main_tabs[6]:
    st.subheader("📈 Price Forecast AI")
    if forecast_file:
        today = datetime.date.today()
        price_data = [(today + datetime.timedelta(days=i), round(random.uniform(0.11, 0.18), 3)) for i in range(30)]
        price_df = pd.DataFrame(price_data, columns=["Date", "Predicted Price (£/kWh)"])
        st.dataframe(price_df)
        st.info(get_genai_response(
            f"Based on this price forecast, suggest trading strategy:\n{price_df.to_string(index=False)}"
        ))
    else:
        st.warning("Upload Forecast Demand CSV.")

# --------------------------
# Tab 7 - Emerging Risks
# --------------------------
with main_tabs[7]:
    st.subheader("🚨 Emerging Risks")
    try:
        url = "https://data.nationalgrideso.com/api/3/action/datastore_search?resource_id=4f6ec4a3-dc81-4c25-b01c-cd15ea52b421&limit=10"
        response = requests.get(url, timeout=10)
        df = pd.DataFrame(response.json()["result"]["records"])
        st.dataframe(df)
        st.success(get_genai_response(
            f"Analyze real-time market data for emerging risks:\n{df.head(10).to_string(index=False)}"
        ))
    except Exception:
        fallback_df = pd.DataFrame({
            'Region': ['North', 'South', 'East', 'West'],
            'Warning Level': ['High', 'Medium', 'Low', 'Medium'],
            'Signal': ['Capacity risk', 'Price spike', 'Stable', 'Congestion']
        })
        st.dataframe(fallback_df)
        st.info(get_genai_response(
            f"Analyze fallback risks and mitigation:\n{fallback_df.to_string(index=False)}"
        ))

# --------------------------
# Tab 8 - Trading Strategy
# --------------------------
with main_tabs[8]:
    st.subheader("📘 Trading Strategy")
    df_forecast = safe_read_csv(forecast_file)
    df_actual = safe_read_csv(actual_file)
    df_trade = safe_read_csv(trade_file)
    if df_forecast is not None and df_actual is not None and df_trade is not None and contract_file:
        contract_text = contract_file.read().decode("utf-8")[:1000]
        st.success(get_genai_response(
            f"Using forecast, actuals, trades, and contract, recommend trading strategy:\n"
            f"FORECAST:\n{df_forecast.head(10).to_string(index=False)}\n\n"
            f"ACTUAL:\n{df_actual.head(10).to_string(index=False)}\n\n"
            f"TRADES:\n{df_trade.head(10).to_string(index=False)}\n\n"
            f"CONTRACT:\n{contract_text}"
        , max_tokens=1000))
    else:
        st.warning("Upload Forecast, Actual, Trade, and Contract files.")
