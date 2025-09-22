import pandas as pd
import os
from dotenv import load_dotenv
import streamlit as st
from openai import AzureOpenAI
import datetime
import random
import requests

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = "2024-12-01-preview"
AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-4o-raj"

client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

# -----------------------------
# Helper Function
# -----------------------------
def get_genai_response(prompt, max_tokens=500, temperature=0.4):
    try:
        resp = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"⚠️ GenAI error: {e}"

# -----------------------------
# Streamlit UI
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
This **Energy Trader Assistant** supports smarter, data-driven trading using GenAI.  
Functional areas include Market Summary, Forecast Deviation, Regulatory Advisory, Trade Logs, Contract Clause Analysis, Price Forecast, Emerging Risks, and Trading Strategy.
""")

# -----------------------------
# Tab 1: Market Summary
# -----------------------------
with main_tabs[1]:
    st.subheader("📈 Market Summary")
    try:
        url = "https://data.nationalgrideso.com/api/3/action/datastore_search?resource_id=4f6ec4a3-dc81-4c25-b01c-cd15ea52b421&limit=10"
        response = requests.get(url)
        records = response.json()["result"]["records"]
        market_df = pd.DataFrame(records)
        st.dataframe(market_df.head())

        prompt = f"Analyze the electricity market summary for price trends and volume shifts:\n\n{market_df.head(10).to_string(index=False)}"
        st.info(get_genai_response(prompt, max_tokens=350))
    except Exception:
        st.warning("⚠️ Could not fetch live market data. Displaying simulated fallback.")
        fallback_df = pd.DataFrame({
            "Region": ["North", "South", "East", "West"],
            "Price (£/kWh)": [0.12, 0.15, 0.14, 0.13],
            "Volume (MWh)": [500, 700, 600, 450],
        })
        st.dataframe(fallback_df)
        prompt = f"Analyze this simulated electricity market summary:\n\n{fallback_df.to_string(index=False)}"
        st.info(get_genai_response(prompt, max_tokens=350))

# -----------------------------
# Tab 2: Forecast Deviation
# -----------------------------
with main_tabs[2]:
    st.subheader("📊 Forecast Deviation")
    if forecast_file and actual_file:
        df_forecast = pd.read_csv(forecast_file)
        df_actual = pd.read_csv(actual_file)
        df_combined = df_forecast.iloc[:, [0, 1, 2]].copy()
        df_combined["Actual (MW)"] = df_actual.iloc[:, 2] if df_actual.shape[1] > 2 else df_actual.iloc[:, 1]
        st.dataframe(df_combined.head())

        prompt = f"Compare forecasted demand vs actual generation and suggest operator actions:\n\n{df_combined.head(10).to_string(index=False)}"
        st.info(get_genai_response(prompt, max_tokens=400))
    else:
        st.warning("Upload both Forecast and Actual Generation CSVs.")

# -----------------------------
# Tab 3: Regulatory Advisory
# -----------------------------
with main_tabs[3]:
    st.subheader("📜 Regulatory Advisory")
    if regulation_file:
        regulation_df = pd.read_csv(regulation_file)
        st.dataframe(regulation_df.head())
        prompt = f"Review regulatory data and highlight trading constraints:\n\n{regulation_df.head(10).to_string(index=False)}"
        st.success(get_genai_response(prompt, max_tokens=400))

# -----------------------------
# Tab 4: Trade Log Insights
# -----------------------------
with main_tabs[4]:
    st.subheader("📒 Trade Log Insights")
    if trade_file:
        trade_df = pd.read_csv(trade_file)
        st.dataframe(trade_df.head())
        prompt = f"From this trade log, identify missed opportunities and suggest buy/sell with price, quantity, and region:\n\n{trade_df.head(10).to_string(index=False)}"
        st.success(get_genai_response(prompt, max_tokens=400))

# -----------------------------
# Tab 5: Contract Clause Analysis
# -----------------------------
with main_tabs[5]:
    st.subheader("📑 Contract Clause Analysis")
    if contract_file:
        contract_text = contract_file.read().decode("utf-8")
        prompt = f"From this energy contract, extract payment terms, penalties, restrictions, and provide recommendations:\n\n{contract_text[:1000]}"
        st.success(get_genai_response(prompt, max_tokens=500))

# -----------------------------
# Tab 6: Price Forecast AI
# -----------------------------
with main_tabs[6]:
    st.subheader("📈 Price Forecast AI")
    if forecast_file:
        df = pd.read_csv(forecast_file)
        today = datetime.date.today()
        price_data = [(today + datetime.timedelta(days=i), round(random.uniform(0.11, 0.18), 3)) for i in range(30)]
        price_df = pd.DataFrame(price_data, columns=["Date", "Predicted Price (£/kWh)"])
        st.dataframe(price_df)

        prompt = f"Based on this 30-day forecast, recommend trading decisions:\n\n{price_df.to_string(index=False)}"
        st.info(get_genai_response(prompt, max_tokens=500))

        if all([forecast_file, actual_file, trade_file, contract_file]):
            forecast_df = pd.read_csv(forecast_file).head(10)
            actual_df = pd.read_csv(actual_file).head(10)
            trade_df = pd.read_csv(trade_file).head(10)
            contract_excerpt = contract_file.read().decode("utf-8")[:1000]

            prompt = f"""You are an energy trading strategist. Based on the following data, give specific buy/sell recommendations:

FORECAST:
{forecast_df.to_string(index=False)}

ACTUAL:
{actual_df.to_string(index=False)}

TRADES:
{trade_df.to_string(index=False)}

CONTRACT:
{contract_excerpt}"""
            st.success(get_genai_response(prompt, max_tokens=1000))
    else:
        st.warning("Upload Forecast Demand CSV.")

# -----------------------------
# Tab 7: Emerging Risks
# -----------------------------
with main_tabs[7]:
    st.subheader("🚨 Emerging Risks")
    try:
        url = "https://data.nationalgrideso.com/api/3/action/datastore_search?resource_id=4f6ec4a3-dc81-4c25-b01c-cd15ea52b421&limit=10"
        response = requests.get(url)
        df = pd.DataFrame(response.json()["result"]["records"])
        st.dataframe(df)
        prompt = f"Analyze real-time electricity market data for emerging risks:\n\n{df.head(10).to_string(index=False)}"
        st.success(get_genai_response(prompt, max_tokens=500))
    except Exception:
        st.warning("⚠️ Live API failed. Showing simulated risks.")
        fallback_df = pd.DataFrame({
            "Region": ["North", "South", "East", "West"],
            "Warning Level": ["High", "Medium", "Low", "Medium"],
            "Signal": ["Capacity risk", "Price spike", "Stable", "Congestion"],
        })
        st.dataframe(fallback_df)
        prompt = f"Analyze fallback risk signals and recommend strategies:\n\n{fallback_df.to_string(index=False)}"
        st.success(get_genai_response(prompt, max_tokens=500))

# -----------------------------
# Tab 8: Trading Strategy
# -----------------------------
with main_tabs[8]:
    st.subheader("📘 Trading Strategy")
    if all([forecast_file, actual_file, trade_file, contract_file]):
        forecast_df = pd.read_csv(forecast_file).head(10)
        actual_df = pd.read_csv(actual_file).head(10)
        trade_df = pd.read_csv(trade_file).head(10)
        contract_excerpt = contract_file.read().decode("utf-8")[:1000]

        prompt = f"""Based on the following data, recommend a specific trading strategy:
- What to buy/sell?
- At what price/load?
- Timing for action
- Contractual or regulatory risks

FORECAST:
{forecast_df.to_string(index=False)}

ACTUAL:
{actual_df.to_string(index=False)}

TRADE LOG:
{trade_df.to_string(index=False)}

CONTRACT:
{contract_excerpt}"""
        st.success(get_genai_response(prompt, max_tokens=1000))
    else:
        st.warning("Upload Forecast, Actual, Trade Log, and Contract files.")
