import pandas as pd
import os
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI, OpenAIError
import datetime
import random
import requests

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(layout="wide")
st.title("⚡ AI-Powered Energy Trading Assistant")

st.sidebar.header("📂 Upload Input Files")
forecast_file = st.sidebar.file_uploader("📁 Forecast Demand CSV", type=["csv"])
actual_file = st.sidebar.file_uploader("📁 Actual Generation CSV", type=["csv"])
regulation_file = st.sidebar.file_uploader("📁 Regulatory Bulletin CSV", type=["csv"])
trade_file = st.sidebar.file_uploader("📁 Trade Log CSV", type=["csv"])
contract_file = st.sidebar.file_uploader("📁 Contract / PPA TXT", type=["txt"])

main_tabs = st.tabs(["📌 Overview", "📈 Market Summary", "📊 Forecast Deviation", "📜 Regulatory Advisory", "📒 Trade Log", "📑 Contract Clause Analysis", "📈 Price Forecast AI", "🚨 Emerging Risks", "📘 Trading Strategy"])

with main_tabs[0]:
    st.subheader("📌 Application Overview: Energy Trader")
    st.markdown("""
### 🌟 Purpose
The **Energy Trader Assistant** is a GenAI-powered platform designed to assist energy utilities and trading professionals in making smarter, data-driven decisions across the power market value chain. It focuses on market trend analysis, forecasting deviation, trading opportunity detection, regulatory interpretation, and contract clause analysis.

### 📅 Inputs
- Forecasted vs Actual Generation (CSV)
- Trade Logs (CSV)
- Regulatory Bulletins (CSV)
- Power Contracts / PPAs (Text file)
- Market Summary Data (Live API or simulated)

### 🧠 Functional Areas
1. **Market Summary** – Analyze price trends and congestion signals; provides GenAI insight into regional price spikes and trading volumes.
2. **Forecast Deviation** – Compare forecasted vs actual generation; flags under/overestimation zones for operator action.
3. **Regulatory Advisory** – Extract impact of rules on trade actions; identifies clauses affecting trade flexibility and recommends compliance steps.
4. **Trade Log Insights** – Spot buy/sell triggers and missed opportunities; suggests improved trade execution based on historical behavior.
5. **Contract Clause Analysis** – Identify risks, penalties, obligations; flags payment terms, penalties, and operational restrictions.
6. **Trading Strategy Generator** – Recommend actions based on all inputs; provides actionable buy/sell moves using GenAI synthesis.
7. **Price Forecast AI** – Forecast prices for the next 30 days and provide GenAI trading recommendations.
8. **Emerging Risks** – Detect real-time market and regulatory risks and propose mitigation strategies.

### 🛠️ Technologies Used
- Streamlit – UI rendering
- OpenAI GPT-3.5 – Natural language analysis
- pandas – Dataframe processing
- dotenv – Secure API key management

### 🌟 User Benefits
- Strategic GenAI insights from raw CSV and market feeds
- Improved contract, regulation, and trade decision-making
- Schedule optimization to reduce imbalance penalties

### 🚀 Making it Production Ready
To move this to production:
- 🔐 Secure API access with role-based controls
- 🔄 Automate file ingestion via IEX APIs, SCADA streams
- 🩵 Enable GenAI output logging with audit trail
- 🧺 Add unit tests and strategy validation rules
- ☁️ Deploy with Docker, Streamlit Cloud or Azure WebApp with monitoring hooks
""")

with main_tabs[1]:
    st.subheader("📈 Market Summary")
    try:
        url = "https://data.nationalgrideso.com/api/3/action/datastore_search?resource_id=4f6ec4a3-dc81-4c25-b01c-cd15ea52b421&limit=10"
        response = requests.get(url)
        records = response.json()["result"]["records"]
        market_df = pd.DataFrame(records)
        st.dataframe(market_df.head())

        prompt = f"""Analyze the electricity market summary for price trends and volume shifts:

{market_df.head(10).to_string(index=False)}"""
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=350,
            temperature=0.4,
        )
        st.info(resp.choices[0].message.content)
    except Exception as e:
        st.warning("⚠️ Could not fetch live market data. Displaying simulated fallback.")
        fallback_df = pd.DataFrame({
            'Region': ['North', 'South', 'East', 'West'],
            'Price (£/kWh)': [0.12, 0.15, 0.14, 0.13],
            'Volume (MWh)': [500, 700, 600, 450]
        })
        st.dataframe(fallback_df)

        prompt = f"""Analyze this simulated electricity market summary for price trends and volume shifts:

{fallback_df.to_string(index=False)}"""
        try:
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=350,
                temperature=0.4,
            )
            st.info(resp.choices[0].message.content)
        except Exception as e2:
            st.error(f"GenAI fallback failed: {e2}")
    else:
        st.warning("Upload both forecast and actual files.")



with main_tabs[2]:
    st.subheader("📊 Forecast Deviation")
    if forecast_file and actual_file:
        forecast_file.seek(0)
        actual_file.seek(0)
        df_forecast = pd.read_csv(forecast_file)
        df_actual = pd.read_csv(actual_file)
        df_combined = df_forecast.iloc[:, [0, 1, 2]].copy()
        df_combined['Actual (MW)'] = df_actual.iloc[:, 2] if df_actual.shape[1] > 2 else df_actual.iloc[:, 1]
        st.dataframe(df_combined.head())

        prompt = f"""Compare forecasted demand vs actual generation. Identify mismatches and suggest operator actions:

{df_combined.head(10).to_string(index=False)}"""
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.4,
        )
        st.info(resp.choices[0].message.content)
    else:
        st.warning("Upload both Forecast and Actual Generation CSVs to proceed.")
with main_tabs[3]:
    st.subheader("📜 Regulatory Advisory")
    if regulation_file:
        regulation_file.seek(0)
        regulation_df = pd.read_csv(regulation_file)
        st.dataframe(regulation_df.head())

        prompt = f"""Review regulatory data and highlight any trading constraints or compliance clauses:

{regulation_df.head(10).to_string(index=False)}"""
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.4,
        )
        st.success(resp.choices[0].message.content)

with main_tabs[4]:
    st.subheader("📒 Trade Log Insights")
    if trade_file:
        trade_file.seek(0)
        trade_df = pd.read_csv(trade_file)
        st.dataframe(trade_df.head())

        prompt = f"""Based on this trade log, identify missed opportunities AND recommend what should be bought or sold at what price, quantity, and region:

{trade_df.head(10).to_string(index=False)}"""
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.4,
        )
        st.success(resp.choices[0].message.content)

with main_tabs[5]:
    st.subheader("📑 Contract Clause Analysis")
    if contract_file:
        contract_file.seek(0)
        contract_text = contract_file.read().decode("utf-8")
        prompt = f"""From this energy contract excerpt, extract:
- 📌 Key payment terms
- ❗ Penalty clauses
- 🚫 Trade restrictions

Then provide GenAI recommendations for trade alignment and risk mitigation:


{contract_text[:1000]}"""
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.4,
        )
        st.success(resp.choices[0].message.content)

with main_tabs[6]:
    st.subheader("📈 Price Forecast AI")
    if forecast_file:
        forecast_file.seek(0)
        df = pd.read_csv(forecast_file)
        today = datetime.date.today()
        price_data = [(today + datetime.timedelta(days=i), round(random.uniform(0.11, 0.18), 3)) for i in range(30)]
        price_df = pd.DataFrame(price_data, columns=["Date", "Predicted Price (£/kWh)"])
        st.dataframe(price_df)

        prompt = f"""Based on this 30-day price forecast, recommend specific energy trading decisions:

{price_df.to_string(index=False)}"""
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.4,
        )
        st.info(resp.choices[0].message.content)
    else:
        st.warning("Upload Forecast Demand CSV to generate price forecast.")
    if all([forecast_file, actual_file, trade_file, contract_file]):
        forecast_file.seek(0)
        actual_file.seek(0)
        trade_file.seek(0)
        contract_file.seek(0)
        forecast_df = pd.read_csv(forecast_file).head(10)
        actual_df = pd.read_csv(actual_file).head(10)
        trade_df = pd.read_csv(trade_file).head(10)
        contract_text = contract_file.read().decode("utf-8")[:1000]

        prompt = f"""You are an energy trading strategist. Use the forecast, actuals, trade logs, and contract data below to answer:

✅ What should be bought?
✅ What should be sold?
💷 At what price?
🕒 At what time block?
📌 Incorporate any known risks from market conditions.

Your analysis should be specific and actionable.

Data follows:

FORECAST:
{forecast_df.to_string(index=False)}

ACTUAL:
{actual_df.to_string(index=False)}

TRADES:
{trade_df.to_string(index=False)}

CONTRACT:
{contract_text}"""
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.4,
        )
        st.success(resp.choices[0].message.content)

with main_tabs[7]:
    st.subheader("🚨 Emerging Risks")
    try:
        url = "https://data.nationalgrideso.com/api/3/action/datastore_search?resource_id=4f6ec4a3-dc81-4c25-b01c-cd15ea52b421&limit=10"
        response = requests.get(url)
        df = pd.DataFrame(response.json()["result"]["records"])
        st.dataframe(df)

        prompt = f"""Analyze real-time electricity market data for emerging risks. Recommend mitigation strategies for energy traders:

{df.head(10).to_string(index=False)}"""
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.4,
        )
        st.success(resp.choices[0].message.content)
    except Exception as e:
        st.warning("⚠️ Live API failed. Showing simulated risks.")
        fallback_df = pd.DataFrame({
            'Region': ['North', 'South', 'East', 'West'],
            'Warning Level': ['High', 'Medium', 'Low', 'Medium'],
            'Signal': ['Capacity risk', 'Price spike', 'Stable', 'Congestion']
        })
        st.dataframe(fallback_df)
        prompt = f"""Analyze fallback electricity risk signals. Recommend mitigation strategies for trading teams:

{fallback_df.to_string(index=False)}"""
        try:
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.4,
            )
            st.success(resp.choices[0].message.content)
        except Exception as e2:
            st.error(f"GenAI fallback failed: {e2}")

with main_tabs[8]:
    st.subheader("📘 Trading Strategy")
    if all([forecast_file, actual_file, trade_file, contract_file]):
        forecast_file.seek(0)
        actual_file.seek(0)
        trade_file.seek(0)
        contract_file.seek(0)
        forecast_df = pd.read_csv(forecast_file).head(10)
        actual_df = pd.read_csv(actual_file).head(10)
        trade_df = pd.read_csv(trade_file).head(10)
        contract_text = contract_file.read().decode("utf-8")[:1000]

        prompt = f"""Based on the following market data and contracts, recommend specific trading strategy actions:

- What to buy/sell?
- At what price or load?
- Timing for action
- Contractual or regulatory risks to watch for

FORECAST:
{forecast_df.to_string(index=False)}

ACTUAL:
{actual_df.to_string(index=False)}

TRADE LOG:
{trade_df.to_string(index=False)}

CONTRACT EXCERPT:
{contract_text}"""
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.4,
        )
        st.success(resp.choices[0].message.content)
    else:
        st.warning("Please upload Forecast, Actual, Trade Log, and Contract files.")