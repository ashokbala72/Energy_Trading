import pandas as pd
import os
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI, OpenAIError

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

main_tabs = st.tabs(["📌 Overview", "📈 Market Summary", "📊 Forecast Deviation", "📜 Regulatory Advisory", "📒 Trade Log", "📑 Contract Clause Analysis", "📘 Trading Strategy", "📉 Price/Volume Tracker", "📊 Forecast Accuracy", "⚠️ Risk & Clause Flags", "📆 Schedule Optimizer"])

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
7. **Price/Volume Tracker** – Analyze price-volume dynamics using GenAI; interprets trading patterns to advise on price-sensitive trading windows.
8. **Forecast Accuracy** – Evaluate prediction errors and suggest improvements; GenAI highlights reasons for poor forecasts and fixes.
9. **Risk & Clause Flags** – Flag contract constraints using LLM; extracts hidden risks and alerts on legal constraints in PPAs.
10. **Schedule Optimizer** – Suggest optimal buy/sell time blocks; generates optimized trade schedules aligned with demand and contract rules.

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
    st.info("🔄 Fetching real-time market data from National Grid ESO API...")
    try:
        import requests
        url = "https://data.nationalgrideso.com/api/3/action/datastore_search?resource_id=4f6ec4a3-dc81-4c25-b01c-cd15ea52b421&limit=10"
        response = requests.get(url)
        records = response.json()["result"]["records"]
        market_df = pd.DataFrame(records)
        st.dataframe(market_df.head())

        prompt = f"""Analyze this real-time electricity market summary for price trends, congestion signals, and significant volume shifts:

{market_df.head(10).to_string(index=False)}"""
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=350,
            temperature=0.4,
        )
        st.info(resp.choices[0].message.content)
    except Exception as e:
        st.warning("⚠️ Could not fetch live market data. Displaying simulated data.")
        fallback_df = pd.DataFrame({
            'Region': ['North', 'South', 'East', 'West', 'Midlands', 'Scotland', 'Wales', 'London'],
            'Price (£/kWh)': [0.12, 0.15, 0.14, 0.13, 0.11, 0.16, 0.13, 0.17],
            'Volume (MWh)': [500, 700, 600, 450, 520, 640, 580, 720]
        })
        st.dataframe(fallback_df.head())

        prompt = f"""Analyze this simulated electricity market summary for price trends, congestion signals, and significant volume shifts:

{fallback_df.head(10).to_string(index=False)}"""
        try:
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=350,
                temperature=0.4,
            )
            st.info(resp.choices[0].message.content)
        except Exception as e2:
            st.error(f"GenAI error using fallback data: {e2}")

with main_tabs[2]:
    st.subheader("📊 Forecast Deviation")
    if not forecast_file or not actual_file:
        st.warning("⚠️ You have not uploaded Forecast and Actual Generation CSVs.")
    elif forecast_file and actual_file:
        forecast_file.seek(0)
        actual_file.seek(0)
        df_forecast = pd.read_csv(forecast_file)
        df_actual = pd.read_csv(actual_file)

        combined_df = df_forecast.iloc[:, [0, 1, 2]].copy()
        combined_df['Actual (MW)'] = df_actual.iloc[:, 2] if df_actual.shape[1] > 2 else df_actual.iloc[:, 1]

        st.write("### Forecast vs Actual Generation")
        st.dataframe(combined_df.head())

        prompt = f"""Compare the following demand forecast vs actual generation data. Highlight regions with major mismatches and advise operators on adjustments:

Forecast:
{df_forecast.head(10).to_string(index=False)}

Actual:
{df_actual.head(10).to_string(index=False)}"""
        try:
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.4,
            )
            st.info(resp.choices[0].message.content)
        except Exception as e:
            st.error(f"Error in forecast deviation analysis: {e}")

with main_tabs[3]:
    st.subheader("📜 Regulatory Advisory Analysis")
    if regulation_file:
        try:
            regulation_file.seek(0)
            regulation_df = pd.read_csv(regulation_file)
            st.markdown("### 📜 Advisory Data")
            st.dataframe(regulation_df.head())
            text_sample = regulation_df.to_string(index=False)

            extract_prompt = f"""Extract a table with regulation date, clause ID, and trading impact from the following advisory data:

{text_sample}"""
            extract_resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": extract_prompt}],
                max_tokens=400,
                temperature=0.3,
            )
            st.markdown("### 🗂️ Extracted Regulation Summary")
            st.info(extract_resp.choices[0].message.content)

            prompt = f"""Review the following regulatory advisory. Identify clauses that impact energy trading and list any operational or compliance recommendations:

{text_sample}"""
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.3,
            )
            st.success(resp.choices[0].message.content)
        except Exception as e:
            st.error(f"Error analyzing regulation CSV: {e}")

with main_tabs[4]:
    st.subheader("📒 Trade Log Insights")
    if trade_file:
        try:
            trade_file.seek(0)
            df = pd.read_csv(trade_file)
            st.dataframe(df.head())
            prompt = f"""Based on this trade log, identify buy/sell signals and missed opportunities. Suggest trading actions:

{df.head(10).to_string(index=False)}"""
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=350,
                temperature=0.4,
            )
            st.success(resp.choices[0].message.content)
        except Exception as e:
            st.error(f"Error in trade log analysis: {e}")

with main_tabs[5]:
    st.subheader("📑 Contract Clause Analysis")
    if contract_file:
        try:
            contract_file.seek(0)
            contract_text = contract_file.read().decode("utf-8")
            prompt = f"""Pinpoint key commercial clauses and risks in the following energy PPA contract. Mention payment rules, penalty risks, and any trade constraints:

{contract_text}"""
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=450,
                temperature=0.3,
            )
            st.success(resp.choices[0].message.content)
        except Exception as e:
            st.error(f"Error analyzing contract: {e}")

with main_tabs[6]:
    st.subheader("📘 Trading Strategy")
    if all([forecast_file, actual_file, trade_file, contract_file]):
        try:
            forecast_file.seek(0)
            actual_file.seek(0)
            trade_file.seek(0)
            contract_file.seek(0)
            forecast_df = pd.read_csv(forecast_file).head(10)
            actual_df = pd.read_csv(actual_file).head(10)
            trade_df = pd.read_csv(trade_file).head(10)
            contract_text = contract_file.read().decode("utf-8")[:1000]

            market_df = pd.DataFrame({
                'Region': ['North', 'South', 'East', 'West', 'Midlands', 'Scotland', 'Wales', 'London'],
                'Price (£/kWh)': [0.12, 0.15, 0.14, 0.13, 0.11, 0.16, 0.13, 0.17],
                'Volume (MWh)': [500, 700, 600, 450, 520, 640, 580, 720]
            })

            strategy_prompt = f"""
You are an expert energy trading assistant. Use the following real-world data to recommend a specific trading strategy:

MARKET SUMMARY (Prices in £/kWh, Volumes in MWh):
{market_df.to_string(index=False)}

FORECAST DEMAND (Expected load in kWh):
{forecast_df.to_string(index=False)}

ACTUAL GENERATION (Produced energy in kWh):
{actual_df.to_string(index=False)}

TRADE LOG (Historical trade actions and price points):
{trade_df.to_string(index=False)}

CONTRACT CLAUSES:
{contract_text}

Instructions:
- Use market clearing price and traded volume to identify price trends or congestion zones.
- Compare forecast vs actual generation to flag regions with shortages or surplus.
- Reference trade log to suggest smart timing for buy/sell decisions.
- Align strategy with contractual obligations or constraints.
- Suggest specific buy/sell actions, price points (£/kWh), and load targets (in MWh) where appropriate.
"""
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": strategy_prompt}],
                max_tokens=1000,
                temperature=0.4,
            )
            st.success(resp.choices[0].message.content)
        except Exception as e:
            st.error(f"Error generating trading strategy: {e}")

with main_tabs[7]:
    st.subheader("📉 Price/Volume Tracker")
    st.info("🔄 Fetching recent price/volume trend from National Grid ESO...")
    try:
        import requests
        url = "https://data.nationalgrideso.com/api/3/action/datastore_search?resource_id=4f6ec4a3-dc81-4c25-b01c-cd15ea52b421&limit=20"
        response = requests.get(url)
        records = response.json()["result"]["records"]
        df_tracker = pd.DataFrame(records)
        df_tracker = df_tracker.rename(columns=lambda x: x.strip())
        df_tracker['Price (£/kWh)'] = pd.to_numeric(df_tracker.iloc[:, 1], errors='coerce')
        df_tracker['Volume (MWh)'] = pd.to_numeric(df_tracker.iloc[:, 2], errors='coerce')
        st.line_chart(df_tracker[['Price (£/kWh)', 'Volume (MWh)']].dropna())

        tracker_prompt = f"""Analyze the following price and volume trends and summarize trading implications:

{df_tracker.head(10).to_string(index=False)}"""
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": tracker_prompt}],
            max_tokens=300,
            temperature=0.4,
        )
        st.info(resp.choices[0].message.content)
    except Exception as e:
        st.warning("⚠️ Could not fetch real-time data. Showing simulated trends.")
        fallback_df = pd.DataFrame({
            'Price (£/kWh)': [0.12, 0.13, 0.15, 0.14, 0.16, 0.17, 0.15, 0.14],
            'Volume (MWh)': [500, 620, 710, 640, 730, 760, 710, 690]
        })
        st.line_chart(fallback_df)
        fallback_prompt = f"""Analyze the following simulated price and volume trends and summarize trading implications:

{fallback_df.to_string(index=False)}"""
        try:
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": fallback_prompt}],
                max_tokens=300,
                temperature=0.4,
            )
            st.info(resp.choices[0].message.content)
        except:
            st.info("Unable to generate GenAI insights on fallback data.")



with main_tabs[8]:
    st.subheader("📊 Forecast Accuracy")
    try:
        if forecast_file and actual_file:
            forecast_file.seek(0)
            actual_file.seek(0)
            df_forecast = pd.read_csv(forecast_file)
            df_actual = pd.read_csv(actual_file)
            df_accuracy = pd.DataFrame()
            df_accuracy['Region'] = df_forecast.iloc[:, 1]
            df_accuracy['Forecast (MW)'] = df_forecast.iloc[:, 2]
            df_accuracy['Actual (MW)'] = df_actual.iloc[:, 2] if df_actual.shape[1] > 2 else df_actual.iloc[:, 1]
            df_accuracy['Error (%)'] = ((df_accuracy['Actual (MW)'] - df_accuracy['Forecast (MW)']) / df_accuracy['Forecast (MW)']) * 100
            st.dataframe(df_accuracy.head())

            acc_prompt = f"""Based on this forecast accuracy report, identify forecasting issues and suggest operational improvements:

{df_accuracy.head(10).to_string(index=False)}"""
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": acc_prompt}],
                max_tokens=400,
                temperature=0.4,
            )
            st.info(resp.choices[0].message.content)
        else:
            st.warning("Upload forecast and actual generation files.")
    except Exception as e:
        st.warning("⚠️ Forecast Accuracy fallback activated due to error.")
        fallback_data = pd.DataFrame({
            'Region': ['North', 'South', 'East', 'West'],
            'Forecast (MW)': [1200, 1400, 1300, 1250],
            'Actual (MW)': [1150, 1450, 1280, 1230]
        })
        fallback_data['Error (%)'] = ((fallback_data['Actual (MW)'] - fallback_data['Forecast (MW)']) / fallback_data['Forecast (MW)']) * 100
        st.dataframe(fallback_data)

with main_tabs[9]:
    st.subheader("⚠️ Risk & Clause Flags")
    if contract_file:
        contract_file.seek(0)
        contract_text = contract_file.read().decode("utf-8")
        prompt = f"""Highlight risk flags and constraint clauses in this energy PPA contract:

{contract_text}"""
        try:
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.4,
            )
            st.success(resp.choices[0].message.content)
        except Exception as e:
            st.error(f"Error detecting clause risks: {e}")
    else:
        st.warning("Upload the contract to extract risks and flags.")

with main_tabs[10]:
    st.subheader("📆 Schedule Optimizer")
    if forecast_file and trade_file:
        forecast_file.seek(0)
        trade_file.seek(0)
        forecast_df = pd.read_csv(forecast_file)
        trade_df = pd.read_csv(trade_file)
        prompt = f"""You are a smart energy trading scheduler. Based on the following forecasted demand and past trade logs, suggest an optimal trade schedule (time blocks) for buy/sell maximizing profit and reliability:

FORECAST:
{forecast_df.head(10).to_string(index=False)}

TRADE LOG:
{trade_df.head(10).to_string(index=False)}

Give recommended trade time blocks, action (Buy/Sell), and rationale."""
        try:
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.4,
            )
            st.success(resp.choices[0].message.content)
        except Exception as e:
            st.error(f"Error generating schedule optimization: {e}")
    else:
        st.warning("Please upload Forecast Demand and Trade Log CSVs.")
