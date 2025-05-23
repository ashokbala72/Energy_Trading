from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import pandas as pd
import os
from openai import OpenAI

st.set_page_config(page_title="⚡ GenAI Energy Trading Assistant", layout="wide")
st.title("⚡ GenAI Energy Trading Assistant - Energy & Utilities Focus")

# Sidebar file uploads
st.sidebar.header("📂 Upload Data Files")
market_file = st.sidebar.file_uploader("Market Summary (CSV)", type="csv")
forecast_file = st.sidebar.file_uploader("Forecast Demand & Generation (CSV)", type="csv")
actual_file = st.sidebar.file_uploader("Actual Generation (CSV)", type="csv")
reg_file = st.sidebar.file_uploader("Regulatory Bulletins (CSV)", type="csv")
trade_file = st.sidebar.file_uploader("Trade Log (CSV)", type="csv")

openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)

# Section 1: Market Summary
if market_file:
    st.subheader("📈 Market Summary")
    df_market = pd.read_csv(market_file)
    st.dataframe(df_market, use_container_width=True)

    if st.button("🧠 Generate GenAI Market Summary"):
        try:
            preview = df_market.head(10).to_markdown(index=False)
            prompt = f"""
            You are an expert energy analyst.
            Based on the following market data from IEX or PXIL, summarize today's trading trends for energy & utility sector:

            {preview}

            Include:
            - Overall market clearing price (MCP) trends
            - Congestion or UI price signals
            - Volume spikes or drops
            - Relevant zone-wise highlights
            """
            with st.spinner("Calling OpenAI to summarize market trends..."):
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}]
                )
                summary = response.choices[0].message.content
                st.text_area("📝 GenAI Market Summary", value=summary, height=300)
        except Exception as e:
            st.error(f"Error generating summary: {e}")

# Section 2: Forecast vs Actual
if forecast_file and actual_file:
    st.subheader("📊 Forecast vs Actual Generation")
    df_forecast = pd.read_csv(forecast_file)
    df_actual = pd.read_csv(actual_file)
    df_merged = df_forecast.merge(df_actual, on=["Date", "Region"], how="outer")
    st.dataframe(df_merged, use_container_width=True)

    if st.button("🔍 Analyze Deviations with GenAI"):
        try:
            preview_rows = df_merged.head(10).to_markdown(index=False)
            deviation_prompt = f"""
            You are a power system analyst. Based on the forecast vs actual generation data below:

            {preview_rows}

            Identify notable mismatches between forecasted and actual generation across regions.
            Suggest likely causes (e.g., weather uncertainty, transmission limits, curtailment).
            Recommend actions to improve forecasting or reduce deviation charges.
            """
            with st.spinner("Analyzing forecast deviations using GenAI..."):
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": deviation_prompt}]
                )
                deviation_analysis = response.choices[0].message.content
                st.text_area("🧠 GenAI Deviation Analysis", value=deviation_analysis, height=300)
        except Exception as e:
            st.error(f"Error generating deviation analysis: {e}")

# Section 3: Regulatory Summary
if reg_file:
    st.subheader("📜 Regulatory Bulletin Summary")
    df_reg = pd.read_csv(reg_file)
    st.dataframe(df_reg, use_container_width=True)

    if st.button("📢 Summarize Regulatory Insights"):
        try:
            sample_bulletins = "\n".join(df_reg["Bulletin"].head(5).tolist())
            reg_prompt = f"""
            You are an energy regulation advisor. Summarize these latest regulatory bulletins and explain their impact on trading decisions:
            {sample_bulletins}
            """
            with st.spinner("Analyzing regulations using OpenAI..."):
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": reg_prompt}]
                )
                reg_summary = response.choices[0].message.content
                st.text_area("🔍 GenAI Regulatory Insight", value=reg_summary, height=250)
        except Exception as e:
            st.error(f"Error generating regulatory summary: {e}")

# Section 4: Trade History
if trade_file:
    st.subheader("🔁 Trade History Log")
    df_trades = pd.read_csv(trade_file)
    st.dataframe(df_trades, use_container_width=True)

    if st.button("📊 Analyze Trading Performance with GenAI"):
        try:
            preview_trades = df_trades.head(10).to_markdown(index=False)
            trade_prompt = f"""
            You are an energy trading advisor. Analyze the following historical trades:

            {preview_trades}

            Based on this sample, identify typical buy and sell price ranges, volumes, and regions.
            Recommend what price bands could be considered favorable for upcoming trades in each region.
            Highlight any missed opportunities or patterns that should be repeated or avoided.
            """
            with st.spinner("Analyzing trading data using GenAI..."):
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": trade_prompt}]
                )
                trade_summary = response.choices[0].message.content
                st.text_area("📈 GenAI Trading Insight", value=trade_summary, height=300)
        except Exception as e:
            st.error(f"Error generating trade analysis: {e}")

st.markdown("---")
# Section 5: Trading Strategy Suggestion
if forecast_file and market_file:
    st.subheader("🧠 Trading Strategy Generator")
    try:
        if df_forecast.empty or df_market.empty:
            st.warning("⚠️ One of the uploaded files is empty or has no valid data.")
        else:
            if st.button("📌 Generate Trading Strategy with GenAI"):
                preview_forecast = df_forecast.head(6).to_markdown(index=False)
                preview_market = df_market.head(6).to_markdown(index=False)
                strategy_prompt = f"""
                You are an AI energy trading strategist. Based on the following inputs:

                🔹 Forecasted Demand and Generation:
                {preview_forecast}

                🔸 Current Market Prices and Conditions:
                {preview_market}

                Suggest trading actions for the next day by region. Include:
                - Whether to buy/sell and how much
                - Preferred price bands
                - Risk or constraints (e.g., congestion, regulation, forecast uncertainty)
                - Any scheduling advice
                """
                with st.spinner("Generating trading strategy via OpenAI..."):
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": strategy_prompt}]
                    )
                    strategy_output = response.choices[0].message.content
                    st.text_area("🚀 GenAI Trading Strategy", value=strategy_output, height=350)
    except Exception as e:
        st.error(f"Error generating trading strategy: {e}")

st.markdown("---")
# Section 6: Contract Clause Analysis
st.subheader("📄 Contract Clause Analysis")
contract_file = st.sidebar.file_uploader("Upload Trading Contract / PPA (TXT only)", type="txt")

if contract_file:
    try:
        contract_text = contract_file.read().decode("utf-8")
        st.text_area("📘 Contract Preview", value=contract_text[:1000], height=200)

        if st.button("🔍 Analyze Contract Clauses with GenAI"):
            clause_prompt = f"""
            You are a legal and energy trading expert. Read the following power purchase agreement (PPA) or trading contract:

            {contract_text[:3000]}

            Summarize key clauses such as:
            - Contract duration
            - Termination rules
            - Payment terms
            - Scheduling obligations
            - Penalty or curtailment clauses

            Provide a readable summary.
            """
            with st.spinner("Analyzing contract using GenAI..."):
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": clause_prompt}]
                )
                clause_summary = response.choices[0].message.content
                st.text_area("📑 GenAI Contract Summary", value=clause_summary, height=350)
    except Exception as e:
        st.error(f"Error analyzing contract file: {e}")

st.markdown("---")
st.info("This assistant now supports market analysis, deviations, trading strategies, contract summaries, and regulatory insights.")
