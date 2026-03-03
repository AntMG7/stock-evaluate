import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Page Configuration
st.set_page_config(page_title="Advanced Quant Analyst", layout="wide")
st.title("🔬 Deep-Dive Financial Analysis")
st.markdown("---")

# 2. Sidebar Input
st.sidebar.header("Analysis Parameters")
ticker_symbol = st.sidebar.text_input("Enter Ticker Symbol:", value="NVDA").upper()
n_days = st.sidebar.slider("Forecast Horizon (Days)", 5, 90, 30)

if ticker_symbol:
    with st.spinner(f'Analyzing {ticker_symbol}...'):
        ticker_obj = yf.Ticker(ticker_symbol)
        data = ticker_obj.history(period="1y")
        info = ticker_obj.info
        
        if not data.empty:
            # --- CALCULATIONS ---
            close_prices = data['Close']
            daily_returns = close_prices.pct_change().dropna()
            
            # Trend & Volatility
            ema_50 = close_prices.ewm(span=50, adjust=False).mean()
            ema_200 = close_prices.ewm(span=200, adjust=False).mean()
            current_price = float(close_prices.iloc[-1].item())
            volatility_ann = daily_returns.std() * np.sqrt(252)
            
            # Sharpe & RSI
            sharpe = float(((daily_returns.mean() / daily_returns.std()) * np.sqrt(252)).item())
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rsi = float((100 - (100 / (1 + (gain / loss)))).iloc[-1].item())

            # --- HEADER SECTION: Company Profile ---
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.header(f"{info.get('longName', ticker_symbol)}")
                st.caption(f"Sector: {info.get('sector', 'N/A')} | Industry: {info.get('industry', 'N/A')}")
                st.write(info.get('longBusinessSummary', 'No summary available.'))
            with col_b:
                st.metric("Market Cap", f"${info.get('marketCap', 0):,}")
                st.metric("P/E Ratio (Trailing)", f"{info.get('trailingPE', 'N/A')}")

            st.markdown("---")

            # --- MIDDLE SECTION: Technicals & Verdict ---
            t_col1, t_col2, t_col3, t_col4 = st.columns(4)
            t_col1.metric("Current Price", f"${current_price:.2f}")
            t_col2.metric("Ann. Volatility", f"{volatility_ann:.2%}")
            t_col3.metric("Annual Sharpe", f"{sharpe:.2f}")
            t_col4.metric("RSI (14d)", f"{int(rsi)}")

            # Verdict Logic
            st.subheader("Quantitative Verdict")
            if current_price > ema_200.iloc[-1] and sharpe > 1.0 and 50 < rsi < 70:
                st.success("✅ **GOOD PURCHASE**: Strong structural trend with high risk-adjusted efficiency.")
            else:
                st.warning("⚠️ **CAUTION**: One or more statistical thresholds (Trend, Sharpe, or RSI) are not met.")

            # --- BOTTOM SECTION: Visual Analysis ---
            v_col1, v_col2 = st.columns(2)

            with v_col1:
                st.subheader("Price Action & Moving Averages")
                fig_price, ax_price = plt.subplots(figsize=(10, 6))
                ax_price.plot(close_prices, label='Price', color='black', linewidth=1)
                ax_price.plot(ema_50, label='50-day EMA', color='blue', linestyle='--')
                ax_price.plot(ema_200, label='200-day EMA', color='red')
                ax_price.set_ylabel("USD ($)")
                ax_price.legend()
                st.pyplot(fig_price)

            with v_col2:
                st.subheader(f"Monte Carlo: {n_days}-Day Outlook")
                mu, sigma = daily_returns.mean(), daily_returns.std()
                sims = np.zeros((n_days, 1000))
                sims[0] = current_price
                for t in range(1, n_days):
                    sims[t] = sims[t-1] * np.exp((mu - 0.5 * sigma**2) + sigma * np.random.standard_normal(1000))
                
                fig_sim, ax_sim = plt.subplots(figsize=(10, 6))
                ax_sim.plot(sims, color='royalblue', alpha=0.02)
                ax_sim.plot(np.mean(sims, axis=1), color='black', label='Mean Projection')
                ax_sim.set_ylabel("Price ($)")
                ax_sim.legend()
                st.pyplot(fig_sim)
                
                final = sims[-1, :]
                st.write(f"**95% Confidence Interval:** ${np.percentile(final, 2.5):.2f} - ${np.percentile(final, 97.5):.2f}")
        else:
            st.error("Ticker data not found.")
