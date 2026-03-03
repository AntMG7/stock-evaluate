import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Page Configuration
st.set_page_config(page_title="Advanced Quant Analyst", layout="wide")

# Custom CSS for smaller business description font
st.markdown("""
    <style>
    .small-font {
        font-size:14px !important;
        line-height: 1.4;
        color: #555;
    }
    </style>
    """, unsafe_base64=True)

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
            
            ema_50 = close_prices.ewm(span=50, adjust=False).mean()
            ema_200 = close_prices.ewm(span=200, adjust=False).mean()
            current_price = float(close_prices.iloc[-1].item())
            volatility_ann = daily_returns.std() * np.sqrt(252)
            
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
                # Using the custom CSS class for smaller font
                description = info.get('longBusinessSummary', 'No summary available.')
                st.markdown(f'<p class="small-font">{description}</p>', unsafe_base64=True)
            with col_b:
                st.metric("Market Cap", f"${info.get('marketCap', 0):,}")
                st.metric("P/E Ratio", f"{info.get('trailingPE', 'N/A')}")

            st.markdown("---")

            # --- MIDDLE SECTION: Technicals & Verdict ---
            t_col1, t_col2, t_col3, t_col4 = st.columns(4)
            t_col1.metric("Current Price", f"${current_price:.2f}")
            t_col2.metric("Ann. Volatility", f"{volatility_ann:.2%}")
            t_col3.metric("Annual Sharpe", f"{sharpe:.2f}")
            t_col4.metric("RSI (14d)", f"{int(rsi)}")

            # --- BOTTOM SECTION: Visual Analysis ---
            v_col1, v_col2 = st.columns(2)

            with v_col1:
                st.subheader("Price Action & Vertical Markers")
                fig_price, ax_price = plt.subplots(figsize=(10, 6))
                ax_price.plot(close_prices, label='Price', color='black', alpha=0.7)
                ax_price.plot(ema_50, label='50-day EMA', color='blue', linestyle='--')
                ax_price.plot(ema_200, label='200-day EMA', color='red')
                
                # ADDING VERTICAL MARKERS
                # Highlight the most recent price point
                ax_price.axvline(x=close_prices.index[-1], color='green', linestyle=':', label='Current Date')
                # Optional: Highlight 3 months ago to show recent trend shift
                three_months_ago = close_prices.index[-63] if len(close_prices) > 63 else close_prices.index[0]
                ax_price.axvline(x=three_months_ago, color='gray', linestyle='--', alpha=0.5, label='3M Baseline')
                
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
                ax_sim.plot(np.mean(sims, axis=1), color='black', label='Mean Path')
                
                # VERTICAL MARKER for forecast milestones (e.g., halfway point)
                ax_sim.axvline(x=n_days//2, color='orange', linestyle='--', label='Midpoint')
                
                ax_sim.set_ylabel("Price ($)")
                ax_sim.legend()
                st.pyplot(fig_sim)
                
                final = sims[-1, :]
                low_ci, high_ci = np.percentile(final, 2.5), np.percentile(final, 97.5)
                st.write(f"**95% Confidence Interval:** ${low_ci:.2f} - ${high_ci:.2f}")

            # --- EXPORT SECTION ---
            st.markdown("---")
            export_data = {"Ticker": ticker_symbol, "Sharpe": sharpe, "RSI": rsi, "95%_Low": low_ci, "95%_High": high_ci}
            report_df = pd.DataFrame([export_data])
            csv_export = report_df.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download Quant Report (CSV)", data=csv_export, file_name=f"{ticker_symbol}_report.csv", mime="text/csv")
