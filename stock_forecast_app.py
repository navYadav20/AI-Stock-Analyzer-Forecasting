# stock_forcast_app.py
import os
import textwrap
from datetime import timedelta

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from statsmodels.tsa.arima.model import ARIMA

# -----------------------
# Page / API configuration
# -----------------------
st.set_page_config(page_title="Agentic Stock Analyzer", page_icon="🤖", layout="wide")

# Configure Gemini (google.generativeai)
API_CONFIGURED = False
GEMINI_MODEL = None
GEMINI_KEY = None

# Try to read key from Streamlit secrets first, then environment
if "GEMINI_API_KEY" in st.secrets:
    GEMINI_KEY = st.secrets["GEMINI_API_KEY"]
else:
    GEMINI_KEY = os.environ.get("GEMINI_API_KEY", None)

if GEMINI_KEY:
    try:
        genai.configure(api_key=GEMINI_KEY)
        # Use a stable model name — adjust if your account uses a different model alias
        GEMINI_MODEL = genai.GenerativeModel("gemini-2.5-flash")
        API_CONFIGURED = True
    except Exception as e:
        API_CONFIGURED = False
        st.error("Could not configure Gemini API: " + str(e))
else:
    API_CONFIGURED = False

# -----------------------
# Helper utilities
# -----------------------
def shorten_text(text: str, max_chars: int = 700) -> str:
    """Return a truncated text for preview and add ellipsis when truncated."""
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return textwrap.shorten(text, width=max_chars, placeholder=" ... (click 'Show full' to read more)")

def safe_text_from_gemini(response) -> str:
    """Try common attributes for gemini response objects. Fallback to str()."""
    if response is None:
        return ""
    # Many wrappers store text on .text or .output or similar
    possible_attrs = ["text", "content", "output", "candidates"]
    for a in possible_attrs:
        if hasattr(response, a):
            val = getattr(response, a)
            try:
                # If it's list-like, join
                if isinstance(val, (list, tuple)):
                    return " ".join(map(str, val))
                return str(val)
            except Exception:
                continue
    # fallback
    try:
        return str(response)
    except Exception:
        return ""

# -----------------------
# Gemini wrapper
# -----------------------
def get_gemini_analysis(prompt: str, concise: bool = True) -> str:
    """
    Use Gemini to generate text. If API not configured, return a message string.
    The 'concise' flag is used to tell Gemini to keep answers short.
    """
    if not API_CONFIGURED or GEMINI_MODEL is None:
        return "Gemini not configured. Set GEMINI_API_KEY in Streamlit secrets or environment."

    # Add concise instructions to prompt
    if concise:
        prompt = (
            "Be concise. Limit to ~200-300 words. Use 2-line executive summary followed by short bullet points.\n\n"
            + prompt
        )
    try:
        response = GEMINI_MODEL.generate_content(prompt)
        text = safe_text_from_gemini(response)
        # final safety trim
        return text.strip()
    except Exception as e:
        return f"Error calling Gemini: {e}"

# -----------------------
# Data fetching & caching
# -----------------------
@st.cache_data(ttl=60 * 60 * 1)  # cache for 1 hour
def get_stock_data(ticker: str, period: str = "3y"):
    """
    Download price history (yfinance) and attempt to scrape some headlines (Google Finance).
    Returns (hist_df, info_dict, headlines_list)
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        hist = ticker_obj.history(period=period)
        if hist.empty:
            return None, None, []

        # normalize index timezone if present
        if hasattr(hist.index, "tz") and hist.index.tz is not None:
            hist.index = hist.index.tz_localize(None)

        info = {}
        try:
            info = ticker_obj.info or {}
        except Exception:
            info = {}

        # Try to fetch headlines from Google Finance (best-effort)
        headlines = []
        try:
            ticker_for_url = ticker.split(".")[0]
            exchange = "NSE" if ticker.upper().endswith(".NS") else "NASDAQ"
            url = f"https://www.google.com/finance/quote/{ticker_for_url}:{exchange}"
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(url, headers=headers, timeout=6)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            # Google uses different classes; try a few selectors
            elems = soup.select("div.AoCdqe") or soup.select("div.x2pMke") or soup.select("div.Yfwt5")
            headlines = [el.get_text(strip=True) for el in elems]
            headlines = [h for h in headlines if h]
        except Exception:
            headlines = []

        # fallback: use business summary if nothing else
        if not headlines and isinstance(info, dict):
            business_summary = info.get("longBusinessSummary", "")
            if business_summary:
                # split summary into sentences
                headlines = [s.strip() for s in business_summary.split(".") if s.strip()][:6]

        return hist, info, headlines[:10]
    except Exception:
        return None, None, []

# -----------------------
# Forecasting functions
# -----------------------
def compute_arima_forecast(hist: pd.DataFrame, steps: int = 21):
    """
    Fit ARIMA(5,1,0) to the Close series and forecast `steps` future *business days*.
    Returns a DataFrame indexed by future business dates with columns: predicted, lower, upper.
    If ARIMA fit fails, returns flat forecast equal to last close.
    """
    # Use a simple key to cache in session_state rather than @st.cache to avoid non-serializable model objects
    last_price = float(hist["Close"].iloc[-1])
    last_date = hist.index[-1]

    # future business days (skip weekends)
    try:
        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=steps)
    except Exception:
        # fallback to daily if business calendar fails
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps)

    try:
        # Fit ARIMA on the Close series
        model = ARIMA(hist["Close"].astype(float), order=(5, 1, 0))
        model_fit = model.fit()
        fc = model_fit.get_forecast(steps=steps)
        pred = fc.predicted_mean
        ci = fc.conf_int(alpha=0.05)
        # ci columns might be named like ('lower Close','upper Close') — take first two cols
        lower = ci.iloc[:, 0].values if ci.shape[1] >= 1 else np.full(steps, np.nan)
        upper = ci.iloc[:, 1].values if ci.shape[1] >= 2 else np.full(steps, np.nan)

        df_fc = pd.DataFrame(
            {"predicted": np.asarray(pred).astype(float), "lower": lower.astype(float), "upper": upper.astype(float)},
            index=future_dates,
        )
        return df_fc
    except Exception:
        # fallback: repeat last price
        df_fc = pd.DataFrame(
            {"predicted": np.repeat(last_price, steps), "lower": np.full(steps, np.nan), "upper": np.full(steps, np.nan)},
            index=future_dates,
        )
        return df_fc

# -----------------------
# Agentic Watchdog
# -----------------------
def run_agentic_watchdog(hist: pd.DataFrame, info: dict, headlines_list: list, use_gemini: bool = True):
    """
    Returns a list of short findings (strings). If use_gemini is True and API configured,
    it will ask Gemini for a one-line sentiment. Otherwise it will produce a heuristic summary.
    """
    findings = []
    # Basic numeric checks
    try:
        last_price = float(hist["Close"].iloc[-1])
        window = min(30, len(hist))
        thirty_day_avg = float(hist["Close"].tail(window).mean())
        if last_price < thirty_day_avg * 0.9:
            findings.append(
                f"🔴 Price Alert: Current price ({last_price:.2f}) is >10% below the {window}-day average ({thirty_day_avg:.2f})."
            )
        elif last_price > thirty_day_avg * 1.1:
            findings.append(
                f"🟢 Price Alert: Current price ({last_price:.2f}) is >10% above the {window}-day average ({thirty_day_avg:.2f})."
            )
    except Exception:
        pass

    # SMA cross check
    try:
        hist = hist.copy()
        hist["SMA_50"] = hist["Close"].rolling(window=50, min_periods=10).mean()
        hist["SMA_200"] = hist["Close"].rolling(window=200, min_periods=50).mean()
        if not pd.isna(hist["SMA_50"].iloc[-1]) and not pd.isna(hist["SMA_200"].iloc[-1]):
            if hist["SMA_50"].iloc[-1] > hist["SMA_200"].iloc[-1] and hist["SMA_50"].iloc[-5] <= hist["SMA_200"].iloc[-5]:
                findings.append("🟢 Trend: Recent Golden Cross detected (50-day SMA crossed above 200-day SMA).")
            elif hist["SMA_50"].iloc[-1] < hist["SMA_200"].iloc[-1] and hist["SMA_50"].iloc[-5] >= hist["SMA_200"].iloc[-5]:
                findings.append("🔴 Trend: Recent Death Cross detected (50-day SMA crossed below 200-day SMA).")
    except Exception:
        pass

    # News sentiment: one-line summary via Gemini (concise) or heuristic fallback
    headlines_short = "\n".join(f"- {h}" for h in headlines_list[:8]) if headlines_list else "No headlines available."
    if use_gemini and API_CONFIGURED:
        prompt = (
            "In one short sentence, summarize the overall sentiment (Positive / Neutral / Negative) of these headlines "
            f"for {info.get('longName', '')}:\n\n{headlines_short}\n\nAnswer in one line like: 'Sentiment: Positive — short reason.'"
        )
        gresp = get_gemini_analysis(prompt, concise=True)
        if gresp:
            findings.append(f"📰 News Sentiment: {shorten_text(gresp, max_chars=220)}")
    else:
        # heuristic
        text = " ".join(headlines_list).lower()
        pos = sum(text.count(w) for w in ["gain", "raise", "upgrade", "profit", "beats", "positive", "growth", "record"])
        neg = sum(text.count(w) for w in ["drop", "loss", "downgrade", "decline", "missed", "negative", "weak", "fall"])
        if pos > neg:
            findings.append("📰 News Sentiment (heuristic): Mostly positive.")
        elif neg > pos:
            findings.append("📰 News Sentiment (heuristic): Mostly negative.")
        else:
            findings.append("📰 News Sentiment (heuristic): Neutral / Mixed.")

    return findings

# -----------------------
# Streamlit UI layout
# -----------------------
st.title("🤖 Agentic Stock Analyzer")
st.markdown("An intelligent tool combining AI analysis, agentic monitoring, and statistical forecasting.")

# Sidebar form to avoid unwanted reruns when interacting with other widgets
with st.sidebar.form(key="ticker_form"):
    st.header("Controls & Suggestions")
    ticker_in = st.text_input("Enter Stock Ticker (e.g., RELIANCE.NS, AAPL)", "RELIANCE.NS").upper()
    submit = st.form_submit_button("Run Full Analysis")

    st.markdown(
        """
        **Tips**
        - For Indian stocks on NSE, add `.NS` (e.g., RELIANCE.NS)
        - For US stocks use the symbol directly (e.g., AAPL)
        """
    )

# When the user submits, fetch data and store to session_state
if submit and ticker_in:
    with st.spinner(f"Downloading data for {ticker_in}..."):
        hist_data, info, headlines = get_stock_data(ticker_in, period="3y")
    if hist_data is None:
        st.sidebar.error(f"Could not retrieve data for '{ticker_in}'. Please check the symbol.")
    else:
        # Persist results across reruns in session_state
        st.session_state["hist_data"] = hist_data
        st.session_state["info"] = info
        st.session_state["headlines"] = headlines
        st.session_state["last_ticker"] = ticker_in
        # clear forecast caches for this ticker
        st.session_state.pop(f"forecast_{ticker_in}_21", None)
        st.session_state.pop(f"forecast_{ticker_in}_63", None)
        st.session_state.pop(f"forecast_{ticker_in}_126", None)
        st.success(f"Data loaded for {ticker_in}.")

# If we already had a ticker in session_state, use it
if "hist_data" in st.session_state:
    hist_data = st.session_state["hist_data"]
    info = st.session_state.get("info", {})
    headlines = st.session_state.get("headlines", [])
    ticker = st.session_state.get("last_ticker", "")
else:
    hist_data, info, headlines = None, None, []
    ticker = None

# Main content
if hist_data is not None and info is not None:
    company_name = info.get("longName", ticker)
    st.header(f"{company_name} Dashboard ({ticker})")

    tab1, tab2, tab3 = st.tabs(["🤖 AI Agent Watchdog", "📈 Forecasting", "💡 Fundamental Analysis"])

    # ---- Tab 1: Agentic Watchdog ----
    with tab1:
        st.subheader("Agent's On-Demand Report")
        st.caption("Run the watchdog to get quick signals and a short news sentiment summary.")

        if st.button("Run Agent Watchdog", key=f"watchdog_{ticker}"):
            with st.spinner("Running agent checks..."):
                findings = run_agentic_watchdog(hist_data, info, headlines, use_gemini=True)
                st.session_state[f"agent_findings_{ticker}"] = findings

        findings = st.session_state.get(f"agent_findings_{ticker}", None)
        if findings:
            for f in findings:
                st.markdown(f)
            st.success("Watchdog analysis complete.")
        else:
            st.info("Click 'Run Agent Watchdog' to perform checks (price alerts, SMA cross, news sentiment).")

        # Show latest headlines in small area
        if headlines:
            with st.expander("Recent headlines (click to expand)"):
                for h in headlines:
                    st.write("• " + h)

    # ---- Tab 2: Forecasting ----
    with tab2:
        st.subheader("Short-Term Price Projection (ARIMA Model)")
        st.caption("Select horizon and generate a forecast. Forecasts are cached per ticker + horizon for faster repeat views.")

        # default forecast period state (per-ticker)
        default_period_key = f"forecast_period_{ticker}"
        if default_period_key not in st.session_state:
            st.session_state[default_period_key] = "3 Months"

        forecast_period = st.select_slider(
            "Select forecast horizon:",
            options=["1 Month", "3 Months", "6 Months"],
            value=st.session_state[default_period_key],
            key=default_period_key,
        )
        # Map to business days (approx)
        days_map = {"1 Month": 21, "3 Months": 63, "6 Months": 126}
        days_to_forecast = days_map[forecast_period]

        # Generate forecast button
        if st.button("Generate Forecast", key=f"gen_forecast_{ticker}"):
            with st.spinner(f"Training ARIMA and forecasting next {forecast_period}..."):
                fc_df = compute_arima_forecast(hist_data, steps=days_to_forecast)
                # store in session_state keyed by ticker + days for caching
                st.session_state[f"forecast_{ticker}_{days_to_forecast}"] = fc_df
                st.success("Forecast generated.")

        # If we have cached forecast, display it
        fc_cached = st.session_state.get(f"forecast_{ticker}_{days_to_forecast}", None)
        if fc_cached is not None:
            # Plot last 180 business days of history + forecast
            try:
                hist_plot = hist_data["Close"].last("180D")
            except Exception:
                hist_plot = hist_data["Close"].tail(180)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist_plot.index, y=hist_plot.values, mode="lines", name="Recent History"))
            fig.add_trace(
                go.Scatter(
                    x=fc_cached.index,
                    y=fc_cached["predicted"],
                    mode="lines",
                    name="Forecast",
                    line=dict(dash="dash"),
                )
            )
            # Add confidence band if available
            if "lower" in fc_cached.columns and "upper" in fc_cached.columns:
                fig.add_trace(
                    go.Scatter(
                        x=list(fc_cached.index) + list(fc_cached.index[::-1]),
                        y=list(fc_cached["upper"].values) + list(fc_cached["lower"].values[::-1]),
                        fill="toself",
                        fillcolor="rgba(0,100,80,0.1)",
                        line=dict(color="rgba(255,255,255,0)"),
                        hoverinfo="skip",
                        showlegend=False,
                        name="Confidence interval",
                    )
                )
            fig.update_layout(title=f"Price Forecast for next {forecast_period}", yaxis_title="Price", height=520)
            st.plotly_chart(fig, use_container_width=True)

            # Show forecast table preview and download button
            preview_df = fc_cached.reset_index().rename(columns={"index": "date"})
            preview_df["date"] = preview_df["date"].dt.strftime("%Y-%m-%d")
            st.subheader("Forecast (preview)")
            st.dataframe(preview_df.head(15))

            csv = preview_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download forecast CSV",
                data=csv,
                file_name=f"{ticker}_forecast_{days_to_forecast}d.csv",
                mime="text/csv",
            )
        else:
            st.info("Generate forecast to see chart and data. Forecasts are cached while this session is active.")

    # ---- Tab 3: Fundamental Analysis ----
    with tab3:
        st.subheader("Long-Term Outlook (Gemini AI)")
        st.caption("Generate a short, professional long-term outlook (1-3 years). Full text is available in an expander.")

        long_key = f"long_outlook_{ticker}"
        if st.button("Generate Long-Term Outlook", key=f"gen_long_{ticker}"):
            with st.spinner("Asking Gemini for a concise long-term outlook..."):
                prompt = f"""
                Act as a senior financial analyst. Provide a concise 1-3 year investment outlook for {info.get('longName','') } ({ticker}).
                Follow this structure:
                - Two-line executive summary.
                - Short bullet points (3-6) for Pros.
                - Short bullet points (3-6) for Cons.
                - Keep total output under ~300 words and use short bullets.
                """
                long_text = get_gemini_analysis(prompt, concise=True) if API_CONFIGURED else "Gemini not configured; cannot generate long-term outlook."
                st.session_state[long_key] = long_text

        long_text = st.session_state.get(long_key, None)
        if long_text:
            # show a short preview and allow expansion
            short_preview = shorten_text(long_text, max_chars=600)
            st.markdown(short_preview)
            with st.expander("Show full long-term outlook"):
                st.markdown(long_text)
            st.info("This is AI-generated and for informational purposes only. Not financial advice.")
        else:
            st.info("Click 'Generate Long-Term Outlook' to request a concise summary from Gemini (or show fallback if not configured).")

else:
    # No data loaded yet
    st.info("Enter a stock ticker in the sidebar and click 'Run Full Analysis' to begin.")
    if not API_CONFIGURED:
        st.warning("Gemini API not configured. Long-term AI summaries will be unavailable. Add GEMINI_API_KEY to secrets.toml or environment.")
