import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from statsmodels.tsa.arima.model import ARIMA

# Import from our custom Gemini module
from gemini_utils import get_gemini_analysis, API_CONFIGURED, shorten_text


# --- Data Fetching & Caching ---
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_stock_data(ticker: str, period: str = "3y"):
    """
    Downloads price history and scrapes headlines.
    Returns (hist_df, info_dict, headlines_list)
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        hist = ticker_obj.history(period=period)
        if hist.empty:
            return None, None, []

        if hasattr(hist.index, "tz") and hist.index.tz is not None:
            hist.index = hist.index.tz_localize(None)

        info = ticker_obj.info or {}

        headlines = []
        try:
            ticker_for_url = ticker.split(".")[0]
            exchange = "NSE" if ticker.upper().endswith(".NS") else "NASDAQ"
            url = f"https://www.google.com/finance/quote/{ticker_for_url}:{exchange}"
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(url, headers=headers, timeout=6)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            elems = soup.select("div.AoCdqe, div.x2pMke, div.Yfwt5")
            headlines = [el.get_text(strip=True) for el in elems if el.get_text(strip=True)]
        except Exception:
            pass  # Silently fail on scrape error

        if not headlines and info.get("longBusinessSummary"):
            headlines = [s.strip() for s in info["longBusinessSummary"].split(".") if s.strip()][:6]

        return hist, info, headlines[:10]
    except Exception:
        return None, None, []


# --- Forecasting ---
def compute_arima_forecast(hist: pd.DataFrame, steps: int = 21):
    """
    Fits an ARIMA model and forecasts future business days.
    Returns a DataFrame with columns: predicted, lower, upper.
    """
    last_price = float(hist["Close"].iloc[-1])
    last_date = hist.index[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=steps)

    try:
        model = ARIMA(hist["Close"].astype(float), order=(5, 1, 0))
        model_fit = model.fit()
        fc = model_fit.get_forecast(steps=steps)
        ci = fc.conf_int(alpha=0.05)
        return pd.DataFrame({
            "predicted": fc.predicted_mean.values,
            "lower": ci.iloc[:, 0].values,
            "upper": ci.iloc[:, 1].values
        }, index=future_dates)
    except Exception:
        return pd.DataFrame({
            "predicted": np.repeat(last_price, steps),
            "lower": np.nan, "upper": np.nan
        }, index=future_dates)


# --- Agentic Watchdog ---
def run_agentic_watchdog(hist: pd.DataFrame, info: dict, headlines_list: list):
    """
    Returns a list of short findings (strings).
    """
    findings = []
    # Price alert check
    last_price = float(hist["Close"].iloc[-1])
    window = min(30, len(hist))
    thirty_day_avg = float(hist["Close"].tail(window).mean())
    if last_price < thirty_day_avg * 0.9:
        findings.append(f"🔴 Price Alert: Current price ({last_price:.2f}) is >10% below the {window}-day average.")
    elif last_price > thirty_day_avg * 1.1:
        findings.append(f"🟢 Price Alert: Current price ({last_price:.2f}) is >10% above the {window}-day average.")

    # SMA cross check
    hist = hist.copy()
    hist["SMA_50"] = hist["Close"].rolling(window=50, min_periods=10).mean()
    hist["SMA_200"] = hist["Close"].rolling(window=200, min_periods=50).mean()
    if not pd.isna(hist["SMA_50"].iloc[-1]) and not pd.isna(hist["SMA_200"].iloc[-1]):
        if hist["SMA_50"].iloc[-1] > hist["SMA_200"].iloc[-1] and hist["SMA_50"].iloc[-5] <= hist["SMA_200"].iloc[-5]:
            findings.append("🟢 Trend: Recent Golden Cross detected (50-day SMA > 200-day SMA).")
        elif hist["SMA_50"].iloc[-1] < hist["SMA_200"].iloc[-1] and hist["SMA_50"].iloc[-5] >= hist["SMA_200"].iloc[-5]:
            findings.append("🔴 Trend: Recent Death Cross detected (50-day SMA < 200-day SMA).")

    # News sentiment from Gemini
    if API_CONFIGURED:
        headlines_str = "\n".join(f"- {h}" for h in headlines_list)
        prompt = (f"In one sentence, summarize sentiment (Positive/Neutral/Negative) for {info.get('longName', '')} "
                  f"from these headlines:\n\n{headlines_str}\n\nAnswer like: 'Sentiment: Positive — because...'.")
        gresp = get_gemini_analysis(prompt, concise=True)
        findings.append(f"📰 News Sentiment: {shorten_text(gresp, max_chars=220)}")

    return findings if findings else ["No significant signals detected."]
