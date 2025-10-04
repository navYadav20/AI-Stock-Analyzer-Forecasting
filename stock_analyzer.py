import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from statsmodels.tsa.arima.model import ARIMA

# Import from our custom Gemini module
from gemini_utils import get_gemini_analysis, API_CONFIGURED


# data fetching & caching
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_stock_data(ticker: str, period: str = "3y"):
    """
    downloads price history and scrapes headlines from multiple sources for robustness.
    returns hist_df, info_dict, headlines_list
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        hist = ticker_obj.history(period=period)
        if hist.empty:
            return None, None, []

        # drop timezone info if present
        if hasattr(hist.index, "tz") and hist.index.tz is not None:
            hist.index = hist.index.tz_localize(None)

        info = ticker_obj.info or {}
        headlines = []
        headers = {"User-Agent": "Mozilla/5.0"}

        # yahoo fnance headlines
        try:
            yahoo_url = f"https://finance.yahoo.com/quote/{ticker}/"
            resp = requests.get(yahoo_url, headers=headers, timeout=6)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser") # scraping data

            elems = soup.select("li h3 a")
            for el in elems:
                # Prefer full text from title attribute
                txt = el.get("title") or el.get_text(strip=True)
                if txt and txt not in headlines:
                    headlines.append(txt)
        except Exception:
            headlines = []

        #google finance fallback
        if not headlines:
            try:
                ticker_for_url = ticker.split(".")[0]
                exchange = "NSE" if ticker.upper().endswith(".NS") else "NASDAQ"
                google_url = f"https://www.google.com/finance/quote/{ticker_for_url}:{exchange}"
                resp = requests.get(google_url, headers=headers, timeout=6)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")

                elems = soup.select("div.AoCdqe, div.x2pMke, div.Yfwt5")
                for el in elems:
                    txt = el.get("title") or el.get("aria-label") or el.get_text(strip=True)
                    if txt and txt not in headlines:
                        headlines.append(txt)
            except Exception:
                headlines = []

        # final fallback: use business summary from yfinance
        if not headlines and info.get("longBusinessSummary"):
            headlines = [s.strip() for s in info["longBusinessSummary"].split(".") if s.strip()][:6]

        return hist, info, headlines
    except Exception:
        return None, None, []


#  Forecasting using ARIMA ml mnodel, send data to tab 2 , forecasting
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


#  Agentic Watchdog for tab 1 . this funtion will returns a list of short findings  including a new 7 day average price
def run_agentic_watchdog(hist: pd.DataFrame, info: dict, headlines_list: list):

    findings = []

    # 7 Day Average Price
    try:
        seven_day_avg = float(hist['Close'].tail(7).mean())
        last_price = float(hist['Close'].iloc[-1])
        findings.append(f"📈 **7-Day Avg. Price:** {seven_day_avg:.2f} (Current: {last_price:.2f})")
    except Exception:
        pass

    # Price alert check
    last_price = float(hist["Close"].iloc[-1])
    window = min(30, len(hist))
    thirty_day_avg = float(hist["Close"].tail(window).mean())
    if last_price < thirty_day_avg * 0.9:
        findings.append(f"🔴 **Price Alert:** Current price ({last_price:.2f}) is >10% below the {window}-day average.")
    elif last_price > thirty_day_avg * 1.1:
        findings.append(f"🟢 **Price Alert:** Current price ({last_price:.2f}) is >10% above the {window}-day average.")

    # SMA cross check
    hist_copy = hist.copy()
    hist_copy["SMA_50"] = hist_copy["Close"].rolling(window=50, min_periods=10).mean()
    hist_copy["SMA_200"] = hist_copy["Close"].rolling(window=200, min_periods=50).mean()
    if not pd.isna(hist_copy["SMA_50"].iloc[-1]) and not pd.isna(hist_copy["SMA_200"].iloc[-1]):
        if hist_copy["SMA_50"].iloc[-1] > hist_copy["SMA_200"].iloc[-1] and hist_copy["SMA_50"].iloc[-5] <= hist_copy["SMA_200"].iloc[-5]:
            findings.append(" **Trend:** Recent Golden Cross detected (50-day SMA > 200-day SMA).")
        elif hist_copy["SMA_50"].iloc[-1] < hist_copy["SMA_200"].iloc[-1] and hist_copy["SMA_50"].iloc[-5] >= hist_copy["SMA_200"].iloc[-5]:
            findings.append("🔴 **Trend:** Recent Death Cross detected (50-day SMA < 200-day SMA).")

    # News sentiment from gemini
    if API_CONFIGURED and headlines_list:
        headlines_str = "\n".join(f"- {h}" for h in headlines_list[:8])  # Use first 8 for prompt
        prompt = (
            f"In one detailed sentence, summarize sentiment (Positive/Neutral/Negative) for {info.get('longName', '')} "
            f"from these headlines:\n\n{headlines_str}\n\nAnswer like: 'Sentiment: Positive — because...'.")
        gresp = get_gemini_analysis(prompt, concise=True)
        findings.append(f"📰 **News Sentiment:** {gresp}")

    return findings if findings else ["No significant signals detected."]
