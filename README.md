# AI-Stock-Analyzer-Forecasting
An intelligent stock analysis dashboard powered by Python, Streamlit, and the Gemini API. This tool uses an AI agent for real-time signal monitoring, ARIMA for statistical forecasting, and AI-driven analysis for long-term outlooks. Supports both NASDAQ and NSE stocks to provide clear, actionable insights.


[View Live Demo](https://ai-stock-analyzer-forecasting-lq3u6u6yvpss3hdbv7zxhh.streamlit.app/) 👈


## `About The Project`
This project was developed to provide investors with a modern, AI-enhanced tool that goes beyond simple price charts. Traditional analysis often requires juggling multiple sources and tools. This application consolidates key analytical functions into a single, user-friendly dashboard.

- The AI Stock Analyzer fetches real-time and historical stock data, then applies a three-pronged analysis:

- An AI Agent actively monitors for important technical signals.

- A Statistical Model provides short-term price forecasts.

- A Generative AI offers a qualitative long-term outlook.

- The result is a comprehensive dashboard that supports both US (NASDAQ) and Indian (NSE) markets, empowering users to make better-informed investment decisions.

## `Core Features`
AI Agent Watchdog: Proactively scans for and reports on key market signals at the click of a button:

- 7-Day average price vs. current price.

- Price alerts for significant deviations from the 30-day average.

- Trend analysis via Golden Cross and Death Cross (50-day vs. 200-day SMA) detections.

- AI-Powered News Sentiment: Leverages the Google Gemini API to analyze recent headlines and provide a concise, detailed sentiment summary with reasoning.

- Statistical Forecasting: Utilizes an ARIMA ML model to generate short-term price projections for the next 1, 3, or 6 months, complete with confidence intervals visualized on an interactive chart.

- Long-Term Fundamental Outlook: Employs the Gemini API to act as a financial analyst, generating a professional, qualitative summary of a stock's 1-3 year outlook, structured into pros and cons.

- Interactive Dashboard: Built with Streamlit for a clean, responsive, and easy-to-use interface that works on both desktop and mobile.

## `Technology Stack`
This project was built using a modern, Python-based data science and AI stack.

- Framework: Streamlit

- Data & Analysis: Pandas, NumPy, Statsmodels

- AI & Machine Learning: ARIMA ML model and google-generativeai

- Data Sourcing: yfinance (for historical data), requests & BeautifulSoup (for news scraping)

- Plotting: Plotly

## `Local Setup and Installation`
To clone and run this project locally, you will need Python and Pip installed. Follow these steps:

1. Clone the repository:
git clone [click](https://github.com/navYadav20/AI-Stock-Analyzer-Forecasting.git)

2. Navigate to the project directory: `cd AI-Stock-Analyzer`

3. Create and activate a virtual environment

4. Install required packages: `pip install -r requirements.txt`

5. Set up your API Key ): Create a folder named `.streamlit` in the project root.
 Inside .streamlit, create a file named secrets.toml. Add your API key to this file:

6. Run the application: `streamlit run app.py`

`The application will be available at http://localhost:8501`

## `API Used`
This application relies on several external data sources and APIs:

- Google Gemini API: Used for all generative AI tasks, including news sentiment analysis and the long-term fundamental outlook.

- Yahoo Finance API: Accessed via the yfinance library to fetch historical stock prices, volumes, and company information.

- Google Finance & Yahoo Finance: Web scraped to gather the most recent news headlines for sentiment analysis.

------------------------------------------------------------------------------------
----------------------------Thank_You-----------------------------------------------
