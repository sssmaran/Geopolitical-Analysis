

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, classification_report, accuracy_score, confusion_matrix, r2_score
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, train_test_split
from sklearn.compose import TransformedTargetRegressor
from imblearn.over_sampling import SMOTE
from ta.momentum import RSIIndicator
from ta.trend import MACD
import streamlit as st

def show_predictive_analysis():
    st.title("\U0001F4CA Predictive Economic Forecasts")
    st.markdown("""
    We present **forecasted trends** in **currency exchange rates** and **stock market direction** for India, China, Mexico, and the U.S. under the influence of tariff-related sentiment.
    """)

    country = st.selectbox("Select Country", ["China", "Mexico", "India"])

    if country == "India":
        st.header("\U0001F1EE\U0001F1F3 INR/USD Exchange Rate Prediction")
        df = pd.read_csv("data/merged_exchange_tariff_data.csv")
        df['Date'] = pd.to_datetime(df['Date'])
        df['INR_USD'] = pd.to_numeric(df['INR_USD'], errors='coerce')
        df = df.sort_values("Date")

        df['INR_USD_Lag1'] = df['INR_USD'].shift(1)
        df['INR_USD_Lag3'] = df['INR_USD'].shift(3)
        df['INR_USD_MA7'] = df['INR_USD'].rolling(window=7).mean()
        df['Tariff_Sentiment_MA3'] = df['Tariff_Sentiment_Score'].rolling(3).mean()
        df['Tariff_Article_Count_MA3'] = df['Tariff_Article_Count'].rolling(3).mean()
        df['Return'] = df['INR_USD'].pct_change()
        df['Volatility'] = df['Return'].rolling(3).std()
        df['RSI'] = RSIIndicator(close=df['INR_USD'], window=14).rsi()
        macd = MACD(close=df['INR_USD'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['INR_USD_Log'] = np.log(df['INR_USD'])
        df = df.dropna().reset_index(drop=True)

        features = [
            'INR_USD_Lag1', 'INR_USD_Lag3', 'INR_USD_MA7',
            'Tariff_Sentiment_Score', 'Tariff_Sentiment_MA3',
            'Tariff_Article_Count', 'Tariff_Article_Count_MA3',
            'Return', 'Volatility', 'RSI', 'MACD', 'MACD_signal']

        X = df[features]
        y = df['INR_USD_Log']
        dates = df['Date']
        split_index = int(len(df) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        dates_test = dates[split_index:]

        model = XGBRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        model.fit(X_train, y_train)

        y_pred_log = model.predict(X_test)
        y_pred = np.exp(y_pred_log)
        y_actual = np.exp(y_test)

        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        r2 = r2_score(y_actual, y_pred)
        st.write(f"✅ RMSE: {rmse:.4f}  |  ✅ R²: {r2:.4f}")

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(dates_test, y_actual, label='Actual INR/USD', linewidth=2)
        ax.plot(dates_test, y_pred, label='Predicted INR/USD', linewidth=2)
        ax.set_title("Actual vs Predicted INR/USD Exchange Rate (Enhanced Model)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Exchange Rate")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    elif country == "China":
        st.header("CNY/USD Exchange Rate Prediction")
        china_news = pd.read_csv("data/china_news_with_sentiment.csv")
        china_fx = pd.read_csv("data/CNY_USD_exchange.csv")
        china_fx['Date'] = pd.to_datetime(china_fx['Date']).dt.date
        china_fx['CNY_USD'] = pd.to_numeric(china_fx['Close'], errors='coerce')
        china_news['text_lower'] = china_news['text'].str.lower()
        china_news['publish_date'] = pd.to_datetime(china_news['publish_date']).dt.date
        tariff_news = china_news[china_news['text_lower'].str.contains('|'.join(['tariff', 'duties', 'import tax', 'trade war', 'GSP', 'section 301', 'levies', 'sanctions']), na=False)].copy()
        agg = tariff_news.groupby('publish_date').agg(Tariff_Sentiment_Score=('sentiment_score', 'mean'), Tariff_Article_Count=('text', 'count')).reset_index().rename(columns={'publish_date': 'Date'})
        merged = pd.merge(china_fx, agg, on='Date', how='left').fillna(0)
        merged['Date'] = pd.to_datetime(merged['Date'], errors='coerce')
        merged = merged.dropna(subset=['Date']).sort_values("Date")
        merged['CNY_USD_Lag1'] = merged['CNY_USD'].shift(1)
        merged['CNY_USD_Lag3'] = merged['CNY_USD'].shift(3)
        merged['CNY_USD_MA7'] = merged['CNY_USD'].rolling(7).mean()
        merged['Sentiment_MA3'] = merged['Tariff_Sentiment_Score'].rolling(3).mean()
        merged['Article_Count_MA3'] = merged['Tariff_Article_Count'].rolling(3).mean()
        merged = merged.dropna().reset_index(drop=True)

        X = merged[[
            'CNY_USD_Lag1', 'CNY_USD_Lag3', 'CNY_USD_MA7',
            'Tariff_Sentiment_Score', 'Sentiment_MA3',
            'Tariff_Article_Count', 'Article_Count_MA3']]
        y = merged['CNY_USD']
        dates = pd.to_datetime(merged['Date'])
        split = int(len(merged) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        dates_test = dates[split:]

        model = XGBRegressor(n_estimators=30, learning_rate=0.1, max_depth=3, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        st.write(f"✅ RMSE: {rmse:.4f}  |  ✅ R²: {r2:.4f}")

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(dates_test, y_test, label='Actual CNY/USD', linewidth=2)
        ax.plot(dates_test, y_pred, label='Predicted CNY/USD', linewidth=2)
        ax.set_title("Actual vs Predicted CNY/USD Exchange Rate")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    elif country == "Mexico":
        st.header("MXN/USD Exchange Rate Prediction")
        fx = pd.read_csv("data/MXN_USD_exchange.csv", parse_dates=["Date"])
        fx['Date'] = fx['Date'].dt.date
        fx['exchange_rate'] = pd.to_numeric(fx['Close'], errors='coerce')
        fx = fx[['Date', 'exchange_rate']].dropna().reset_index(drop=True)
        news = pd.read_csv("data/mexico_news_with_sentiment.csv", parse_dates=["publish_date"])
        news['text_lower'] = news['text'].str.lower()
        tariff = news[news['text_lower'].str.contains('|'.join(["tariff", "duties", "import tax", "trade war", "levies", "sanctions"]), na=False)].copy()
        tariff['Date'] = tariff['publish_date'].dt.date
        agg = tariff.groupby('Date').agg(Tariff_Sentiment=('sentiment_score', 'mean'), Tariff_Count=('text', 'count')).reset_index()
        df = fx.merge(agg, on='Date', how='left').fillna({'Tariff_Sentiment': 0, 'Tariff_Count': 0})
        df = df.sort_values('Date').reset_index(drop=True)
        df['lag1'] = df['exchange_rate'].shift(1)
        df['lag3'] = df['exchange_rate'].shift(3)
        df['ma7'] = df['exchange_rate'].rolling(7).mean()
        df['sent_ma3'] = df['Tariff_Sentiment'].rolling(3).mean()
        df['count_ma3'] = df['Tariff_Count'].rolling(3).mean()
        df = df.dropna().reset_index(drop=True)

        X = df[['lag1', 'lag3', 'ma7', 'Tariff_Sentiment', 'sent_ma3', 'Tariff_Count', 'count_ma3']]
        y = df['exchange_rate']
        dates = pd.to_datetime(df['Date'])
        cut = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:cut], X.iloc[cut:]
        y_train, y_test = y.iloc[:cut], y.iloc[cut:]
        dates_test = dates.iloc[cut:]

        xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8, random_state=42, objective='reg:squarederror')
        model = TransformedTargetRegressor(regressor=xgb, func=np.log1p, inverse_func=np.expm1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        st.write(f"✅ RMSE: {rmse:.4f}  |  ✅ R²: {r2:.4f}")

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(dates_test, y_test, label='Actual', linewidth=2)
        ax.plot(dates_test, y_pred, label='Predicted', linewidth=2)
        ax.set_title("Actual vs Predicted MXN/USD (Log-Transformed XGB)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

 