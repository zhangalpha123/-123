import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

# 預測主函式
def predict_stock_price(df, window_size=10, future_days=5, top_n=5):
    df = df[['Close']].dropna().copy()
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    df = df.dropna().reset_index(drop=False)  # 保留日期欄位

    query_pattern = df['log_return'][-window_size:].values
    last_price = df['Close'].iloc[-1]
    last_date = df['Date'].iloc[-1] if 'Date' in df.columns else df.index[-1]

    matches = []
    for i in range(len(df) - window_size - future_days):
        candidate = df['log_return'].iloc[i:i + window_size].values
        distance = euclidean(candidate, query_pattern)
        future_prices = df['Close'].iloc[i + window_size:i + window_size + future_days].values
        matches.append((distance, future_prices))

    matches = sorted(matches, key=lambda x: x[0])[:top_n]
    predicted_paths = [match[1] for match in matches]

    if len(predicted_paths) == 0:
        raise ValueError("歷史資料中找不到足夠相似的價格走勢來預測。請嘗試降低 window_size 或使用不同的股票。")

    normalized_paths = [path / path[0] * last_price for path in predicted_paths]
    average_path = np.mean(normalized_paths, axis=0)
    average_path = np.array(average_path)

    # 統計預測方向
    ups = sum(1 for path in normalized_paths if path[-1] > last_price)
    downs = sum(1 for path in normalized_paths if path[-1] < last_price)
    neutrals = top_n - ups - downs

    stats_dict = {
        'up_probability': ups / top_n,
        'down_probability': downs / top_n,
        'neutral_probability': neutrals / top_n,
        'last_price': last_price,
        'predicted_price': average_path[-1],
        'expected_return_percent': (average_path[-1] - last_price) / last_price * 100
    }

    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days)
    predicted_df = pd.DataFrame({
        '預測日期': future_dates,
        '預測股價': average_path
    })

    return average_path, normalized_paths, stats_dict, predicted_df

# 繪圖函式
def plot_predictions(average_path, normalized_paths, last_price):
    future_days = len(average_path)
    fig, ax = plt.subplots(figsize=(10, 6))
    for path in normalized_paths:
        ax.plot(range(1, future_days + 1), path, alpha=0.4, linestyle='--')
    ax.plot(range(1, future_days + 1), average_path, label='Average Prediction', linewidth=2, color='black')
    ax.axhline(last_price, color='gray', linestyle=':', label='Current Price')
    ax.set_title('Stock Price Prediction Based on Historical Similarity')
    ax.set_xlabel('Days Ahead')
    ax.set_ylabel('Predicted Price')
    ax.grid(True)
    ax.legend()
    return fig

# ========== Streamlit Web App ==========
st.title("📈 股價預測 Web App")
ticker = st.text_input("輸入股票代碼（例如：AAPL, TSM）", "AAPL")

if st.button("開始預測"):
    try:
        df = yf.download(ticker, period='5y', interval='1d')
        if df.empty:
            raise ValueError("查無股票資料，請確認代碼是否正確。")
        
        avg_path, all_paths, stats, output_df = predict_stock_price(df)

        fig = plot_predictions(avg_path, all_paths, stats['last_price'])
        st.pyplot(fig)

        st.success(f"漲機率: {stats['up_probability']:.2%}，跌機率: {stats['down_probability']:.2%}，預測報酬率: {stats['expected_return_percent']:.2f}%")
        st.subheader("📊 預測結果表格")
        st.dataframe(output_df)

        csv = output_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📁 下載預測結果 (CSV)", csv, f"{ticker}_prediction.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ 錯誤：{e}")
