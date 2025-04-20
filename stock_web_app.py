import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

# ===== 股價預測主函數 =====
def predict_stock_price(df, window_size=10, future_days=5, top_n=5):
    df = df[['Close']].dropna().copy()
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    df = df.dropna().reset_index(drop=True)

    if len(df) <= window_size + future_days:
        raise ValueError("歷史資料太少，無法進行預測。請選擇其他股票或調整參數。")

    query_pattern = df['log_return'][-window_size:].values
    last_price = df['Close'].iloc[-1]

    matches = []
    for i in range(len(df) - window_size - future_days):
        candidate = df['log_return'].iloc[i:i + window_size].values
        distance = euclidean(candidate, query_pattern)
        future_prices = df['Close'].iloc[i + window_size:i + window_size + future_days].values
        if len(future_prices) == future_days:
            matches.append((distance, future_prices))

    matches = sorted(matches, key=lambda x: x[0])[:top_n]

    if not matches:
        raise ValueError("找不到足夠的匹配模式。")

    predicted_paths = [match[1] for match in matches]
    normalized_paths = [path / path[0] * last_price for path in predicted_paths]
    average_path = np.mean(normalized_paths, axis=0)

    ups = sum(path[-1] > last_price for path in normalized_paths)
    downs = sum(path[-1] < last_price for path in normalized_paths)
    neutrals = top_n - ups - downs

    stats = {
        'up_probability': ups / top_n,
        'down_probability': downs / top_n,
        'neutral_probability': neutrals / top_n,
        'last_price': last_price,
        'predicted_price': average_path[-1],
        'expected_return_percent': (average_path[-1] - last_price) / last_price * 100
    }

    predicted_df = pd.DataFrame({
        'Day': np.arange(1, future_days + 1),
        'PredictedPrice': average_path
    })

    return average_path, normalized_paths, stats, predicted_df

# ===== 圖表繪製 =====
def plot_predictions(average_path, normalized_paths, last_price):
    fig, ax = plt.subplots(figsize=(10, 6))
    for path in normalized_paths:
        ax.plot(range(1, len(path) + 1), path, alpha=0.4, linestyle='--')
    ax.plot(range(1, len(average_path) + 1), average_path, color='black', linewidth=2, label='平均預測')
    ax.axhline(last_price, linestyle=':', color='gray', label='當前價格')
    ax.set_title('📉 預測未來股價趨勢')
    ax.set_xlabel('未來第 N 天')
    ax.set_ylabel('預測股價')
    ax.legend()
    ax.grid(True)
    return fig

# ===== Streamlit 主介面 =====
st.set_page_config(page_title="股價預測 Web App", layout="centered")
st.title("📈 股價預測 Web App")

ticker = st.text_input("輸入股票代碼（例如：AAPL、TSM）", "AAPL")

if st.button("開始預測"):
    try:
        df = yf.download(ticker, period='5y', interval='1d')
        if df.empty:
            raise ValueError("找不到該股票代碼的資料。請確認輸入是否正確。")

        avg_path, all_paths, stats, output_df = predict_stock_price(df)

        fig = plot_predictions(avg_path, all_paths, stats['last_price'])
        st.pyplot(fig)

        st.success(
            f"📊 預測結果：漲機率 {stats['up_probability']:.2%}，跌機率 {stats['down_probability']:.2%}，"
            f"預測報酬率 {stats['expected_return_percent']:.2f}%"
        )

        st.dataframe(output_df)

        csv = output_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📁 下載預測結果 (CSV)", csv, f"{ticker}_prediction.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ 錯誤：{e}")
