import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import tensorflow as tf
from datetime import date, timedelta

st.set_page_config(page_title="Forecasting Qty Penjualan Tanaman", page_icon="🌿", layout="wide")

@st.cache_resource
def load_resources():
    lstm = tf.keras.models.load_model("model_lstm.keras")
    gru = tf.keras.models.load_model("model_gru.keras")
    with open("artifacts.pkl", "rb") as f:
        artifacts = pickle.load(f)
    dataset = pd.read_csv("clean_dataset_final.csv")
    dataset["tanggal"] = pd.to_datetime(dataset["tanggal"])
    return lstm, gru, artifacts, dataset

try:
    model_lstm, model_gru, artifacts, df = load_resources()
    scalers = artifacts["scalers"]
    product_meta = artifacts["product_meta"]
    window_size = artifacts["window_size"]
    metrics_df = pd.DataFrame(artifacts["metrics"])
except Exception as e:
    st.error(f"Gagal memuat file pendukung: {e}")
    st.stop()

st.title("🌿 Prediksi Penjualan Tanaman & Saprotan")
st.caption("Forecasting Time Series Cepat Rentang Kalender Menggunakan Model LSTM & GRU")

# --- SIDEBAR PENGATURAN ---
st.sidebar.header("Filter & Parameter")
prod_map = {item["nama_produk"]: item["product_id"] for item in product_meta}
selected_prod_name = st.sidebar.selectbox("Pilih Produk:", list(prod_map.keys()))
selected_pid = prod_map[selected_prod_name]
selected_model_type = st.sidebar.radio("Pilih Model:", ["Bandingkan Keduanya", "LSTM", "GRU"])

st.sidebar.subheader("Pilih Tanggal Target")
st.sidebar.info("Maksimal rentang prediksi kalender 30 hari.")

# Default rentang tanggal: 3 - 14 Oktober 2026
default_start = date(2026, 10, 3)
default_end = date(2026, 10, 14)

picked_dates = st.sidebar.date_input("Rentang Tanggal:", value=(default_start, default_end))

if not isinstance(picked_dates, tuple) or len(picked_dates) != 2:
    st.info("Tentukan tanggal awal dan akhir di kalender samping.")
    st.stop()

start_date, end_date = picked_dates
num_days = (end_date - start_date).days + 1

if num_days <= 0:
    st.error("Tanggal akhir harus sama atau setelah tanggal mulai.")
    st.stop()
if num_days > 30:
    st.error(f"Rentang kalender ({num_days} hari) melebihi batas 30 hari.")
    st.stop()

# --- DISPLAY KPI EVALUASI TEST SET (RMSE, MAE, MAPE) ---
st.subheader("Evaluasi Performa Model (Test Set)")
kpi1, kpi2, kpi3 = st.columns(3)

lstm_row = metrics_df[metrics_df["Model"] == "LSTM"].iloc[0]
gru_row = metrics_df[metrics_df["Model"] == "GRU"].iloc[0]

with kpi1:
    st.metric(label="RMSE", value=f"{lstm_row['RMSE']:.2f}", delta=f"GRU: {gru_row['RMSE']:.2f}", delta_color="off")
with kpi2:
    st.metric(label="MAE", value=f"{lstm_row['MAE']:.2f}", delta=f"GRU: {gru_row['MAE']:.2f}", delta_color="off")
with kpi3:
    st.metric(label="MAPE", value=f"{lstm_row['MAPE (%)']:.2f}%", delta=f"GRU: {gru_row['MAPE (%)']:.2f}%", delta_color="off")

st.markdown("---")

# --- PROSES FORECASTING CEPAT ---
prod_data = df[df["product_id"] == selected_pid].sort_values("tanggal").reset_index(drop=True)
scaler = scalers[selected_pid]
last_hist_date = prod_data["tanggal"].max().date()

if start_date <= last_hist_date:
    st.warning(f"Pilih rentang tanggal setelah data histori ({last_hist_date.strftime('%d-%m-%Y')}).")
    st.stop()

total_steps = (end_date - last_hist_date).days
future_date_list = [last_hist_date + timedelta(days=i) for i in range(1, total_steps + 1)]

with st.spinner("Memproses prediksi multi-langkah..."):
    # Window input terakhir
    recent_sub = prod_data.iloc[-window_size:]
    init_seq = []
    for _, r in recent_sub.iterrows():
        init_seq.append([
            scaler.transform([[r["qty_filled"]]])[0][0],
            r["tanggal"].dayofweek / 6.0,
            (r["tanggal"].month - 1) / 11.0,
            (r["tanggal"].day - 1) / 30.0
        ])
        
    seq_lstm = np.array(init_seq, dtype=np.float32)
    seq_gru = np.array(init_seq, dtype=np.float32)
    
    preds_lstm_scaled = []
    preds_gru_scaled = []
    
    # Fast forward loop
    for d in future_date_list:
        dow_v = d.weekday() / 6.0
        mon_v = (d.month - 1) / 11.0
        day_v = (d.day - 1) / 30.0
        
        # LSTM
        in_l = seq_lstm[-window_size:].reshape(1, window_size, 4)
        out_l = float(model_lstm(in_l, training=False)[0, 0])
        preds_lstm_scaled.append(out_l)
        seq_lstm = np.vstack([seq_lstm, [out_l, dow_v, mon_v, day_v]])
        
        # GRU
        in_g = seq_gru[-window_size:].reshape(1, window_size, 4)
        out_g = float(model_gru(in_g, training=False)[0, 0])
        preds_gru_scaled.append(out_g)
        seq_gru = np.vstack([seq_gru, [out_g, dow_v, mon_v, day_v]])

# Potong sesuai rentang tanggal kustom yang dipilih
str_dates = [d.strftime("%Y-%m-%d") for d in future_date_list]
idx_s = str_dates.index(start_date.strftime("%Y-%m-%d"))
idx_e = str_dates.index(end_date.strftime("%Y-%m-%d")) + 1

target_dates = future_date_list[idx_s:idx_e]
res_lstm = np.clip(scaler.inverse_transform(np.array(preds_lstm_scaled[idx_s:idx_e]).reshape(-1, 1)).flatten(), 0, None)
res_gru = np.clip(scaler.inverse_transform(np.array(preds_gru_scaled[idx_s:idx_e]).reshape(-1, 1)).flatten(), 0, None)

# --- VISUALISASI PLOTLY ---
st.subheader(f"Estimasi Qty: {selected_prod_name.replace('_', ' ').title()}")
st.caption(f"Rentang Prediksi: {start_date.strftime('%d %b %Y')} s.d. {end_date.strftime('%d %b %Y')} ({num_days} Hari)")

fig = go.Figure()
date_labels = [d.strftime("%Y-%m-%d") for d in target_dates]

if selected_model_type in ["Bandingkan Keduanya", "LSTM"]:
    fig.add_trace(go.Scatter(x=date_labels, y=res_lstm, mode="lines+markers", name="LSTM", line=dict(color="#1f77b4", width=3)))

if selected_model_type in ["Bandingkan Keduanya", "GRU"]:
    fig.add_trace(go.Scatter(x=date_labels, y=res_gru, mode="lines+markers", name="GRU", line=dict(color="#ff7f0e", width=3, dash="dash")))

fig.update_layout(xaxis_title="Tanggal", yaxis_title="Perkiraan Qty", hovermode="x unified", margin=dict(l=20, r=20, t=30, b=20))
st.plotly_chart(fig, use_container_width=True)

# --- TABEL RINCIAN ---
out_df = pd.DataFrame({"Tanggal": date_labels, "Hari": [d.strftime("%A") for d in target_dates]})
if selected_model_type in ["Bandingkan Keduanya", "LSTM"]:
    out_df["Prediksi LSTM"] = np.round(res_lstm, 1)
if selected_model_type in ["Bandingkan Keduanya", "GRU"]:
    out_df["Prediksi GRU"] = np.round(res_gru, 1)

st.dataframe(out_df, use_container_width=True)