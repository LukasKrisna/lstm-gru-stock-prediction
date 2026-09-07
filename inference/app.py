import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import tensorflow as tf
from datetime import date, timedelta

st.set_page_config(page_title="Forecasting Qty Penjualan Tanaman", page_icon="🌿", layout="wide")

# ---------------------------------------------------------------------------
# Horizon rollout tetap (fixed), dihitung SEKALI untuk SEMUA produk & kedua
# model saat startup, lepas dari input UI apa pun. Semua interaksi user
# (ganti produk / geser tanggal / ganti model) sesudahnya tinggal MENGIRIS
# array hasil ini -> tidak ada rollout ulang (lihat run_full_rollout()).
#
# 420 hari (~14 bulan) dipilih agar mencakup target default (~314 langkah
# dari histori s.d. Okt 2026) + buffer. Jika verifikasi menunjukkan histori
# antar produk berakhir di tanggal yang jauh berbeda, sesuaikan angka ini
# supaya tetap mencakup rentang tanggal yang ingin ditampilkan di UI.
# ---------------------------------------------------------------------------
FORECAST_HORIZON_DAYS = 420


@st.cache_resource
def load_resources():
    lstm = tf.keras.models.load_model("model_lstm.keras")
    gru = tf.keras.models.load_model("model_gru.keras")
    with open("artifacts.pkl", "rb") as f:
        artifacts = pickle.load(f)
    dataset = pd.read_csv("clean_dataset_final.csv")
    dataset["tanggal"] = pd.to_datetime(dataset["tanggal"])

    # --- fix #3: group & sort SEKALI di sini, bukan filter+sort 48k baris
    # setiap script rerun (tiap klik widget) ---
    grouped = {
        pid: g.sort_values("tanggal").reset_index(drop=True)
        for pid, g in dataset.groupby("product_id")
    }
    return lstm, gru, artifacts, dataset, grouped


try:
    model_lstm, model_gru, artifacts, df, grouped_by_product = load_resources()
    scalers = artifacts["scalers"]
    product_meta = artifacts["product_meta"]
    window_size = artifacts["window_size"]
    metrics_df = pd.DataFrame(artifacts["metrics"])
except Exception as e:
    st.error(f"Gagal memuat file pendukung: {e}")
    st.stop()

prod_ids = [item["product_id"] for item in product_meta]


# ---------------------------------------------------------------------------
# fix #1 & #6: rollout multi-step di-porting SEPENUHNYA ke dalam graph TF
# (tf.while_loop lewat AutoGraph `for t in tf.range(...)`), dan dibatch utk
# SEMUA produk sekaligus. Ini mengubah ~628 pemanggilan model(x) eager
# batch=1 dari Python jadi 1 pemanggilan graph per model (2 total), dan
# trace pertama terjadi saat run_full_rollout() dipanggil di bawah -- yaitu
# saat script pertama kali start di server, bukan saat user pertama klik.
# ---------------------------------------------------------------------------
def _build_rollout_fn(model):
    @tf.function(reduce_retracing=True)
    def rollout(init_seq, date_feats):
        # init_seq  : (n_produk, window_size, 4) window awal per produk
        # date_feats: (n_produk, horizon, 3) fitur (dow, bulan, tanggal)
        #             tiap langkah ke depan, dihitung dari kalender asli
        #             per produk di luar graph (numpy)
        seq = init_seq
        preds = tf.TensorArray(dtype=tf.float32, size=FORECAST_HORIZON_DAYS)
        for t in tf.range(FORECAST_HORIZON_DAYS):
            out = model(seq, training=False)  # (n_produk, 1) scaled
            preds = preds.write(t, out[:, 0])
            step_feat = tf.gather(date_feats, t, axis=1)  # (n_produk, 3)
            new_row = tf.concat([out, step_feat], axis=1)  # (n_produk, 4)
            seq = tf.concat([seq[:, 1:, :], new_row[:, tf.newaxis, :]], axis=1)
        return tf.transpose(preds.stack())  # (n_produk, horizon)

    return rollout


def _build_init_and_feats(pid):
    prod_df = grouped_by_product[pid]
    last_hist_date = prod_df["tanggal"].max().date()
    recent = prod_df.iloc[-window_size:]
    scaler = scalers[pid]

    # fix #2: scaler.transform dipanggil SEKALI utk seluruh window (vektor),
    # bukan 14x dalam loop baris-per-baris.
    qty_scaled = scaler.transform(recent[["qty_filled"]].values).flatten()
    dow = recent["tanggal"].dt.dayofweek.values / 6.0
    mon = (recent["tanggal"].dt.month.values - 1) / 11.0
    day = (recent["tanggal"].dt.day.values - 1) / 30.0
    init_seq = np.stack([qty_scaled, dow, mon, day], axis=1).astype(np.float32)

    future_dates = [last_hist_date + timedelta(days=i) for i in range(1, FORECAST_HORIZON_DAYS + 1)]
    fd = pd.to_datetime(future_dates)
    feats = np.stack(
        [
            fd.dayofweek.values / 6.0,
            (fd.month.values - 1) / 11.0,
            (fd.day.values - 1) / 30.0,
        ],
        axis=1,
    ).astype(np.float32)

    return init_seq, feats, last_hist_date, future_dates


@st.cache_data(show_spinner=False)
def run_full_rollout():
    """
    Rollout SEMUA produk x SEMUA hari (FORECAST_HORIZON_DAYS) x kedua model,
    dihitung sekali. Fungsi ini TIDAK menerima parameter dari UI (produk,
    tanggal, atau model_type) sama sekali -- fix #4 & #5: cache key tidak
    pernah ikut berubah gara-gara user ganti pilihan model atau geser
    tanggal 1 hari, jadi cache tidak pernah invalid selama artifacts.pkl /
    dataset yang di-load di load_resources() belum berubah.

    Catatan: fungsi ini TIDAK mengasumsikan seluruh produk punya tanggal
    akhir histori yang sama -- tiap produk roll-forward dari tanggal akhir
    histori-nya SENDIRI. Jadi batching ini tetap valid & benar walaupun
    asumsi itu ternyata salah; asumsi tsb hanya relevan utk menyederhanakan
    persiapan fitur tanggal (lihat pesan di bawah), bukan syarat mutlak.
    """
    init_all, feats_all, meta = [], [], []
    for pid in prod_ids:
        init_seq, feats, last_hist_date, future_dates = _build_init_and_feats(pid)
        init_all.append(init_seq)
        feats_all.append(feats)
        meta.append({"product_id": pid, "last_hist_date": last_hist_date, "future_dates": future_dates})

    init_batch = tf.constant(np.stack(init_all, axis=0))
    feats_batch = tf.constant(np.stack(feats_all, axis=0))

    preds_lstm_scaled = _build_rollout_fn(model_lstm)(init_batch, feats_batch).numpy()
    preds_gru_scaled = _build_rollout_fn(model_gru)(init_batch, feats_batch).numpy()

    preds_lstm, preds_gru = {}, {}
    for i, pid in enumerate(prod_ids):
        sc = scalers[pid]
        preds_lstm[pid] = np.clip(sc.inverse_transform(preds_lstm_scaled[i].reshape(-1, 1)).flatten(), 0, None)
        preds_gru[pid] = np.clip(sc.inverse_transform(preds_gru_scaled[i].reshape(-1, 1)).flatten(), 0, None)

    return preds_lstm, preds_gru, {m["product_id"]: m for m in meta}


# Dipanggil TANPA argumen & TANPA syarat, langsung setelah resources
# ter-load -- ini yang memindahkan biaya trace graph + rollout awal ke saat
# server start, bukan menunggu interaksi pertama user (fix #6).
with st.spinner("Menyiapkan model & rollout awal (hanya terjadi sekali di server)..."):
    preds_lstm_all, preds_gru_all, meta_by_pid = run_full_rollout()

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

last_hist_date = meta_by_pid[selected_pid]["last_hist_date"]
future_dates = meta_by_pid[selected_pid]["future_dates"]
min_selectable_date = last_hist_date + timedelta(days=1)
max_selectable_date = future_dates[-1]

# Default rentang tanggal: 3 - 14 Oktober 2026, di-clamp supaya tetap valid
# walau utk produk tertentu histori-nya berakhir jauh lebih telat/awal.
default_start = min(max(date(2026, 10, 3), min_selectable_date), max_selectable_date)
default_end = min(max(date(2026, 10, 14), default_start), max_selectable_date)

picked_dates = st.sidebar.date_input(
    "Rentang Tanggal:",
    value=(default_start, default_end),
    min_value=min_selectable_date,
    max_value=max_selectable_date,
)

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

# --- AMBIL HASIL DARI CACHE (tinggal mengiris array, tanpa rollout ulang) ---
idx_s = (start_date - last_hist_date).days - 1
idx_e = (end_date - last_hist_date).days

target_dates = future_dates[idx_s:idx_e]
res_lstm = preds_lstm_all[selected_pid][idx_s:idx_e]
res_gru = preds_gru_all[selected_pid][idx_s:idx_e]

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