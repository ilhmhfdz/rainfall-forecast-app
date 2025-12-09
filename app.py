import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import streamlit as st
import plotly.express as px

# ============ BASIC CONFIG ============
st.set_page_config(
    page_title="Bogor Rainfall Forecasting",
    page_icon="🌧️",
    layout="wide",
)

# ==== Path dasar ====
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "rainfall_bogor.csv"
MODELS_DIR = BASE_DIR / "models"


# ============ FUNGSI UTIL ============

@st.cache_resource
def load_artifact():
    artifact_path = MODELS_DIR / "rainfall_rf.joblib"
    artifact = joblib.load(artifact_path)
    return artifact


@st.cache_data
def load_history_series():
    """
    Baca CSV (format wide: Tahun, CH_Jan..CH_Dec) -> series bulanan (date index).
    """
    df_wide = pd.read_csv(DATA_PATH, sep=";")
    df_wide = df_wide.dropna(axis=1, how="all")  # buang kolom kosong
    df_wide.columns = df_wide.columns.str.strip()

    # wide -> long
    df_long = df_wide.melt(
        id_vars=["Tahun"],
        var_name="bulan",
        value_name="rain_mm",
    )

    # parse nama bulan dari kolom CH_Jan, CH_Feb, dst.
    month_map = {
        "CH_Jan": 1, "CH_Feb": 2, "CH_Mar": 3, "CH_Apr": 4,
        "CH_May": 5, "CH_Jun": 6, "CH_Jul": 7, "CH_Aug": 8,
        "CH_Sep": 9, "CH_Oct": 10, "CH_Nov": 11, "CH_Dec": 12,
    }
    df_long["Month"] = df_long["bulan"].map(month_map)

    df_long["date"] = pd.to_datetime(
        dict(year=df_long["Tahun"], month=df_long["Month"], day=1)
    )

    df_long = df_long.sort_values("date").reset_index(drop=True)

    series = df_long[["date", "rain_mm"]].set_index("date")["rain_mm"]
    series = series.sort_index()
    return series


def forecast_months(model, series, n_lags: int, months_ahead: int) -> pd.Series:
    """
    Iteratif: pakai prediksi bulan sebelumnya sebagai lag untuk bulan berikutnya.
    """
    history = series.copy()
    last_date = history.index.max()
    last_values = history.copy()

    future_dates = []
    future_vals = []

    for step in range(1, months_ahead + 1):
        next_date = last_date + pd.DateOffset(months=step)

        # siapkan fitur lag
        lag_vals = last_values.iloc[-n_lags:]
        row = {}
        # lag_1 = bulan terakhir, lag_12 = 12 bulan sebelumnya
        for i, val in enumerate(reversed(lag_vals), start=1):
            row[f"lag_{i}"] = val

        m = next_date.month
        row["month"] = m
        row["month_sin"] = np.sin(2 * np.pi * m / 12.0)
        row["month_cos"] = np.cos(2 * np.pi * m / 12.0)

        X_new = pd.DataFrame([row])
        y_hat = model.predict(X_new)[0]

        future_dates.append(next_date)
        future_vals.append(y_hat)

        last_values = pd.concat(
            [last_values, pd.Series([y_hat], index=[next_date])]
        )

    future_series = pd.Series(future_vals, index=future_dates, name="rain_mm")
    return future_series


def generate_insight(year_series: pd.Series, year: int) -> str:
    """
    Insight simple berbasis aturan dari satu tahun forecast.
    """
    if year_series.empty:
        return "Tidak ada data untuk tahun tersebut."

    avg = year_series.mean()
    max_val = year_series.max()
    min_val = year_series.min()
    max_month = year_series.idxmax().strftime("%B")
    min_month = year_series.idxmin().strftime("%B")

    text = f"- Rata-rata curah hujan {year}: sekitar **{avg:.1f} mm/bulan**.\n"
    text += (
        f"- Puncak curah hujan diprediksi pada **{max_month} {year}** "
        f"dengan sekitar **{max_val:.1f} mm**.\n"
    )
    text += (
        f"- Bulan paling kering diperkirakan **{min_month} {year}** "
        f"dengan sekitar **{min_val:.1f} mm**.\n"
    )

    if max_val > 400:
        text += (
            "- Terdapat bulan dengan curah hujan sangat tinggi. "
            "Perlu diwaspadai untuk aktivitas konstruksi / outdoor skala besar.\n"
        )
    elif max_val > 300:
        text += (
            "- Musim hujan cukup intens, sebaiknya jadwalkan pekerjaan lapangan "
            "di bulan-bulan dengan curah hujan lebih rendah.\n"
        )
    else:
        text += (
            "- Pola curah hujan relatif sedang, lebih fleksibel untuk penjadwalan aktivitas.\n"
        )

    return text


# ============ UI STREAMLIT ============

st.title("Kabupaten Bogor Rainfall Forecasting App")
st.caption(
    "Prediksi curah hujan hingga 10 tahun ke depan menggunakan Machine Learning "
    "(Random Forest) berdasarkan data historis BMKG."
)

# ---- Sidebar: kontrol user ----
artifact = load_artifact()
model = artifact["model"]
n_lags = artifact["n_lags"]
last_date = artifact["last_date"]

history = load_history_series()

st.sidebar.header("Pengaturan Forecast")

years_ahead = st.sidebar.slider(
    "Berapa tahun ke depan?",
    min_value=1,
    max_value=10,
    value=10,
    help="Model akan memprediksi per bulan hingga jumlah tahun yang dipilih.",
)

# Tema chart
theme_choice = st.sidebar.radio(
    "Tema chart",
    options=["Light", "Dark"],
    index=0,
)
plotly_template = "plotly_white" if theme_choice == "Light" else "plotly_dark"

# Download forecast (disiapkan di bawah setelah future dihitung)
st.sidebar.markdown("---")

months_ahead = years_ahead * 12
future = forecast_months(model, history, n_lags, months_ahead)

start_forecast_year = last_date.year + 1
end_forecast_year = start_forecast_year + years_ahead - 1

focus_year = st.sidebar.selectbox(
    "Fokus lihat tahun berapa?",
    options=list(range(start_forecast_year, end_forecast_year + 1)),
)

# ============ GABUNG HISTORY + FORECAST ============

df_hist = history.rename("rain_mm").to_frame()
df_hist["type"] = "History"

df_future = future.rename("rain_mm").to_frame()
df_future["type"] = "Forecast"

df_all = pd.concat([df_hist, df_future]).reset_index().rename(columns={"index": "date"})

# ============ LAYOUT DENGAN TABS ============

tab1, tab2 = st.tabs([" Overview", " Detail per Tahun"])

# -------------------------------------------------------------------
# TAB 1: OVERVIEW
# -------------------------------------------------------------------
with tab1:
    st.subheader("Rainfall Forecast (History vs Forecast)")

    fig_all = px.line(
        df_all,
        x="date",
        y="rain_mm",
        color="type",
        line_dash="type",
        labels={"rain_mm": "Curah Hujan (mm)", "date": "Bulan", "type": ""},
        template=plotly_template,
    )
    fig_all.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode="x unified",
    )
    st.plotly_chart(fig_all, use_container_width=True)

    # Download hanya forecast
    st.markdown("### 📥 Download Data Forecast")
    forecast_df = df_future.copy().reset_index().rename(columns={"index": "date"})
    csv_bytes = forecast_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download forecast (CSV)",
        data=csv_bytes,
        file_name="rainfall_forecast_bogor.csv",
        mime="text/csv",
        help="Unduh hasil prediksi curah hujan untuk analisis lebih lanjut.",
    )

# -------------------------------------------------------------------
# TAB 2: DETAIL PER TAHUN
# -------------------------------------------------------------------
with tab2:
    st.subheader(f"Detail Curah Hujan Tahun {focus_year}")

    year_series = future[future.index.year == focus_year]

    if not year_series.empty:
        df_year = year_series.to_frame().reset_index()
        df_year["month_name"] = df_year["index"].dt.strftime("%b")

        # ---- KPI / METRICS ----
        mean_rain = year_series.mean()
        max_rain = year_series.max()
        min_rain = year_series.min()
        peak_month = year_series.idxmax().strftime("%b")
        dry_month = year_series.idxmin().strftime("%b")

        col1, col2, col3 = st.columns(3)
        col1.metric("Rata-rata", f"{mean_rain:.1f} mm/bulan")
        col2.metric("Puncak", f"{max_rain:.1f} mm", peak_month)
        col3.metric("Terkering", f"{min_rain:.1f} mm", dry_month)

        # ---- BAR CHART DETAIL TAHUN ----
        fig_year = px.bar(
            df_year,
            x="month_name",
            y="rain_mm",
            labels={"rain_mm": "Curah Hujan (mm)", "month_name": "Bulan"},
            template=plotly_template,
        )
        fig_year.update_layout(
            margin=dict(l=40, r=40, t=40, b=40),
            hovermode="x",
        )
        st.plotly_chart(fig_year, use_container_width=True)

        # ---- INSIGHT ----
        st.markdown("### AI-style Insight (Rule-Based)")
        insight_text = generate_insight(year_series, focus_year)
        st.markdown(insight_text)

    else:
        st.info("Tidak ada data untuk tahun tersebut (di luar horizon prediksi).")
