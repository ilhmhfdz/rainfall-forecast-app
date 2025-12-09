import numpy as np
import pandas as pd
from pathlib import Path
import joblib


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "rainfall_rf.joblib"


def load_model():
    artifact = joblib.load(MODEL_PATH)
    return artifact["model"], artifact["n_lags"], artifact["last_date"]


def iterative_forecast(model, history_series, n_lags, steps=120):
    """
    Forecast masa depan bulan demi bulan secara autoregresif
    history_series: pandas Series dengan index datetime dan nilai rain_mm
    """
    forecast_values = []
    history = history_series.copy()

    for _ in range(steps):
        # Ambil lag feature
        last_values = history[-n_lags:].values
        if len(last_values) < n_lags:
            raise ValueError("Series terlalu pendek untuk n_lags.")

        # Buat DataFrame 1 row fitur
        df_feat = pd.DataFrame([{
            **{f"lag_{i+1}": last_values[-(i+1)] for i in range(n_lags)},
            "month": history.index[-1].month,
        }])

        # Feature musiman
        df_feat["month_sin"] = np.sin(2 * np.pi * df_feat["month"] / 12.0)
        df_feat["month_cos"] = np.cos(2 * np.pi * df_feat["month"] / 12.0)

        # Prediksi bulan selanjutnya
        y_pred = model.predict(df_feat)[0]
        forecast_values.append(y_pred)

        # Tambahkan ke history sebagai data baru
        next_date = history.index[-1] + pd.DateOffset(months=1)
        history.loc[next_date] = y_pred

    # Convert forecast ke DataFrame
    forecast_index = pd.date_range(
        start=history_series.index.max() + pd.DateOffset(months=1),
        periods=steps,
        freq="MS"
    )

    return pd.DataFrame({"forecast": forecast_values}, index=forecast_index)


def load_historical():
    """Load ulang series hasil training untuk dikombinasi dengan forecast."""
    artifact = joblib.load(MODEL_PATH)
    return artifact["last_date"]
