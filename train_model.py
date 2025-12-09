# train_model.py

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# === Path dasar proyek ===
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "rainfall_bogor.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


def load_and_reshape_data(path: Path) -> pd.Series:
    """
    Baca data wide (Tahun, CH_Jan..CH_Dec) → jadi seri time-series bulanan.
    Index = datetime (YYYY-MM-01), values = rain_mm.
    """
    # 1. Baca CSV dengan delimiter ';'
    df_wide = pd.read_csv(path, sep=';')

    # 2. Buang kolom kosong (karena ;;; di akhir)
    df_wide = df_wide.dropna(axis=1, how='all')

    # Rapikan nama kolom
    df_wide.columns = df_wide.columns.str.strip()

    print("Kolom di CSV:", list(df_wide.columns))

    # 3. Ubah wide → long
    # ambil semua kolom bulan kecuali 'Tahun'
    month_cols = [c for c in df_wide.columns if c != "Tahun"]

    df_long = df_wide.melt(
        id_vars=["Tahun"],
        value_vars=month_cols,
        var_name="month_name",
        value_name="rain_mm",
    )

    # Mapping nama kolom → nomor bulan
    month_map = {
        "CH_Jan": 1,
        "CH_Feb": 2,
        "CH_Mar": 3,
        "CH_Apr": 4,
        "CH_May": 5,
        "CH_Jun": 6,
        "CH_Jul": 7,
        "CH_Aug": 8,
        "CH_Sep": 9,
        "CH_Oct": 10,
        "CH_Nov": 11,
        "CH_Dec": 12,
    }

    df_long["Month"] = df_long["month_name"].map(month_map)
    # buang baris yang month-nya ga ke-map (kalau ada typo)
    df_long = df_long.dropna(subset=["Month"])

    # pastikan numeric
    df_long["rain_mm"] = pd.to_numeric(df_long["rain_mm"], errors="coerce")
    df_long = df_long.dropna(subset=["rain_mm"])

    # 4. Buat kolom tanggal (YYYY-MM-01)
    df_long["Tahun"] = df_long["Tahun"].astype(int)
    df_long["Month"] = df_long["Month"].astype(int)

    df_long["date"] = pd.to_datetime(
        dict(year=df_long["Tahun"], month=df_long["Month"], day=1)
    )

    df_long = df_long.sort_values("date").reset_index(drop=True)

    # 5. Jadikan Series dengan index datetime
    series = df_long.set_index("date")["rain_mm"].copy()
    series = series.sort_index()
    series.name = "rain_mm"

    print("Sample series:")
    print(series.head())

    return series


def make_supervised(series: pd.Series, n_lags: int = 12) -> pd.DataFrame:
    """
    Buat fitur supervised learning:
    - target: rain_mm
    - fitur: lag_1..lag_n, month_sin, month_cos
    """
    df = series.to_frame()

    # fitur lag
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df["rain_mm"].shift(lag)

    # fitur musiman
    df["month"] = df.index.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

    df = df.dropna()
    return df


def train_model():
    # --- Load & reshape ---
    series = load_and_reshape_data(DATA_PATH)

    # --- Buat dataset supervised ---
    n_lags = 12
    data = make_supervised(series, n_lags=n_lags)

    X = data.drop(columns=["rain_mm"])
    y = data["rain_mm"]

    # ===== Split: fokus 5 tahun terakhir sebagai test (60 bulan) =====
    test_horizon = 60  # 5 tahun
    test_horizon = min(test_horizon, len(data) // 3)  # jaga-jaga kalau data pendek

    X_train, X_test = X.iloc[:-test_horizon], X.iloc[-test_horizon:]
    y_train, y_test = y.iloc[:-test_horizon], y.iloc[-test_horizon:]

    print("Train size:", X_train.shape[0], "bulan")
    print("Test  size:", X_test.shape[0], "bulan")

    # ===== Model Random Forest =====
    model = RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    print(f"\nMAE (5 tahun terakhir): {mae:.2f} mm")

    # ===== Simpan artifact =====
    last_date = series.index.max()
    artifact = {
        "model": model,
        "n_lags": n_lags,
        "last_date": last_date,
    }

    joblib_path = MODELS_DIR / "rainfall_rf.joblib"
    joblib.dump(artifact, joblib_path)

    print(f"\nModel disimpan ke: {joblib_path}")
    print(f"Data terakhir: {last_date.strftime('%Y-%m')}")


if __name__ == "__main__":
    train_model()
