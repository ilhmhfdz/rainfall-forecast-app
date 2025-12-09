# 🌧️ Bogor Rainfall Forecasting App  
Machine Learning for Climate Risk & Operational Decision Support

Aplikasi interaktif untuk **memprediksi curah hujan bulanan hingga 10 tahun ke depan** menggunakan model **Random Forest Regressor**, berbasis data historis BMKG Kabupaten Bogor.

---

## 🔗 Live Demo & Source Code

| Resource | Link |
|---------|------|
| 🌍 Streamlit App | https://rainfall-forecast-app-sygttgiqfy3aehbhoyefvr.streamlit.app/ |
| 📦 GitHub Repo | https://github.com/ilhmhfdz/rainfall-forecast-app |

---

## 🚀 Features

- 🔮 Prediksi curah hujan hingga **10 tahun ke depan**
- 📊 Visualisasi interaktif (Plotly line & bar chart)
- 🎯 Pilih fokus analisis per tahun
- 📌 Automated AI-style insights untuk pengambilan keputusan
- 🌗 Tema Light & Dark untuk tampilan chart
- ⚙️ Model ML teroptimasi dengan fitur musiman (sin/cos)

---

## 🎯 Project Motivation

Prediksi curah hujan sangat penting untuk:

- Perencanaan konstruksi
- Mitigasi risiko banjir
- Perencanaan pertanian dan logistik
- Manajemen operasional kegiatan outdoor

Solusi ini memberikan dukungan keputusan berbasis data untuk sektor publik & industri.

---

## 🧠 Machine Learning Model

| Item | Detail |
|------|-------|
| Model | RandomForestRegressor |
| Input | 12 bulan curah hujan sebelumnya + fitur musiman |
| Teknik | Iterative autoregression forecasting |
| Evaluasi | MAE pada 5 tahun terakhir data |
| Sumber data | Historis BMKG Kabupaten Bogor |

📌 **MAE: ±114 mm** → cocok untuk prediksi trend & pola musiman.

---

## 🗂️ Project Structure

