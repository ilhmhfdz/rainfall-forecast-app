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



## 🛠️ Tech Stack

- Python (NumPy, Pandas)
- Scikit-learn (Random Forest)
- Plotly (Visualization)
- Streamlit (Deployment)
- GitHub (Version Control)

---

## 🧪 How to Run Locally

```bash
git clone https://github.com/ilhmhfdz/rainfall-forecast-app
cd rainfall-forecast-app

python3 -m venv .venv
source .venv/bin/activate  # MacOS/Linux
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt
streamlit run app.py

📌 Future Improvements

RNN/LSTM atau Prophet untuk forecast lebih akurat

Insight berbasis LLM (OpenAI / local model)

Download hasil dalam format PDF/Excel

API BMKG untuk multi-kota Indonesia

---

👨‍💻 Author — Ilham Hafidz

Aspiring Data Scientist | Machine Learning Enthusiast
📍 Indonesia
📧 Email: ilhamhafidz666@gmail.com


