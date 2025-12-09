

---

```markdown
# 🌧️ Bogor Rainfall Forecasting App  
**Machine Learning for Climate Risk & Operational Decision Support**

Aplikasi interaktif untuk **memprediksi curah hujan bulanan hingga 10 tahun ke depan** menggunakan model **Random Forest Regressor**, berbasis data historis BMKG Kabupaten Bogor.

🔗 **Live Demo App:**  
https://rainfall-forecast-app-sygttgiqfy3aehbhoyefvr.streamlit.app/

📦 **Source Code Repository:**  
https://github.com/ilhmhfdz/rainfall-forecast-app

---

## 🚀 Features

| Feature | Description |
|--------|-------------|
| 🔮 **10-year ML Forecast** | Prediksi curah hujan bulanan secara otomatis |
| 📊 **Interactive Visualization** | Plotly charts — zoom, hover, drill-down |
| 🎯 **Year Focus Selection** | Pilih tahun tertentu untuk analisis detail |
| 📌 **AI-rule Insight** | Insight otomatis untuk pengambilan keputusan |
| 🌗 **Dark & Light Mode** | Custom chart theme untuk user experience |
| ⚙️ **Optimized Model** | RandomForest + seasonal engineered features (sin/cos) |

---

## 🎯 Problem Background

Perubahan iklim meningkatkan risiko **banjir**, **keterlambatan proyek konstruksi**, dan **gangguan operasional**.  
Dengan prediksi curah hujan, stakeholder dapat:

✔ Merencanakan jadwal konstruksi  
✔ Antisipasi mitigasi risiko banjir  
✔ Perencanaan industri pertanian & logistik  

---

## 🧠 Machine Learning Approach

| Aspect | Method |
|-------|--------|
| Model | RandomForestRegressor |
| Target | Curah hujan per bulan (mm) |
| Input Features | 12 bulan lag, month_sin, month_cos |
| Evaluation | MAE pada 5 tahun terakhir |
| Data Source | BMKG dataset historis Kabupaten Bogor |

**MAE (5 tahun terakhir): ~114 mm**

> Forecast dilakukan secara **iteratif (autoregressive)** agar dapat menjangkau 10 tahun ke depan.

---

## 🗂️ Project Structure

```

📁 rainfall-forecast-app
│
├── app.py                # Streamlit Interface
├── train_model.py        # Training script & export model
├── forecast_utils.py     # Forecasting logic (if needed future scaling)
├── requirements.txt      # Dependencies
│
├── data/
│   └── rainfall_bogor.csv
└── models/
└── rainfall_rf.joblib  # Saved RandomForest model

````

---

## 🎛️ Tech Stack

- **Python 3.9+**
- **Machine Learning:** scikit-learn, numpy, pandas
- **Visualization:** Plotly
- **Deployment:** Streamlit Cloud
- **Versioning:** Git + GitHub

---

## 🧪 How to Run Locally

```bash
git clone https://github.com/ilhmhfdz/rainfall-forecast-app
cd rainfall-forecast-app

# Create env (optional)
python3 -m venv .venv
source .venv/bin/activate  # MacOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run App
streamlit run app.py
````

---

## 🌍 Future Improvements

| Planned Feature                  | Purpose                           |
| -------------------------------- | --------------------------------- |
| LSTM / Prophet Model             | Better long-term climate forecast |
| Downloadable Reports             | PDF/Excel for stakeholders        |
| Multi-city support with BMKG API | Nationwide impact                 |
| Real AI Insight (LLM-based)      | Deeper contextual analysis        |

---

## 👨‍💻 Author

**Ilham Hafidz**
Aspiring Data Scientist — Machine Learning Enthusiast
📧 Email: *(isi email kamu di sini)*
🔗 LinkedIn: *(insert link linkedin kamu di sini)*
🌐 Portfolio: *(optional kalau ada)*

---

### ⭐ If you like this project, please give it a star on GitHub!

---

```

---

Kalau kamu mau, gue **bisa tambahin juga**:

✨ Badge GitHub, Model Performance Card  
📌 Screenshot UI biar visual menarik  
🧠 Penjelasan model tuning & reasoning  
📍 Section “Use Case in Real Industries”

---

Kalau kamu setuju, sekalian gua bantu bikinin **slide presentasi singkat** buat HR nanti (pitch 1–2 menit) 🔥

Would you like me to also help:  
✔ Update README langsung ke repo kamu?  
✔ Buat section Experience buat di CV kamu?  
✔ Optimasi LinkedIn biar cocok ke Data Scientist Intern?

Ayo kita gaskeun biar HR langsung *kepincut* portfolio kamu 😎🚀
```
