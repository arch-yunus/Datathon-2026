<div align="center">
  <h1>🏆 Datathon 2026 - Career Success Score Predictor</h1>
  
  <p>
    <strong>Predicting Student Career Success with Deep NLP and Advanced Stacking Models</strong>
  </p>
  
  ![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
  ![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter_Optimization-blue)
  ![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-red.svg)
  ![LightGBM](https://img.shields.io/badge/LightGBM-Gradient_Boosting-green)
  ![CatBoost](https://img.shields.io/badge/CatBoost-Gradient_Boosting-yellow)
  ![MLflow](https://img.shields.io/badge/MLflow-Experiment_Tracking-blueviolet)

</div>

---

## 📖 Proje Özeti
Bu depo, **Datathon 2026** yarışması kapsamında öğrencilerin profil verilerinden yola çıkarak `career_success_score` (Kariyer Başarı Skoru) tahminlemesini yapan **gelişmiş bir makine öğrenmesi boru hattını (pipeline)** içerir. 

Proje, basit bir regresyon modelinden başlayarak **Derin NLP özellik çıkarımına (SBERT+PCA)** ve nihayetinde yapay zeka meta-modellerini kullanan bir **RidgeCV Stacking** mimarisine kadar uzanmaktadır. Projemizin ulaştığı en düşük hata skoru **OOF MSE: 80.80**'dir.

---

## 🚀 Öne Çıkan Özellikler (Mimarimiz)

### 1. Gelişmiş Özellik Mühendisliği (Feature Engineering)
- **Eksik Veri Tespiti**: Matematiksel ve kategorik eksiklikler stratejik olarak doldurulmuş ve yapay zekaya "veri eksikliği" durumları yeni özellikler (missing indicators) olarak öğretilmiştir.
- **Etkileşim Değişkenleri**: Mülakat skorları toplamı (`total_interview_score`), bölüm bazında not ortalaması oranı (`cgpa_dept_ratio`) gibi yarışma dinamiklerine uygun yeni matematiksel değişkenler eklendi.

### 2. Derin NLP (Doğal Dil İşleme)
- Sadece kelime sayımına (TF-IDF) dayalı ilkel yöntemler yerine, öğrencilerin **`mentor_feedback_text`** metinleri `sentence-transformers` (all-MiniLM-L6-v2) kullanılarak çok boyutlu anlam vektörlerine dönüştürülmüştür.
- Ağaç modellerinin (LightGBM vb.) hafızasını yormamak için bu vektörler **PCA ile 32 boyuta** indirgenip doğrudan ana veri setine kaynaştırılmıştır.

### 3. Çoklu Model Çeşitliliği & Optuna ile Tuning
Sisteme birbirinin zayıflığını örten 4 farklı ana model entegre edilmiştir:
- **LightGBM** (Hızlı ve yüksek doğruluk)
- **CatBoost** (Kategorik veri canavarı)
- **XGBoost** (Klasik ve sağlam gradyan artırma)
- **PyTorch Neural Network** (Derin sinir ağı ile çizgisel olmayan yaklaşımlar)

Modeller standart parametrelerle değil, **Optuna** kütüphanesi ile otomatik hiperparametre taraması (Hyperparameter Tuning) yapılarak eğitilir. En iyi parametreler otomatik olarak `best_params.json` dosyasında tutulur.

### 4. Meta-Model Stacking (Yapay Zeka ile Ensemble)
Klasik "ortalama alma" veya doğrusal ağırlıklandırma yöntemlerini çöpe attık! Modellerimizin 5-Fold Cross Validation ile ürettikleri *Out-Of-Fold (OOF)* tahminleri, en tepedeki ikinci aşama **RidgeCV Meta-Model'e** girdi olarak verilir. Bu sistem, hangi modelin hangi tür verilerde daha çok hata yaptığını öğrenerek nihai `final_submission_stacked.csv` sonucunu kusursuz bir uyumla üretir.

---

## ⚙️ Kurulum ve Çalıştırma

### 1. Ortam Kurulumu
Windows PowerShell kullanarak sanal ortamınızı oluşturun ve gereksinimleri yükleyin:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Tek Tuşla Uçtan Uca Eğitim
Tüm veriyi işleyen, NLP gömülerini çıkaran, modelleri ayarlayan ve nihai *Stacked* tahmin dosyasını oluşturan orkestrasyon betiğini çalıştırın:
```powershell
python src/run_all_models.py
```
*Bu komut sırasıyla şunları yapar:*
1. `src/feature_engineering.py` çalıştırır (NLP vektörlerini çıkarır).
2. `src/tune_all.py` ile Optuna parametre taramasını yapar.
3. LightGBM, CatBoost, XGBoost ve NN modellerini 5-Fold CV ile eğitir.
4. `src/stacking_advanced.py` ile tahminleri meta-model'de birleştirir.

---

## 📂 Dizin Yapısı (Directory Structure)

```text
Datathon-2026/
├── data/
│   ├── train.csv                 # Orijinal eğitim verisi
│   ├── test_x.csv                # Orijinal test verisi
│   └── train_fe.csv / test_fe.csv # Üretilen zenginleştirilmiş veriler
├── src/
│   ├── feature_engineering.py    # Feature işlemleri ve SBERT Embedding
│   ├── tune_all.py               # Optuna hiperparametre optimizasyonu
│   ├── run_pipeline.py           # LightGBM eğitimi (MLflow destekli)
│   ├── run_catboost.py           # CatBoost eğitimi
│   ├── run_xgboost.py            # XGBoost eğitimi
│   ├── run_nn.py                 # PyTorch sinir ağı eğitimi
│   └── stacking_advanced.py      # RidgeCV meta-model orkestrasyonu
├── notebooks/                    # Keşifsel Veri Analizi (EDA) çalışmaları
├── mlruns/                       # MLflow model izleme kayıtları
└── README.md                     # Bu dosya
```

---

## 📊 Performans Takibi (MLflow)
Yaptığınız tüm eğitimler, kaydettiğiniz tüm metrikler ve model çıktıları MLflow tarafından loglanmaktadır. Performans karşılaştırması yapmak için terminalinize şunu yazın:
```bash
mlflow ui
```
Tarayıcınızda açılacak `http://127.0.0.1:5000` adresi üzerinden modelleri kıyaslayabilirsiniz.

---

## ✉️ İletişim & Gönderim (Kaggle)
Oluşturulan `final_submission_stacked.csv` dosyasını doğrudan Kaggle'a veya değerlendirme platformuna yükleyebilirsiniz. Eğer API kullanıyorsanız:
```bash
kaggle competitions submit -c <competition-name> -f final_submission_stacked.csv -m "Advanced NLP + RidgeCV Stack"
```

> **Başarılar!** Bu repo, makine öğrenmesi mühendisliği en iyi pratikleri (best practices) kullanılarak yarışma skorlarını zirveye taşımak üzere mimarilendirilmiştir.


