# Datathon 2026 - Kariyer Başarı Skoru Tahmin Yarışması

Bu repo, Datathon 2026 için yarışma çözüm çalışmaları ve referans kodlarını içerecek şekilde düzenlenmek üzere hazırlanmıştır.

**Project Overview**
- **Amaç**: Öğrencilerin profil verilerine göre `career_success_score` (0-100) tahmin etmek.
- **Veri Türleri**: Sayısal, kategorik ve doğal dil (mentor_feedback_text).
- **Değerlendirme**: Mean Squared Error (MSE).

**Repository Structure**
- **Data**: Veri dosyaları kök dizinde bulunur: [train.csv](train.csv), [test.csv](test.csv), [sample_submission.csv](sample_submission.csv)
- **Notebooks**: Deneysel Jupyter notebook'ları `notebooks/` klasöründe.
- **Src**: Eğitim ve tahmin betikleri `src/` klasöründe (`train.py`, `predict.py`, `features.py`).
- **README**: Bu dosya `README.md`.

**Datasets**
- **train.csv**: Eğitim verisi ve hedef `career_success_score` içerir.
- **test.csv**: Tahmin yapılacak veri.
- **sample_submission.csv**: Gönderim formatı.

**How to Run**
- **Sanal ortam oluştur**: `python -m venv .venv` ve `pip install -r requirements.txt`.
- **Eğitim** (örnek): `python src/train.py --data train.csv --out model.pkl`.
- **Tahmin** (örnek): `python src/predict.py --model model.pkl --test test.csv --out submission.csv`.

**Quickstart (one-line)**

1) Tüm bağımlılıkları kurup çalıştırmak için (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
.
# CV ile eğitim ve submission üretimi
python src/run_pipeline.py --train train.csv --test test_x.csv --out submission.csv --model model.pkl --cv 5
```

2) Alternatif: hızlıca yalnızca eğitim ve kayıt için

```bash
python src/train.py --data train.csv --out model.pkl
python src/predict.py --model model.pkl --test test_x.csv --out submission.csv
```

**Modeling Notes**
- **Feature Engineering**: Sayısal özetler, kategorik encoding, mentor_feedback_text için TF-IDF veya gömülü (embedding) kullanın.
- **NLP**: `mentor_feedback_text` alanı önemli; basit TF-IDF + LightGBM veya transformer tabanlı gömülerle deneyin.
- **Validation**: Zaman bazlı veya stratified CV yerine gruplama varsa grup CV önerilir.
- **Baseline**: Önce basit regresyon/LightGBM ile baseline oluşturun, sonra NLP katkısını ölçün.

**Evaluation**
- **Metric**: MSE (düşük MSE daha iyi).
- **Submission**: Üretilen `submission.csv` formatı `sample_submission.csv` ile aynı olmalıdır.

**Submission & Notebook Sharing**
- **Kaggle**: Final submit için `submission.csv` yükleyin.
- **İlk 10 için**: Eğer ilk 10'a kalırsanız, notebook veya kodlarınızı paylaşmanız zorunludur.

**Competition Rules & Eligibility**
- **Kayıt**: Yarışmaya katılabilir olmak için BTK Akademi üzerinden kayıt zorunludur.
- **Davranış**: Birden fazla hesap, özel kod paylaşımı vb. kurallara dikkat edin (yarışma kuralları geçerlidir).

**İletişim / İleri Adımlar**
- **Dosya oluşturuldu**: Bu README dosyası köke eklendi.
- **Next**: İsterseniz `src/` için temel `train.py` ve `predict.py` oluşturayım veya notebook şablonu ekleyeyim.

---

**Detailed Quickstart & Workflow**

- Clone the repo and open the folder.
- Prepare virtual environment and install dependencies (see `requirements.txt`).
- Run `src/feature_engineering.py` to generate feature-engineered CSVs (`train_fe.csv`, `test_fe.csv`) or use raw CSVs.
- Choose a pipeline: baseline (`src/run_pipeline.py`) or embedding (`src/run_with_embeddings.py`).
- Run CV training to produce model file(s) and submission.

Example (PowerShell):

```powershell
# feature engineering
python src/feature_engineering.py --train train.csv --test test_x.csv --out-train train_fe.csv --out-test test_fe.csv

# baseline CV + submission
python src/run_pipeline.py --train train_fe.csv --test test_fe.csv --out submission_fe.csv --model model_fe.pkl --cv 5

# embeddings-based pipeline (may download models)
python src/run_with_embeddings.py --train train.csv --test test_x.csv --out submission_embeddings.csv --model model_embeddings.pkl --cv 3
```

**Experiment Tracking & Results**
- Baseline 5-fold CV (example run): OOF MSE ≈ 91.5761
- Embedding (sentence-transformers) 3-fold CV (example run): OOF MSE ≈ 88.2508
- Feature-engineered baseline 5-fold CV (example run): OOF MSE ≈ 92.1856

These example scores were produced during development runs in this repo — use them as a starting point and improve via tuning and ensembling.

**Ensembling**
- A simple ensemble script is provided at `src/ensemble.py` which averages submissions weighted by inverse OOF MSEs. Use it to combine `submission.csv`, `submission_embeddings.csv`, and `submission_fe.csv` into `final_submission.csv`.

**Kaggle Submission**
- Create an account and join the competition on Kaggle.
- Prepare `final_submission.csv` (student_id, career_success_score) and upload on the competition page.
- To automate: install `kaggle` CLI and set `KAGGLE_USERNAME` and `KAGGLE_KEY` in your environment, then run:

```bash
kaggle competitions submit -c <competition-name> -f final_submission.csv -m "My submission"
```

**Next Steps & Suggestions**
- Run full Optuna tuning (`src/tune_optuna.py`) for longer (e.g., 100 trials) to improve LightGBM hyperparameters.
- Replace TF-IDF with `sentence-transformers` embeddings (already included) for mentor feedback text, then re-run CV.
- Try stacking/ensembling multiple diverse models (LightGBM, CatBoost, simple neural nets).
- Add logging and MLflow or Weights & Biases for experiment tracking.

**Files of Interest**
- `src/run_pipeline.py` — baseline end-to-end pipeline (preprocess + LightGBM)
- `src/run_with_embeddings.py` — builds sentence-transformers embeddings and trains a model
- `src/tune_optuna.py` — Optuna tuning script
- `src/feature_engineering.py` — feature engineering helpers and CLI
- `src/ensemble.py` — combines multiple submission files into `final_submission.csv`
- `notebooks/` — interactive notebooks to reproduce experiments

**Contact / Attribution**
- Repo prepared for Datathon 2026. If you reuse code, please credit the team and follow the competition rules.

--
Bu README güncellendi ve uzak repoya gönderilecek.

