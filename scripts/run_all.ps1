# Run full pipeline: install deps, run CV training and produce submission
python -m pip install --upgrade pip
pip install -r requirements.txt
python src/run_pipeline.py --train train.csv --test test_x.csv --out submission.csv --model model.pkl --cv 5
