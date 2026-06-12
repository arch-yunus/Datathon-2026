import subprocess
import sys

def run_cmd(cmd):
    print(f"==================================================")
    print(f"Running: {cmd}")
    print(f"==================================================")
    process = subprocess.Popen(cmd, shell=True)
    process.communicate()
    if process.returncode != 0:
        print(f"Error executing {cmd}")
        sys.exit(1)

commands = [
    # 1. Feature Engineering with NLP Embeddings
    f"{sys.executable} src/feature_engineering.py --train train.csv --test test_x.csv --out-train train_fe.csv --out-test test_fe.csv",
    
    # 2. Hyperparameter Tuning (Fast mode for testing: 10 trials)
    f"{sys.executable} src/tune_all.py --train train_fe.csv --out best_params.json --trials 10",
    
    # 3. Train Models (using the engineered features including NLP)
    f"{sys.executable} src/run_pipeline.py --train train_fe.csv --test test_fe.csv --out submission.csv --model model.pkl --cv 5",
    f"{sys.executable} src/run_catboost.py --train train_fe.csv --test test_fe.csv --out submission_catboost.csv --model model_catboost.pkl --cv 5",
    f"{sys.executable} src/run_xgboost.py --train train_fe.csv --test test_fe.csv --out submission_xgb.csv --model model_xgb.pkl --cv 5",
    f"{sys.executable} src/run_nn.py --train train_fe.csv --test test_fe.csv --out submission_nn.csv --model model_nn.pt --cv 5",
    f"{sys.executable} src/run_with_embeddings.py --train train.csv --test test_x.csv --out submission_embeddings.csv --model model_embeddings.pkl --cv 5",

    # 4. Advanced Stacking
    f"{sys.executable} src/stacking_advanced.py --subs submission.csv submission_catboost.csv submission_xgb.csv submission_nn.csv submission_embeddings.csv --oofs oof_predictions.csv oof_catboost.csv oof_predictions_xgb.csv oof_predictions_nn.csv oof_embeddings.csv --out final_submission_stacked.csv"
]

for c in commands:
    run_cmd(c)

print("All models trained and ensembled successfully! 'final_submission.csv' is ready.")
