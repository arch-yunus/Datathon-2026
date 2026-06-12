"""
End-to-end training + CV + submission script with PyTorch MLP.
Usage:
  python src/run_nn.py --train train.csv --test test_x.csv --out submission_nn.csv --model model_nn.pt --cv 5
"""
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import os
import mlflow

class MLPModel(nn.Module):
    def __init__(self, input_dim):
        super(MLPModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        return self.net(x).squeeze()

def train_nn_model(X_tr, y_tr, X_val, y_val, input_dim, epochs=50, batch_size=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tr_dataset = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    
    tr_loader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = MLPModel(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_model_state = None
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in tr_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                val_losses.append(loss.item())
        
        avg_val_loss = np.mean(val_losses)
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            
    model.load_state_dict(best_model_state)
    return model, best_val_loss

def run_cv_and_submit(train_path, test_path, out_submission, model_out, cv=5, random_state=42):
    mlflow.set_experiment("Datathon2026_PyTorch_NN")
    with mlflow.start_run():
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)

        if 'career_success_score' not in train.columns:
            raise ValueError('train file must contain career_success_score')

        y = train['career_success_score'].values
        X = train.drop(columns=['career_success_score', 'student_id', 'mentor_feedback_text'], errors='ignore')
        X_test = test.drop(columns=['student_id', 'mentor_feedback_text'], errors='ignore')

        num_cols = X.select_dtypes(include=[np.number]).columns
        cat_cols = X.select_dtypes(exclude=[np.number]).columns

        num_imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        
        cat_imputer = SimpleImputer(strategy='most_frequent')
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

        kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
        oof_preds = np.zeros(len(X))
        scores = []

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
            X_tr_raw, X_val_raw = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]

            X_tr_num = scaler.fit_transform(num_imputer.fit_transform(X_tr_raw[num_cols]))
            X_val_num = scaler.transform(num_imputer.transform(X_val_raw[num_cols]))

            X_tr_cat = np.empty((len(X_tr_raw), 0))
            X_val_cat = np.empty((len(X_val_raw), 0))
            if len(cat_cols) > 0:
                X_tr_cat = ohe.fit_transform(cat_imputer.fit_transform(X_tr_raw[cat_cols]))
                X_val_cat = ohe.transform(cat_imputer.transform(X_val_raw[cat_cols]))

            X_tr_processed = np.hstack([X_tr_num, X_tr_cat])
            X_val_processed = np.hstack([X_val_num, X_val_cat])

            model, val_loss = train_nn_model(X_tr_processed, y_tr, X_val_processed, y_val, input_dim=X_tr_processed.shape[1])
            
            model.eval()
            with torch.no_grad():
                val_pred = model(torch.tensor(X_val_processed, dtype=torch.float32).to(device)).cpu().numpy()
            
            oof_preds[val_idx] = val_pred
            mse = mean_squared_error(y_val, val_pred)
            scores.append(mse)
            print(f'Fold {fold+1} MSE: {mse:.4f}')

        mean_cv = np.mean(scores)
        std_cv = np.std(scores)
        overall_mse = mean_squared_error(y, oof_preds)
        print(f'CV mean MSE: {mean_cv:.4f} ± {std_cv:.4f}  | OOF MSE: {overall_mse:.4f}')
        
        mlflow.log_param("cv_folds", cv)
        mlflow.log_param("model", "PyTorch_MLP")
        mlflow.log_metric("cv_mean_mse", mean_cv)
        mlflow.log_metric("oof_mse", overall_mse)

        X_num = scaler.fit_transform(num_imputer.fit_transform(X[num_cols]))
        X_test_num = scaler.transform(num_imputer.transform(X_test[num_cols]))
        
        X_cat = np.empty((len(X), 0))
        X_test_cat = np.empty((len(X_test), 0))
        if len(cat_cols) > 0:
            X_cat = ohe.fit_transform(cat_imputer.fit_transform(X[cat_cols]))
            X_test_cat = ohe.transform(cat_imputer.transform(X_test[cat_cols]))
            
        X_full_processed = np.hstack([X_num, X_cat])
        X_test_processed = np.hstack([X_test_num, X_test_cat])
        
        final_model, _ = train_nn_model(X_full_processed, y, X_full_processed, y, input_dim=X_full_processed.shape[1], epochs=30)
        
        os.makedirs(os.path.dirname(model_out) or '.', exist_ok=True)
        torch.save(final_model.state_dict(), model_out)
        print('Saved NN model to', model_out)

        final_model.eval()
        with torch.no_grad():
            preds_test = final_model(torch.tensor(X_test_processed, dtype=torch.float32).to(device)).cpu().numpy()
            
        submission = pd.DataFrame({
            'student_id': test['student_id'] if 'student_id' in test.columns else np.arange(len(preds_test)),
            'career_success_score': preds_test
        })
        submission.to_csv(out_submission, index=False)
        print('Saved submission to', out_submission)

        oof_df = train[['student_id']].copy() if 'student_id' in train.columns else pd.DataFrame({'student_id': np.arange(len(oof_preds))})
        oof_df['oof_pred'] = oof_preds
        oof_df.to_csv('oof_predictions_nn.csv', index=False)
        print('Saved OOF predictions to oof_predictions_nn.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--test', default='test_x.csv')
    parser.add_argument('--out', default='submission_nn.csv')
    parser.add_argument('--model', default='model_nn.pt')
    parser.add_argument('--cv', type=int, default=5)
    args = parser.parse_args()

    run_cv_and_submit(args.train, args.test, args.out, args.model, cv=args.cv)
