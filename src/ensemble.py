import pandas as pd
import numpy as np
from scipy.optimize import minimize

def rmse_objective(weights, oof_preds, true_target):
    weights = np.array(weights)
    blend_pred = np.zeros(len(true_target))
    for w, pred in zip(weights, oof_preds):
        blend_pred += w * pred
    return np.mean((true_target - blend_pred) ** 2)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--subs', nargs='*', default=['submission.csv','submission_embeddings.csv','submission_fe.csv','submission_xgb.csv','submission_nn.csv'])
    parser.add_argument('--oofs', nargs='*', default=['oof_predictions.csv','oof_embeddings.csv','oof_predictions.csv','oof_predictions_xgb.csv','oof_predictions_nn.csv'])
    parser.add_argument('--out', default='final_submission.csv')
    args = parser.parse_args()

    available_subs = []
    available_oofs = []

    for i, s in enumerate(args.subs):
        try:
            df = pd.read_csv(s)
            available_subs.append((s, df))
            
            oof_file = args.oofs[i] if i < len(args.oofs) else None
            if oof_file:
                try:
                    oof_df = pd.read_csv(oof_file)
                    available_oofs.append((oof_file, oof_df))
                except Exception:
                    pass
        except Exception:
            pass

    if not available_subs:
        raise SystemExit('No submission files found')

    print(f"Loaded {len(available_subs)} submission files.")

    # Try to optimize weights if OOFs are available
    weights = None
    if len(available_oofs) == len(available_subs):
        base_oof = available_oofs[0][1]
        if 'career_success_score' in base_oof.columns and 'oof_pred' in base_oof.columns:
            true_target = base_oof.set_index('student_id')['career_success_score']
            
            oof_preds_list = []
            for oof_name, oof_df in available_oofs:
                aligned_pred = oof_df.set_index('student_id').reindex(true_target.index)['oof_pred']
                oof_preds_list.append(aligned_pred.values)
                
            true_target_vals = true_target.values
            
            # initial equal weights
            init_weights = [1.0 / len(oof_preds_list)] * len(oof_preds_list)
            
            bounds = [(0, 1) for _ in range(len(oof_preds_list))]
            cons = {'type': 'eq', 'fun': lambda w: 1.0 - np.sum(w)}
            
            print("Optimizing blending weights...")
            res = minimize(rmse_objective, init_weights, args=(oof_preds_list, true_target_vals),
                           method='SLSQP', bounds=bounds, constraints=cons)
            
            if res.success:
                weights = res.x
                print("Optimal weights found:", weights)
                print(f"Optimized OOF MSE: {res.fun:.4f}")
            else:
                print("Optimization failed. Using equal weights.")
    
    if weights is None:
        weights = [1.0 / len(available_subs)] * len(available_subs)
        print("Using equal weights:", weights)

    # blend submissions
    base = available_subs[0][1][['student_id']].copy()
    preds = np.zeros(len(base))

    for (sname, sdf), w in zip(available_subs, weights):
        sdf = sdf.set_index('student_id')
        aligned = sdf.reindex(base['student_id']).reset_index()
        preds += aligned['career_success_score'].values * w

    out = base.copy()
    out['career_success_score'] = preds
    out.to_csv(args.out, index=False)
    print('Saved optimized ensemble to', args.out)

