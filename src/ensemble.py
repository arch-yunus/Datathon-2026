import pandas as pd
import numpy as np

# Simple ensemble: load available submission files and OOFs to compute weights
subs = []
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--subs', nargs='*', default=['submission.csv','submission_embeddings.csv','submission_fe.csv'])
    parser.add_argument('--oofs', nargs='*', default=['oof_predictions.csv','oof_embeddings.csv','oof_embeddings.csv'])
    parser.add_argument('--out', default='final_submission.csv')
    args = parser.parse_args()

    available = []
    for s in args.subs:
        try:
            df = pd.read_csv(s)
            available.append((s, df))
        except Exception:
            pass

    if not available:
        raise SystemExit('No submission files found')

    # compute weights from OOF MSE if possible
    mses = []
    for i, (sname, sdf) in enumerate(available):
        oof_file = args.oofs[i] if i < len(args.oofs) else None
        if oof_file:
            try:
                oof = pd.read_csv(oof_file)
                if 'career_success_score' in oof.columns and 'oof_pred' in oof.columns:
                    mse = ((oof['career_success_score'] - oof['oof_pred'])**2).mean()
                    mses.append(mse)
                    continue
            except Exception:
                pass
        # fallback large mse
        mses.append(1e6)

    inv = [1.0/m if m>0 else 0.0 for m in mses]
    total = sum(inv)
    weights = [v/total for v in inv]

    print('Using weights:', weights)

    # align by student_id
    base = available[0][1][['student_id']].copy()
    preds = np.zeros(len(base))

    for (sname, sdf), w in zip(available, weights):
        sdf = sdf.set_index('student_id')
        aligned = sdf.reindex(base['student_id']).reset_index()
        preds += aligned['career_success_score'].values * w

    out = base.copy()
    out['career_success_score'] = preds
    out.to_csv(args.out, index=False)
    print('Saved ensemble to', args.out)
