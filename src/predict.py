import argparse
import pandas as pd
import joblib


def main(args):
    df = pd.read_csv(args.test)
    model = joblib.load(args.model)

    X = df.drop(columns=['student_id'], errors='ignore')
    preds = model.predict(X)

    submission = pd.DataFrame({'student_id': df['student_id'], 'career_success_score': preds})
    submission.to_csv(args.out, index=False)
    print('Submission saved to', args.out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--test', required=True)
    parser.add_argument('--out', default='submission.csv')
    args = parser.parse_args()
    main(args)
