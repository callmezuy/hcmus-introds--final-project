import os
import argparse
import sys
import joblib
import pandas as pd

# Ensure UTF-8 output on Windows
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

# Import builder from service
from service.model_service_wrapper import build_model_input


def main():
    parser = argparse.ArgumentParser(description='Run model prediction on 50-row input')
    parser.add_argument('--symbol', type=str, default='VNM', help='Stock symbol (e.g., VNM)')
    parser.add_argument('--server_url', type=str, default='http://127.0.0.1:8000', help='Server URL hosting the /model-input endpoint')
    parser.add_argument('--days', type=int, default=60, help='Past days to fetch to ensure 50 rows')
    parser.add_argument('--source', type=str, default='VNStock', choices=['VNStock', 'local'], help='Data source for input building')
    args = parser.parse_args()

    # Build input DataFrame
    df_input: pd.DataFrame = build_model_input(
        symbol=args.symbol,
        server_url=args.server_url,
        days=args.days,
        source=args.source,
    )

    if df_input is None or len(df_input) != 50:
        print(f"Input not ready: expected 50 rows, got {len(df_input) if df_input is not None else 0}")
        sys.exit(1)

    # Load model pipeline
    here = os.path.dirname(__file__)
    model_path = os.path.join(here, 'model', 'best_model.pkl')
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        sys.exit(1)

    pipeline = joblib.load(model_path)

    # Predict and predict_proba
    try:
        y_pred = pipeline.predict(df_input)
    except Exception as e:
        print(f"Error in predict: {e}")
        raise

    try:
        y_prob = pipeline.predict_proba(df_input)
    except Exception:
        y_prob = None

    # Print results for the last (newest) day
    last_row = df_input.iloc[-1]
    lines = []
    lines.append("=== Prediction Result ===")
    lines.append(f"Symbol: {last_row['symbol']}")
    lines.append(f"Date: {last_row['time']}")
    lines.append(f"Predicted label: {y_pred[0] if len(y_pred) > 0 else y_pred}")
    if y_prob is not None:
        try:
            lines.append(f"Buy probability (class 1): {y_prob[0,1]:.4f}")
        except Exception:
            lines.append(f"Probabilities: {y_prob}")

    lines.append("\nInput sample (last 5 rows):")
    lines.append(df_input.tail(5).to_string(index=False))

    out_text = "\n".join(lines)
    print(out_text)

    # Also write to file for inspection
    out_path = os.path.join(here, 'prediction_output.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(out_text)


if __name__ == '__main__':
    main()
