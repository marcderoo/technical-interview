import argparse
import pandas as pd
import numpy as np
from powergrid.viz import plot_prediction_errors

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=str, default="results/y_true.parquet")
    parser.add_argument("--predictions", type=str, default="results/y_pred.parquet")
    args = parser.parse_args()

    # Load true targets and predictions
    y_true = pd.read_parquet(args.targets)
    y_pred = pd.read_parquet(args.predictions)

    # Plot errors
    plot_prediction_errors(y_true, y_pred)

if __name__ == "__main__":
    main()
