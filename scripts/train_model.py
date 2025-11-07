import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from powergrid.model import RandomForestPowerGrid

def load_data(features_path, targets_path):
    """Charge directement les fichiers df_features et df_targets."""

    print("📥 Chargement des données...")
    df_features = pd.read_parquet(features_path)
    df_targets = pd.read_parquet(targets_path)

    print(f"✅ Features : {df_features.shape}, Targets : {df_targets.shape}")
    return df_features, df_targets


def clean_targets(y_train, y_test):
    """Replace inf targets with max valid + 0.1."""
    y_train = y_train.copy()
    y_test = y_test.copy()
    max_train = np.max(y_train[np.isfinite(y_train)])
    max_test  = np.max(y_test[np.isfinite(y_test)])
    y_train[~np.isfinite(y_train)] = max_train + 0.1
    y_test[~np.isfinite(y_test)]   = max_test + 0.1
    return y_train, y_test

def main():
    parser = argparse.ArgumentParser(description="Train RandomForest on Grid2Op data")
    parser.add_argument("--features", type=str, default="data/cache/df_features.parquet")
    parser.add_argument("--targets", type=str, default="data/cache/df_targets.parquet")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    # Load full data
    df_features, df_targets = load_data(args.features, args.targets)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        df_features, df_targets, test_size=args.test_size, random_state=args.random_state
    )

    # Clean inf targets
    y_train, y_test = clean_targets(y_train.values, y_test.values)

    # Initialize model
    model = RandomForestPowerGrid()

    # Fit
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    pd.DataFrame(y_pred).to_parquet("results/y_pred.parquet")
    pd.DataFrame(y_test).to_parquet("results/y_true.parquet")

    # Evaluate
    rmse = model.rmse(y_test, y_pred)
    print("Global RMSE:", rmse)

    # Feature importances
    importances = model.feature_importances(X_train)
    plt.figure(figsize=(10,6))
    importances.nlargest(20).plot(kind="barh")
    plt.title("Top 20 Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
