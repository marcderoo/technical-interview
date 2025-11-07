import numpy as np
from powergrid.data import generate_and_cache
from powergrid.model import RandomForestPowerGrid
from sklearn.model_selection import train_test_split

def test_random_forest_training(tmp_path):
    # Génération d'un petit dataset pour test rapide
    features, targets = generate_and_cache(cache_dir=str(tmp_path / "cache"), episode_count=1, n_actions=5, force=True)
    X = features.to_pandas()
    y = targets.to_pandas()

    # Nettoyage rapide des inf
    y = y.replace([np.inf, -np.inf], np.nan)
    y = y.fillna(y.max().max() + 0.1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestPowerGrid(n_estimators=10, random_state=0, n_jobs=1)
    y_train_clean, y_test_clean = model.clean_targets(y_train.values, y_test.values)

    # Entraînement
    model.fit(X_train, y_train_clean)
    preds = model.predict(X_test)

    # Validation
    rmse = model.rmse(y_test_clean, preds)
    assert np.isfinite(rmse), "RMSE non fini."
    assert rmse >= 0.0
    assert preds.shape == y_test_clean.shape
