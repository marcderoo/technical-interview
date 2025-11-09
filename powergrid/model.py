import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

class RandomForestPowerGrid:
    def __init__(self, n_estimators=100, random_state=42, n_jobs=-1):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=n_jobs
        )
    
    def clean_targets(self, y_train, y_test):
        """Replace inf in targets by max_valid + 0.1"""
        max_valid = np.max(y_train[np.isfinite(y_train)])
        y_train[~np.isfinite(y_train)] = max_valid + 0.1
        max_valid = np.max(y_test[np.isfinite(y_test)])
        y_test[~np.isfinite(y_test)] = max_valid + 0.1
        return y_train, y_test

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def rmse(self, y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    def feature_importances(self, X_train):
        return pd.Series(self.model.feature_importances_, index=X_train.columns)
