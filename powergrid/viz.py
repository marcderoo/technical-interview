import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_prediction_errors(y_true: pd.DataFrame, y_pred: np.ndarray):
    """
    Affiche deux boxplots empilés pour les erreurs de prédiction par action.

    Args:
        y_true (pd.DataFrame): True rho values
        y_pred (np.ndarray): Predicted rho values
    """
    errors = y_true.to_numpy() - y_pred.to_numpy()
    n_actions = y_true.shape[1]
    action_labels = [f"action_{i}" for i in range(n_actions)]

    # Découpage en deux moitiés
    mid = n_actions // 2
    halves = [(errors[:, :mid], action_labels[:mid]), (errors[:, mid:], action_labels[mid:])]

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharey=True)

    for i, (err_half, labels_half) in enumerate(halves):
        axes[i].boxplot(err_half, labels=labels_half)
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].set_ylabel("Erreur prédiction (rho)")
        axes[i].set_title(f"Distribution des erreurs (actions {i*mid} à {(i+1)*mid - 1})")

    plt.tight_layout()
    plt.show()
