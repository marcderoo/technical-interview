#!/usr/bin/env python3
"""
scripts/generate_data.py
------------------------
Génère les données pour l'entraînement :
- Simule des observations Grid2Op
- Extrait les features et targets
- Met en cache les fichiers
- Fait un split train/test
"""

import argparse
from powergrid.data import generate_and_cache


def main():
    parser = argparse.ArgumentParser(description="Generate and cache dataset for Grid2Op ML model.")
    parser.add_argument("--cache-dir", type=str, default="data/cache", help="Répertoire de cache des fichiers parquet.")
    parser.add_argument("--processed-dir", type=str, default="data/processed", help="Répertoire pour les splits train/test.")
    parser.add_argument("--episodes", type=int, default=2, help="Nombre d'épisodes simulés.")
    parser.add_argument("--actions", type=int, default=20, help="Nombre d'actions aléatoires testées par observation.")
    parser.add_argument("--force", action="store_true", help="Forcer la régénération du dataset (ignore le cache).")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proportion du jeu de test (par défaut 0.2).")
    parser.add_argument("--seed", type=int, default=42, help="Graine aléatoire pour la reproductibilité.")

    args = parser.parse_args()

    print("\n🚀 Lancement de la génération du dataset Grid2Op...")
    print(f"   - Cache dir    : {args.cache_dir}")
    print(f"   - Processed dir: {args.processed_dir}")
    print(f"   - Episodes     : {args.episodes}")
    print(f"   - Actions      : {args.actions}")
    print(f"   - Force regen  : {args.force}\n")

    # Étape 1 : génération + cache
    df_features, df_targets = generate_and_cache(
        cache_dir=args.cache_dir,
        episode_count=args.episodes,
        n_actions=args.actions,
        force=args.force
    )

    print(f"✅ Données générées : {df_features.shape[0]} observations, {df_features.shape[1]} features")
    print(f"✅ Targets générées  : {df_targets.shape[0]} lignes, {df_targets.shape[1]} actions")
    print(f"💾 Données sauvegardées dans {args.cache_dir}")

    print("\n🎯 Jeu de données prêt pour l'entraînement ou la visualisation !\n")


if __name__ == "__main__":
    main()
