from pathlib import Path
import polars as pl
from powergrid.data import generate_and_cache

def test_generate_and_cache_creates_files(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir_str = str(cache_dir)

    # Génération rapide d'un petit dataset
    df_features, df_targets = generate_and_cache(cache_dir=cache_dir_str, episode_count=1, n_actions=5, force=True)

    fpath = Path(cache_dir) / "df_features.parquet"
    tpath = Path(cache_dir) / "df_targets.parquet"

    # Vérifications
    assert fpath.exists(), "Le fichier df_features.parquet n'a pas été créé."
    assert tpath.exists(), "Le fichier df_targets.parquet n'a pas été créé."
    assert isinstance(df_features, pl.DataFrame)
    assert isinstance(df_targets, pl.DataFrame)
    assert df_features.shape[0] == df_targets.shape[0]

def test_cache_reuse_same_data(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir_str = str(cache_dir)

    # Première génération
    features1, targets1 = generate_and_cache(cache_dir=cache_dir_str, episode_count=1, n_actions=5, force=True)
    # Rechargement depuis le cache
    features2, targets2 = generate_and_cache(cache_dir=cache_dir_str, episode_count=1, n_actions=5, force=False)

    assert features1.shape == features2.shape
    assert targets1.shape == targets2.shape
