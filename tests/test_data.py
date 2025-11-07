from powergrid.data import generate_and_cache

def test_features_targets_format(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir_str = str(cache_dir)

    features, targets = generate_and_cache(cache_dir=cache_dir_str, episode_count=1, n_actions=5, force=True)

    assert features.shape[0] > 0
    assert features.shape[1] > 0
    assert all(isinstance(c, str) and c for c in features.columns)
    assert all(c.startswith("action_") for c in targets.columns)

