from __future__ import annotations
import os
import numpy as np
import polars as pl
from tqdm import tqdm
from pathlib import Path
import grid2op
from grid2op.Observation import BaseObservation
from grid2op.Action import BaseAction
from grid2op.Environment import Environment
from lightsim2grid import LightSimBackend


def extract_features(obs: BaseObservation) -> pl.DataFrame:
    """
    Transforme une observation du réseau en DataFrame Polars avec noms de colonnes explicites.
    """
    feature_names = (
        [f"gen_p_{i}" for i in range(len(obs.gen_p))] +
        [f"gen_q_{i}" for i in range(len(obs.gen_q))] +
        [f"load_p_{i}" for i in range(len(obs.load_p))] +
        [f"load_q_{i}" for i in range(len(obs.load_q))] +
        [f"topo_{i}" for i in range(len(obs.topo_vect))] +
        [f"rho_{i}" for i in range(len(obs.rho))]
    )

    feature_array = np.concatenate([
        obs.gen_p,
        obs.gen_q,
        obs.load_p,
        obs.load_q,
        obs.topo_vect,
        obs.rho
    ])

    return pl.DataFrame([feature_array], schema=feature_names, orient="row")


def create_realistic_observation(episode_count: int, env: Environment) -> list[BaseObservation]:
    """Crée une liste d'observations réalistes avec actions 'rien faire'."""
    list_obs = []
    for _ in tqdm(range(episode_count), desc="⚙️ Simulating episodes"):
        obs = env.reset()
        list_obs.append(obs)
        for _ in range(env.chronics_handler.max_timestep()):
            obs, reward, done, info = env.step(env.action_space())  # action = "do nothing"
            if done:
                break
            list_obs.append(obs)
    return list_obs


def create_training_data(list_obs: list[BaseObservation], all_actions: list[BaseAction]):
    """Crée features et targets pour ML"""
    df_features = []
    df_targets = []

    for obs in tqdm(list_obs, desc="⚙️ Creating training data"):
        action_score = []
        simulator = obs.get_simulator()
        for act in all_actions:
            sim_after_act = simulator.predict(act=act)
            n_obs = sim_after_act.current_obs
            action_score.append(n_obs.rho.max() if sim_after_act.converged else np.inf)
        df_targets.append(action_score)
        df_features.append(extract_features(obs))

    df_features = pl.concat(df_features)
    df_targets = pl.DataFrame(df_targets).transpose()
    df_targets.columns = [f"action_{i}" for i in range(df_targets.width)]

    return df_features, df_targets


def generate_and_cache(
    cache_dir: str = "data/cache",
    episode_count: int = 2,
    n_actions: int = 100,
    force: bool = False
):
    """
    Génère les features et targets avec cache local.
    Si les fichiers existent déjà, les recharge.
    """
    os.makedirs(cache_dir, exist_ok=True)
    fpath = Path(cache_dir) / "df_features.parquet"
    tpath = Path(cache_dir) / "df_targets.parquet"

    if not force and fpath.exists() and tpath.exists():
        print("✅ Using cached dataset")
        df_features = pl.read_parquet(fpath)
        df_targets = pl.read_parquet(tpath)
        print("Dimensions features:", df_features.shape)
        print("Dimensions targets :", df_targets.shape)
        return df_features, df_targets

    print("⚙️ Generating new dataset...")
    env = grid2op.make("l2rpn_case14_sandbox", backend=LightSimBackend(), n_busbar=3)

    all_actions = [env.action_space.sample() for _ in range(n_actions)]
    all_actions.append(env.action_space())  # "do nothing" action

    list_obs = create_realistic_observation(episode_count, env)
    df_features, df_targets = create_training_data(list_obs, all_actions)

    print("Dimensions features:", df_features.shape)
    print("Dimensions targets :", df_targets.shape)
    df_features.write_parquet(fpath)
    df_targets.write_parquet(tpath)
    print(f"💾 Dataset cached at {cache_dir}")

    return df_features, df_targets
