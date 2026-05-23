"""
Orbit Wars - Nearest Planet Sniper Agent

A simple agent that captures the nearest unowned planet when it has
enough ships to guarantee the takeover.

Strategy:
  For each planet we own, find the closest planet we don't own.
  If we have more ships than the target's garrison, send exactly
  enough to capture it (garrison + 1). Otherwise, wait and accumulate.

Key concepts demonstrated:
  - Parsing the observation (planets, player ID)
  - Computing angles with atan2 for fleet direction
  - Sending moves as [from_planet_id, angle, num_ships]
"""

from __future__ import annotations

import importlib
import os
import math
import random
import sys
import types
from collections import namedtuple
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from baseline.config import TrainConfig, load_train_config
from baseline.features import TurnBatch, candidate_feature_dim, encode_turn, global_feature_dim, self_feature_dim
from baseline.policy import PlanetPolicy
from baseline.ppo import sample_actions

repo_root = Path("/kaggle_simulations/agent")
checkpoint_name = "ckpt_002000.pt"
config_name = "config.yaml"

def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_policy(cfg: TrainConfig, device: torch.device) -> PlanetPolicy:
    return PlanetPolicy(
        self_dim=self_feature_dim(),
        candidate_dim=candidate_feature_dim(),
        global_dim=global_feature_dim(),
        candidate_count=cfg.env.candidate_count,
        hidden_size=cfg.model.hidden_size,
    ).to(device)

def register_checkpoint_module_aliases() -> None:
    sys.modules.setdefault("baseline", types.ModuleType("baseline"))
    sys.modules.setdefault("baseline.rl_template", types.ModuleType("baseline.rl_template"))
    module_candidates = {
        "config": ["baseline.rl_template.config", "baseline.config", "config"],
        "features": ["baseline.rl_template.features", "baseline.features", "features"],
        "policy": ["baseline.rl_template.policy", "baseline.policy", "policy"],
        "ppo": ["baseline.rl_template.ppo", "baseline.ppo", "ppo"],
        "game_types": ["baseline.rl_template.game_types", "baseline.game_types", "game_types"],
        "opponents": ["baseline.rl_template.opponents", "baseline.opponents", "opponents"],
        "env": ["baseline.rl_template.env", "baseline.env", "env"],
        "train": ["baseline.rl_template.train", "baseline.train", "train"],
    }

    for canonical_name, candidates in module_candidates.items():
        module = None
        for candidate in candidates:
            try:
                module = importlib.import_module(candidate)
                break
            except ModuleNotFoundError:
                continue
        if module is None:
            continue
        sys.modules[f"baseline.rl_template.{canonical_name}"] = module
        sys.modules[f"baseline.{canonical_name}"] = module

def load_checkpoint_if_available(policy: PlanetPolicy, checkpoint_path: str | None, device: torch.device) -> None:
    register_checkpoint_module_aliases()
    if checkpoint_path is None:
        return
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("policy", checkpoint)
    policy.load_state_dict(state_dict)


def build_moves(batch: TurnBatch, policy: PlanetPolicy, device: torch.device, deterministic: bool) -> list[list[float | int]]:
    if batch.self_features_array.shape[0] == 0:
        return []
    with torch.inference_mode():
        outputs = policy(
            torch.from_numpy(batch.self_features_array).to(device),
            torch.from_numpy(batch.candidate_features_array).to(device),
            torch.from_numpy(batch.global_features_array).to(device),
            torch.from_numpy(batch.candidate_mask).to(device).bool(),
        )
        sampled = sample_actions(outputs, deterministic=deterministic)
    target_indices = sampled.target_index.detach().cpu().numpy()

    moves: list[list[float | int]] = []
    for row_idx, context in enumerate(batch.contexts):
        target_idx = int(target_indices[row_idx])
        if target_idx == 0:
            continue
        if target_idx >= len(context.candidate_ids):
            continue
        if not context.candidate_mask[target_idx]:
            continue
        ships = int(context.ship_counts[target_idx])
        if ships <= 0:
            continue
        moves.append([context.source_id, float(context.target_angles[target_idx]), ships])
    return moves


def extract_observation(state: Any) -> Any:
    if isinstance(state, dict):
        return state.get("observation")
    return getattr(state, "observation")


def extract_status(state: Any) -> str:
    if isinstance(state, dict):
        return str(state.get("status", "UNKNOWN"))
    return str(getattr(state, "status", "UNKNOWN"))


def extract_reward(state: Any) -> float:
    if isinstance(state, dict):
        value = state.get("reward", 0.0)
    else:
        value = getattr(state, "reward", 0.0)
    return 0.0 if value is None else float(value)

def agent(obs):
    print(os.listdir("."))
    cfg = load_train_config(str(repo_root / config_name))
    device = resolve_device("auto")
    deterministic = True
    seed_everything(42)
    policy = build_policy(cfg, device)
    load_checkpoint_if_available(policy, str(repo_root / checkpoint_name), device)
    policy.eval()
    batch = encode_turn(obs, cfg.env, env_index=0)
    moves = build_moves(batch, policy, device, deterministic)
    return moves

# def nearest_planet_sniper(obs: Any) -> list[list[float | int]]:
#     moves: list[list[float | int]] = []
#     player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
#     raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
#     planets = [Planet(*p) for p in raw_planets]
#     my_planets = [p for p in planets if p.owner == player]
#     targets = [p for p in planets if p.owner != player]
#     if not targets:
#         return moves
#     for mine in my_planets:
#         nearest = None
#         min_dist = float("inf")
#         for target in targets:
#             dist = math.hypot(mine.x - target.x, mine.y - target.y)
#             if dist < min_dist:
#                 min_dist = dist
#                 nearest = target
#         if nearest is None:
#             continue
#         ships_needed = max(nearest.ships + 1, 20)
#         if mine.ships < ships_needed:
#             continue
#         angle = math.atan2(nearest.y - mine.y, nearest.x - mine.x)
#         moves.append([mine.id, angle, ships_needed])
#     return moves

# if __name__ == "__main__":
#     from kaggle_environments import make

#     seed = 42

#     env = make(
#         "orbit_wars",
#         configuration={"seed": int(seed), "randomSeed": int(seed)},
#         debug=False,
#     )
#     env.reset(num_agents=2)
#     states = env.step([[], []])
#     player_obs = extract_observation(states[0])
#     opponent_obs = extract_observation(states[1])
#     done = extract_status(states[0]) != "ACTIVE"
#     step_count = 0

#     while not done:
#         player_action = agent(player_obs)
#         print(player_action)
#         opponent_action = nearest_planet_sniper(opponent_obs)
#         states = env.step([player_action, opponent_action])
#         player_obs = extract_observation(states[0])
#         opponent_obs = extract_observation(states[1])
#         done = extract_status(states[0]) != "ACTIVE"
#         step_count += 1
