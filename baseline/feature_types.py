from __future__ import annotations

from dataclasses import dataclass, astuple
import numpy as np

class ArrayMixin:
    def __array__(self, dtype=np.float32) -> np.ndarray:
        return np.asarray(astuple(self), dtype=dtype)

"""Normalized features of the player."""
@dataclass(slots=True)
class SelfFeatures(ArrayMixin):
    id: float = 0.0
    x: float = 0.0
    y: float = 0.0
    radius: float = 0.0
    ships: float = 0.0
    is_rotating: float = 0.0
    production: float = 0.0
    owned_planets: float = 0.0
    enemy_planets: float = 0.0
    total_own_ships: float = 0.0
    total_enemy_ships: float = 0.0

"""Normalized features of a candidate target planet for a potential action."""
@dataclass(slots=True)
class CandidateFeatures(ArrayMixin):
    id: float = 0.0
    is_neutral: float = 0.0
    is_owner: float = 0.0
    is_enemy: float = 0.0
    # Relative position of target to source planet
    tgt_dx: float = 0.0
    tgt_dy: float = 0.0
    #
    src_dx: float = 0.0
    src_dy: float = 0.0
    src_tgt_dist: float = 0.0
    tgt_ships: float = 0.0
    tgt_production: float = 0.0
    is_tgt_rotating: float = 0.0
    is_shot_crosses_sun: float = 0.0
    src_ships: float = 0.0

"""Global features of the game state that provide context for decision-making."""
@dataclass(slots=True)
class GlobalFeatures(ArrayMixin):
    step: float = 0.0
    own_planets: float = 0.0
    enemy_planets: float = 0.0
    neutral_planets: float = 0.0
    total_own_ships: float = 0.0
    total_enemy_ships: float = 0.0
    total_own_fleet_ships: float = 0.0
    total_enemy_fleet_ships: float = 0.0