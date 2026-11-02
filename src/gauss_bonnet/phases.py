"""Phase labels of the input 3D array (.npy) and droplet phase selection."""

from __future__ import annotations

import numpy as np

SOLID = 0  # rock matrix (s)
WATER = 1  # water
OIL = 2  # oil

ALL_PHASES = (SOLID, WATER, OIL)

# Droplet phase. Inside the pipeline droplets always carry the OIL label and the surrounding
# fluid the WATER label. For water droplets the fluid labels are swapped (swap_fluids) and the
# angle is converted back to the angle through water: theta_w = pi - theta.
DROPLET_OIL = "oil"
DROPLET_WATER = "water"
DROPLET_PHASES = (DROPLET_OIL, DROPLET_WATER)


def swap_fluids(volume: np.ndarray) -> np.ndarray:
    """Copy of the volume with water and oil labels swapped (solid unchanged)."""
    out = volume.copy()
    out[volume == WATER] = OIL
    out[volume == OIL] = WATER
    return out


def droplet_volume(volume: np.ndarray, phase: str) -> np.ndarray:
    """Volume in which the droplet phase ``phase`` carries the OIL label (the volume itself for ``"oil"``)."""
    if phase == DROPLET_OIL:
        return volume
    if phase == DROPLET_WATER:
        return swap_fluids(volume)
    raise ValueError(f"droplet phase must be one of {DROPLET_PHASES}, got {phase!r}")
