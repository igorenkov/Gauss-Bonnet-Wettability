"""Droplet extraction: connected components of the OIL phase (Algorithm 1, "for each droplet").

A droplet is a connected component of oil. For each droplet we compute its volume, centroid,
bounding box and a preliminary status. Statuses by priority:

1. ``border``     - the droplet touches the image boundary, so its surface is not closed;
2. ``too_small``  - the volume is below ``min_voxels``;
3. ``no_contact`` - no droplet voxel shares a face with the solid, i.e. N = 0;
4. ``ok``.

The final N and Euler characteristic are computed by ``topology.py``; here the status is only
a voxel-level filter.

Connectivity: 26 by default (face, edge, vertex), or 6 (face only).
Consistency with the marching cubes mesh topology is handled in ``mesh.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from gauss_bonnet.io import (
    STATUS_BORDER,
    STATUS_NO_CONTACT,
    STATUS_OK,
    STATUS_TOO_SMALL,
)
from gauss_bonnet.phases import OIL, SOLID, WATER

DEFAULT_CONNECTIVITY = 26
DEFAULT_MIN_VOXELS = 27  # ~3x3x3; the paper still handles a 12x13x12 droplet (Fig. 6c)
DEFAULT_PAD = 2  # padding around a cropped droplet, vox

_STRUCTURES = {
    6: ndimage.generate_binary_structure(3, 1),
    26: ndimage.generate_binary_structure(3, 3),
}
_FACE = _STRUCTURES[6]


@dataclass(frozen=True)
class Droplet:
    """One oil droplet. Coordinates are voxels of the source volume, axes (z, y, x)."""

    id: int  # label in the labels array, 1 ... n
    volume_vox: int
    centroid: tuple[float, float, float]
    bbox: tuple[slice, slice, slice]  # tight bounding box of the droplet, no padding
    touches_border: bool
    touches_solid: bool  # has a face-adjacent solid neighbour
    status: str

    @property
    def bbox_shape(self) -> tuple[int, int, int]:
        return tuple(s.stop - s.start for s in self.bbox)


@dataclass(frozen=True)
class DropletCrop:
    """Local volume of one droplet for meshing.

    In ``volume`` the voxels of this droplet are OIL, other oil droplets are replaced by WATER
    (the surrounding fluid for this droplet), solid and water are unchanged, and the region
    outside the image is WATER. The global coordinate of voxel ``volume[i, j, k]`` is
    ``offset + (i, j, k)``.
    """

    volume: np.ndarray  # uint8
    offset: tuple[int, int, int]
    droplet: Droplet


def label_droplets(volume: np.ndarray, connectivity: int = DEFAULT_CONNECTIVITY):
    """Label the connected components of oil.

    Returns
    -------
    labels : np.ndarray (int32)
        0 = not oil, 1 ... n = droplet number.
    n : int
        Number of droplets.
    """
    if connectivity not in _STRUCTURES:
        raise ValueError(f"connectivity must be 6 or 26, got {connectivity}")
    labels = np.zeros(volume.shape, dtype=np.int32)
    n = ndimage.label(volume == OIL, structure=_STRUCTURES[connectivity], output=labels)
    return labels, int(n)


def find_droplets(
    volume: np.ndarray,
    *,
    connectivity: int = DEFAULT_CONNECTIVITY,
    min_voxels: int = DEFAULT_MIN_VOXELS,
) -> tuple[np.ndarray, list[Droplet]]:
    """Find all oil droplets and assign their preliminary status.

    Parameters
    ----------
    volume
        Labelled uint8 volume (see ``io.load_volume``).
    connectivity
        6 or 26.
    min_voxels
        Minimum volume of an ok droplet, vox (>= 1).

    Returns
    -------
    labels, droplets
        Droplet label array (as in ``label_droplets``) and the droplets sorted by id.
    """
    if min_voxels < 1:
        raise ValueError(f"min_voxels must be >= 1, got {min_voxels}")
    labels, n = label_droplets(volume, connectivity)
    shape = volume.shape
    droplets: list[Droplet] = []

    for idx, bbox in enumerate(ndimage.find_objects(labels), start=1):
        if bbox is None:  # cannot happen after ndimage.label, but find_objects allows it
            continue
        mask = labels[bbox] == idx
        coords = np.nonzero(mask)
        volume_vox = int(coords[0].size)
        centroid = tuple(float(c.mean() + s.start) for c, s in zip(coords, bbox))

        touches_border = any(s.start == 0 or s.stop == dim for s, dim in zip(bbox, shape))

        # Face neighbours: the box grown by 1 voxel within the image.
        grown = tuple(slice(max(s.start - 1, 0), min(s.stop + 1, dim)) for s, dim in zip(bbox, shape))
        inner = tuple(slice(s.start - g.start, s.stop - g.start) for s, g in zip(bbox, grown))
        local_mask = np.zeros(tuple(g.stop - g.start for g in grown), dtype=bool)
        local_mask[inner] = mask
        halo = ndimage.binary_dilation(local_mask, structure=_FACE) & ~local_mask
        touches_solid = bool(np.any(volume[grown][halo] == SOLID))

        if touches_border:
            status = STATUS_BORDER
        elif volume_vox < min_voxels:
            status = STATUS_TOO_SMALL
        elif not touches_solid:
            status = STATUS_NO_CONTACT
        else:
            status = STATUS_OK

        droplets.append(
            Droplet(
                id=idx,
                volume_vox=volume_vox,
                centroid=centroid,
                bbox=bbox,
                touches_border=touches_border,
                touches_solid=touches_solid,
                status=status,
            )
        )
    assert len(droplets) == n
    return labels, droplets


def extract_droplet(
    volume: np.ndarray,
    labels: np.ndarray,
    droplet: Droplet,
    pad: int = DEFAULT_PAD,
) -> DropletCrop:
    """Crop the local droplet volume with ``pad`` voxels of padding on each side.

    The padding lets marching cubes build a closed surface.
    """
    if pad < 1:
        raise ValueError(f"pad must be >= 1, got {pad}")
    shape = volume.shape
    lo = [s.start - pad for s in droplet.bbox]
    hi = [s.stop + pad for s in droplet.bbox]
    src = tuple(slice(max(a, 0), min(b, dim)) for a, b, dim in zip(lo, hi, shape))
    dst = tuple(slice(s.start - a, s.stop - a) for s, a in zip(src, lo))

    local = np.full(tuple(b - a for a, b in zip(lo, hi)), WATER, dtype=np.uint8)
    part = volume[src].copy()
    part[part == OIL] = WATER
    part[labels[src] == droplet.id] = OIL
    local[dst] = part
    return DropletCrop(volume=local, offset=tuple(lo), droplet=droplet)
