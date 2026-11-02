# Gauss-Bonnet Wettability

Macroscopic contact angle θ_macro from segmented 3D rock images (solid, water, oil),
computed per droplet with the Gauss–Bonnet theorem.

Method: C. Sun, J.E. McClure, P. Mostaghimi, A.L. Herring, D.E. Meisenheimer, D. Wildenschild,
S. Berg, R.T. Armstrong, *Characterization of wetting using topological principles*,
J. Colloid Interface Sci. **578** (2020) 106–115,
[doi:10.1016/j.jcis.2020.05.076](https://doi.org/10.1016/j.jcis.2020.05.076).

## Features

- Input: a 3D `.npy` array, axes `(z, y, x)`, labels `0` solid, `1` water, `2` oil.
- Droplets are connected components of the non-wetting phase (`--droplet-phase auto|oil|water`).
- For each droplet: marching cubes surface, oil–fluid / oil–solid labelling, region-wise
  smoothing, Euler characteristic and contact line loops N, Gaussian curvature, then
  θ_macro = (4πχ(D) − Σ κ_G·A) / (4N). The angle is always reported through water.
- Droplets touching the image edge, too small or without solid contact are flagged, not measured.
- Optional random subsampling of droplets for large samples (`--max-droplets`, `--seed`).
- Output: `droplets.csv` (one row per droplet) and `summary.json` (statistics and parameters),
  plus optional plots.

## Usage

```bash
pip install -e .
```

```bash
gauss-bonnet sample.npy --voxel-size 1.0 --out results/ --plots
```

```bash
gauss-bonnet-plot results/ --true-angle 45
```

Run `gauss-bonnet --help` for all algorithm parameters. Tests: `pip install -e .[dev]`, then `pytest`.
