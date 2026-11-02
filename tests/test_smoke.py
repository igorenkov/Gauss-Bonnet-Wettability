"""Smoke test: the package imports and the phase labels are as documented."""

import gauss_bonnet
from gauss_bonnet.phases import ALL_PHASES, OIL, SOLID, WATER


def test_package_imports_with_expected_phase_labels():
    assert gauss_bonnet.__version__
    assert (SOLID, WATER, OIL) == (0, 1, 2)
    assert ALL_PHASES == (0, 1, 2)
