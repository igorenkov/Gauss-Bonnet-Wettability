import numpy as np
import pytest

from gauss_bonnet.phases import OIL, SOLID, WATER, droplet_volume, swap_fluids


def test_swap_fluids_exchanges_water_and_oil_only():
    vol = np.array([[[SOLID, WATER, OIL]]], dtype=np.uint8)
    out = swap_fluids(vol)
    assert out.tolist() == [[[SOLID, OIL, WATER]]]
    assert vol.tolist() == [[[SOLID, WATER, OIL]]]  # input is unchanged
    assert np.array_equal(swap_fluids(out), vol)


def test_droplet_volume_by_phase():
    vol = np.array([[[SOLID, WATER, OIL]]], dtype=np.uint8)
    assert droplet_volume(vol, "oil") is vol
    assert droplet_volume(vol, "water").tolist() == [[[SOLID, OIL, WATER]]]
    with pytest.raises(ValueError, match="droplet phase"):
        droplet_volume(vol, "gas")
