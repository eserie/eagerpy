from typing import Any

import pytest
import numpy as np
import eagerpy as ep


@pytest.mark.parametrize("fill_value", [0.0, 0, "0"])
def test_astensors_list_float(fill_value: Any):  # type: ignore
    x = [
        fill_value,
    ] * 3
    ex = ep.astensors(x)
    assert isinstance(ex[0], ep.Tensor)
    ex_stacked = ep.stack(ex)
    x_stacked = ex_stacked.raw
    assert isinstance(x_stacked, np.ndarray)
