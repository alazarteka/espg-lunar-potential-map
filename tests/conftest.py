import numpy as np
import pytest

from src import config
from src.utils.synthetic import prepare_phis


@pytest.fixture(autouse=True)
def mock_data_files(monkeypatch):
    """
    Mock the data files (solid_angles.tab and theta.tab) that are loaded via np.loadtxt.
    This ensures tests don't fail due to missing data files.
    """

    # Solid angles come straight from the canonical generator in
    # src/utils/synthetic.py (prepare_phis), so this stays in sync with the
    # values ERData/PitchAngle actually use when building synthetic data.
    # prepare_phis() doesn't return the paired theta array, so we rebuild it
    # here using the same (theta -> channel count) mapping and abs-latitude
    # ordering that prepare_phis() uses internally. config.BINS_BY_LATITUDE is
    # the canonical source for those counts (it's also what ERData uses), so
    # no physical constants are duplicated here -- only the merge/sort order.
    _phis, mock_solid_angles = prepare_phis()

    temp_thetas: list[float] = []
    for latitude, count in config.BINS_BY_LATITUDE.items():
        temp_thetas.extend([latitude] * count)
    sorted_thetas = sorted(temp_thetas, key=lambda x: abs(x))

    mock_thetas = np.array(sorted_thetas, dtype=np.float64)
    mock_solid_angles = np.asarray(mock_solid_angles, dtype=np.float64)

    original_loadtxt = np.loadtxt

    def side_effect(fname, *args, **kwargs):
        fname_str = str(fname)
        if config.SOLID_ANGLES_FILE in fname_str:
            return mock_solid_angles
        elif config.THETA_FILE in fname_str:
            return mock_thetas
        else:
            return original_loadtxt(fname, *args, **kwargs)

    monkeypatch.setattr(np, "loadtxt", side_effect)

    # Patch cached theta loader to avoid file access in CI.
    import src.flux as flux_module
    import src.utils.thetas as thetas_module

    thetas_module.get_thetas.cache_clear()
    monkeypatch.setattr(
        thetas_module, "get_thetas", lambda theta_path=None: mock_thetas
    )
    monkeypatch.setattr(flux_module, "get_thetas", lambda theta_path=None: mock_thetas)
