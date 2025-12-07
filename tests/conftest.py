import numpy as np
import pytest

from src import config


@pytest.fixture(autouse=True)
def mock_data_files(monkeypatch):
    """
    Mock the data files (solid_angles.tab and theta.tab) that are loaded via np.loadtxt.
    This ensures tests don't fail due to missing data files.
    """

    # Reconstruct the theta values and solid angles as in src/utils/synthetic.py
    # We copy the logic here to avoid importing synthetic which imports flux/erdata

    phis_by_latitude = {
        78.75: ([], 4, 0.119570),
        56.25: ([], 8, 0.170253),
        33.75: ([], 16, 0.127401),
        11.25: ([], 16, 0.150279),
        -11.25: ([], 16, 0.150279),
        -33.75: ([], 16, 0.127401),
        -56.25: ([], 8, 0.170253),
        -78.75: ([], 4, 0.119570),
    }

    temp_thetas = []
    for key, (_, count, _) in phis_by_latitude.items():
        temp_thetas.extend([key] * count)

    # Sort by absolute latitude
    sorted_thetas = sorted(np.array(temp_thetas), key=lambda x: abs(x))

    # Map theta back to solid angle
    theta_to_sa = {k: v[2] for k, v in phis_by_latitude.items()}
    sorted_solid_angles = [theta_to_sa[t] for t in sorted_thetas]

    mock_thetas = np.array(sorted_thetas, dtype=np.float64)
    mock_solid_angles = np.array(sorted_solid_angles, dtype=np.float64)

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
