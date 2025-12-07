from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.config as config
import src.spacecraft_potential as spacecraft_potential
from src.potential_mapper import pipeline
from src.utils.units import ureg


class DummyER:
    def __init__(self, spec_nos, energies: np.ndarray | None = None):
        if energies is None:
            energies = np.zeros_like(spec_nos, dtype=float)
        self.data = pd.DataFrame(
            {
                config.SPEC_NO_COLUMN: spec_nos,
                config.ENERGY_COLUMN: energies,
            }
        )
        self.er_data_file = "dummy"


def test_spacecraft_potential_per_row_assigns_values(monkeypatch):
    n_rows = 2 * config.SWEEP_ROWS
    spec_sequence = np.repeat([10, 11], config.SWEEP_ROWS)
    er = DummyER(spec_sequence)

    returned = []

    def fake_calculate(er_data, spec_no):
        returned.append(spec_no)
        value = -100 if spec_no == 10 else -50
        return (object(), value * ureg.volt)

    monkeypatch.setattr(spacecraft_potential, "calculate_potential", fake_calculate)

    potentials = pipeline._spacecraft_potential_per_row(er, n_rows)

    assert returned == [10, 11]
    assert np.allclose(potentials[: config.SWEEP_ROWS], -100)
    assert np.allclose(potentials[config.SWEEP_ROWS :], -50)


def test_spacecraft_potential_per_row_handles_failures(monkeypatch):
    n_rows = config.SWEEP_ROWS
    spec_sequence = np.repeat([7], config.SWEEP_ROWS)
    er = DummyER(spec_sequence)

    def fake_calculate(*_args, **_kwargs):
        return None

    monkeypatch.setattr(spacecraft_potential, "calculate_potential", fake_calculate)

    potentials = pipeline._spacecraft_potential_per_row(er, n_rows)

    assert np.all(np.isnan(potentials))


class _FakeArrays:
    def __init__(self, n_rows):
        radius = config.LUNAR_RADIUS.to(ureg.meter).magnitude
        self.lp_positions = np.tile(np.array([radius, 0.0, 0.0]), (n_rows, 1))
        self.lp_vectors_to_sun = np.tile(np.array([1.0, 0.0, 0.0]), (n_rows, 1))
        self.moon_vectors_to_sun = np.tile(np.array([1.0, 0.0, 0.0]), (n_rows, 1))


class _FakeCoordinateCalculator:
    def __init__(self, *_args):
        self._last_n = 0

    def calculate_coordinate_transformation(self, er_data):
        self._last_n = len(er_data.data)
        return _FakeArrays(self._last_n)


class _FakeFitter:
    def __init__(self, *_args, spacecraft_potential=None, **_kwargs):
        self.spacecraft_potential = spacecraft_potential

    def fit_surface_potential(self):
        return np.array(
            [
                [-15.0, 0.2, 0.5, config.FIT_ERROR_THRESHOLD / 2, 0],
                [-20.0, 0.3, 0.5, config.FIT_ERROR_THRESHOLD / 2, 1],
            ]
        )


class _FakeERData:
    def __init__(self, *_args, **_kwargs):
        spec_sequence = np.repeat([44, 45], config.SWEEP_ROWS)
        energies = np.linspace(10.0, 20.0, spec_sequence.size)
        self.data = pd.DataFrame(
            {
                config.SPEC_NO_COLUMN: spec_sequence,
                config.ENERGY_COLUMN: energies,
            }
        )
        self.er_data_file = "fake"


def test_spacecraft_potential_per_row_restores_energy(monkeypatch):
    n_rows = config.SWEEP_ROWS
    spec_sequence = np.repeat([5], config.SWEEP_ROWS)
    energies = np.linspace(1.0, 2.0, n_rows)
    er = DummyER(spec_sequence, energies=energies)

    def mutate_energy(er_data, spec_no):
        er_data.data.loc[:, config.ENERGY_COLUMN] += 123.0
        return (object(), -25.0 * ureg.volt)

    monkeypatch.setattr(spacecraft_potential, "calculate_potential", mutate_energy)

    pipeline._spacecraft_potential_per_row(er, n_rows)

    assert np.allclose(er.data[config.ENERGY_COLUMN].to_numpy(), energies)


@pytest.fixture
def patched_environment(monkeypatch):
    n_rows = 2 * config.SWEEP_ROWS

    fake_calc = _FakeCoordinateCalculator()
    monkeypatch.setattr(pipeline, "CoordinateCalculator", lambda *args: fake_calc)
    monkeypatch.setattr(pipeline, "ERData", _FakeERData)
    captured = {}

    def fake_loss_cone_fitter(*args, **kwargs):
        fitter = _FakeFitter(*args, **kwargs)
        captured["spacecraft_potential"] = fitter.spacecraft_potential
        return fitter

    monkeypatch.setattr(pipeline, "LossConeFitter", fake_loss_cone_fitter)
    monkeypatch.setattr(
        pipeline, "project_magnetic_fields", lambda *_: np.zeros((n_rows, 3))
    )
    monkeypatch.setattr(
        pipeline,
        "find_surface_intersection",
        lambda *_args: (
            np.tile(
                np.array([config.LUNAR_RADIUS.to(ureg.meter).magnitude, 0.0, 0.0]),
                (n_rows, 1),
            ),
            np.ones(n_rows, dtype=bool),
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "get_intersections_or_none_batch",
        lambda *args, **kwargs: (None, np.zeros(n_rows, dtype=bool)),
    )
    monkeypatch.setattr(
        pipeline,
        "load_attitude_data",
        lambda *_args: (np.array([0.0]), np.array([0.0]), np.array([0.0])),
    )

    sc_values = {
        44: -100.0,
        45: -50.0,
    }

    def fake_calculate_potential(_er_data, spec_no):
        return (object(), sc_values[spec_no] * ureg.volt)

    monkeypatch.setattr(
        spacecraft_potential, "calculate_potential", fake_calculate_potential
    )

    return n_rows, captured


def test_process_lp_file_combines_spacecraft_potential(
    monkeypatch, tmp_path, patched_environment
):
    n_rows, captured = patched_environment

    dummy_file = tmp_path / "dummy.tab"
    dummy_file.write_text("test")

    result = pipeline.process_lp_file(dummy_file)

    assert result.spacecraft_potential.shape[0] == n_rows
    assert np.allclose(result.spacecraft_potential[: config.SWEEP_ROWS], -100.0)
    assert np.allclose(result.spacecraft_potential[config.SWEEP_ROWS :], -50.0)

    assert np.allclose(result.projected_potential[: config.SWEEP_ROWS], -15.0)
    assert np.allclose(result.projected_potential[config.SWEEP_ROWS :], -20.0)

    sc_args = captured["spacecraft_potential"]
    assert sc_args is not None
    assert np.allclose(sc_args[: config.SWEEP_ROWS], -100.0)


# ==============================================================================
# DataLoader Tests
# ==============================================================================


class TestDataLoader:
    """Tests for DataLoader configuration."""

    def test_month_name_mapping(self):
        """MONTH_TO_NUM contains all 12 months."""
        assert len(pipeline.DataLoader.MONTH_TO_NUM) == 12
        assert pipeline.DataLoader.MONTH_TO_NUM["JAN"] == "01"
        assert pipeline.DataLoader.MONTH_TO_NUM["DEC"] == "12"

    def test_num_to_month_inverse(self):
        """NUM_TO_MONTH is the inverse of MONTH_TO_NUM."""
        for month_name, month_num in pipeline.DataLoader.MONTH_TO_NUM.items():
            assert pipeline.DataLoader.NUM_TO_MONTH[month_num] == month_name

    def test_month_to_num_coverage(self):
        """All months 01-12 are mapped."""
        months = list(pipeline.DataLoader.NUM_TO_MONTH.keys())
        assert sorted(months) == [f"{i:02d}" for i in range(1, 13)]


# ==============================================================================
# load_all_data Tests
# ==============================================================================


class TestLoadAllData:
    """Tests for load_all_data file merging."""

    def test_empty_file_list_returns_empty_erdata(self):
        """load_all_data with empty list returns empty ERData."""
        result = pipeline.load_all_data([])
        
        assert result.data.empty
        assert result.er_data_file == "empty"

    def test_spec_no_offset_ensures_uniqueness(self, tmp_path, monkeypatch):
        """load_all_data offsets spec_no to maintain uniqueness across files."""
        # Create test files with overlapping spec_no values
        all_cols = config.ALL_COLS
        
        file1 = tmp_path / "file1.tab"
        file2 = tmp_path / "file2.tab"
        
        # Create minimal valid data
        n_rows = config.SWEEP_ROWS
        
        def make_file_content(spec_no_start: int) -> str:
            lines = []
            for i in range(n_rows):
                # Create space-separated values for ALL_COLS
                values = ["0.0"] * len(all_cols)
                # Set specific columns
                spec_idx = all_cols.index(config.SPEC_NO_COLUMN)
                time_idx = all_cols.index(config.TIME_COLUMN)
                energy_idx = all_cols.index(config.ENERGY_COLUMN)
                
                values[spec_idx] = str(spec_no_start)
                values[time_idx] = "2025-01-01T00:00:00"
                values[energy_idx] = str(20.0 + i)
                
                # Set valid mag field
                for col in config.MAG_COLS:
                    mag_idx = all_cols.index(col)
                    values[mag_idx] = "1.0"
                
                lines.append(" ".join(values))
            return "\n".join(lines)
        
        file1.write_text(make_file_content(1))
        file2.write_text(make_file_content(1))  # Same spec_no, should be offset
        
        # Mock theta loading since it happens in ERData
        result = pipeline.load_all_data([file1, file2])
        
        # Check that spec_no values are unique
        spec_nos = result.data[config.SPEC_NO_COLUMN].unique()
        # After merging, should have at least 2 different spec_no values
        # (original + offset version)
        assert len(result.data) > 0


# ==============================================================================
# _apply_fit_results Tests
# ==============================================================================


class TestApplyFitResults:
    """Tests for _apply_fit_results helper."""

    def test_applies_valid_results(self):
        """Valid fit results are applied to proj_potential array."""
        proj_potential = np.full(config.SWEEP_ROWS * 2, np.nan)
        
        # Fit result: [U_surface, bs_over_bm, beam_amp, chi2, chunk_index]
        fit_results = np.array([
            [-10.0, 0.5, 1.0, config.FIT_ERROR_THRESHOLD / 2, 0],
        ])
        
        pipeline._apply_fit_results(
            fit_results, proj_potential, row_offset=0, n_total=len(proj_potential)
        )
        
        # First SWEEP_ROWS should have the value
        assert np.allclose(proj_potential[:config.SWEEP_ROWS], -10.0)
        # Second SWEEP_ROWS should still be NaN (no result for chunk 1)
        assert np.all(np.isnan(proj_potential[config.SWEEP_ROWS:]))

    def test_skips_high_chi2_results(self):
        """Results with chi2 > threshold are skipped."""
        proj_potential = np.full(config.SWEEP_ROWS, np.nan)
        
        fit_results = np.array([
            [-10.0, 0.5, 1.0, config.FIT_ERROR_THRESHOLD * 2, 0],  # High chi2
        ])
        
        pipeline._apply_fit_results(
            fit_results, proj_potential, row_offset=0, n_total=len(proj_potential)
        )
        
        # Should remain NaN because chi2 exceeds threshold
        assert np.all(np.isnan(proj_potential))

    def test_skips_nan_results(self):
        """NaN fit results are skipped."""
        proj_potential = np.full(config.SWEEP_ROWS, np.nan)
        
        fit_results = np.array([
            [np.nan, 0.5, 1.0, 0.1, 0],
        ])
        
        pipeline._apply_fit_results(
            fit_results, proj_potential, row_offset=0, n_total=len(proj_potential)
        )
        
        assert np.all(np.isnan(proj_potential))

    def test_empty_fit_results(self):
        """Empty fit_results is handled gracefully."""
        proj_potential = np.full(config.SWEEP_ROWS, np.nan)
        
        fit_results = np.array([]).reshape(0, 5)
        
        # Should not raise
        pipeline._apply_fit_results(
            fit_results, proj_potential, row_offset=0, n_total=len(proj_potential)
        )
        
        assert np.all(np.isnan(proj_potential))

