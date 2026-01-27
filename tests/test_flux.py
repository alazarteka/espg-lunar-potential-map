"""Tests for src/flux.py - ERData, PitchAngle, and LossConeFitter classes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.flux import ERData, LossConeFitter, PitchAngle, build_lillis_mask
from src.utils.synthetic import prepare_synthetic_er

# ==============================================================================
# ERData Tests
# ==============================================================================


class TestERDataFromDataframe:
    """Tests for ERData.from_dataframe factory method."""

    def test_from_dataframe_creates_valid_instance(self):
        """Verify factory method creates ERData with provided data."""
        spec_sequence = np.repeat([1], config.SWEEP_ROWS)
        energies = np.linspace(20.0, 20000.0, config.SWEEP_ROWS)

        df = pd.DataFrame(
            {
                config.SPEC_NO_COLUMN: spec_sequence,
                config.ENERGY_COLUMN: energies,
                config.TIME_COLUMN: "2025-07-25T12:30:00",
                **dict.fromkeys(config.MAG_COLS, 0.5),
                **dict.fromkeys(config.FLUX_COLS, 100000.0),
                **dict.fromkeys(config.PHI_COLS, 45.0),
            }
        )

        er = ERData.from_dataframe(df, "test_file.tab")

        assert er.er_data_file == "test_file.tab"
        assert len(er.data) == config.SWEEP_ROWS
        assert config.SPEC_NO_COLUMN in er.data.columns

    def test_from_dataframe_cleans_invalid_bfield(self):
        """Sweeps with B-field magnitude outside [1e-9, 1e3] are removed."""
        spec_sequence = np.repeat([1, 2], config.SWEEP_ROWS)
        energies = np.tile(np.linspace(20.0, 20000.0, config.SWEEP_ROWS), 2)

        df = pd.DataFrame(
            {
                config.SPEC_NO_COLUMN: spec_sequence,
                config.ENERGY_COLUMN: energies,
                config.TIME_COLUMN: "2025-07-25T12:30:00",
                **dict.fromkeys(config.FLUX_COLS, 100000.0),
                **dict.fromkeys(config.PHI_COLS, 45.0),
            }
        )

        # First sweep: valid B-field
        df.loc[df[config.SPEC_NO_COLUMN] == 1, config.MAG_COLS] = [0.5, 0.5, 0.5]
        # Second sweep: invalid B-field (too small)
        df.loc[df[config.SPEC_NO_COLUMN] == 2, config.MAG_COLS] = [0.0, 0.0, 0.0]

        er = ERData.from_dataframe(df, "test")

        # Only first sweep should remain
        assert len(er.data) == config.SWEEP_ROWS
        assert set(er.data[config.SPEC_NO_COLUMN].unique()) == {1}

    def test_from_dataframe_cleans_invalid_timestamp(self):
        """Sweeps with epoch timestamp (1970-01-01) are removed."""
        spec_sequence = np.repeat([1, 2], config.SWEEP_ROWS)
        energies = np.tile(np.linspace(20.0, 20000.0, config.SWEEP_ROWS), 2)

        df = pd.DataFrame(
            {
                config.SPEC_NO_COLUMN: spec_sequence,
                config.ENERGY_COLUMN: energies,
                **dict.fromkeys(config.MAG_COLS, 0.5),
                **dict.fromkeys(config.FLUX_COLS, 100000.0),
                **dict.fromkeys(config.PHI_COLS, 45.0),
            }
        )

        # First sweep: valid timestamp
        df.loc[df[config.SPEC_NO_COLUMN] == 1, config.TIME_COLUMN] = (
            "2025-07-25T12:30:00"
        )
        # Second sweep: invalid timestamp
        df.loc[df[config.SPEC_NO_COLUMN] == 2, config.TIME_COLUMN] = (
            "1970-01-01T00:00:00"
        )

        er = ERData.from_dataframe(df, "test")

        # Only first sweep should remain
        assert len(er.data) == config.SWEEP_ROWS
        assert set(er.data[config.SPEC_NO_COLUMN].unique()) == {1}


class TestERDataCountColumns:
    """Tests for count reconstruction from flux."""

    def test_count_columns_added(self):
        """Verify count columns are added to data."""
        er = prepare_synthetic_er()

        assert config.COUNT_COLS[0] in er.data.columns
        assert config.COUNT_COLS[1] in er.data.columns

    def test_count_values_nonnegative(self):
        """Count estimates should be non-negative."""
        er = prepare_synthetic_er()

        counts = er.data[config.COUNT_COLS[0]].to_numpy()
        assert np.all(counts >= 0)


# ==============================================================================
# PitchAngle Tests
# ==============================================================================


class TestPitchAngle:
    """Tests for PitchAngle class."""

    def test_cartesian_conversion_known_values(self):
        """Spherical to Cartesian conversion for known angles."""
        er = prepare_synthetic_er()
        pa = PitchAngle(er)

        # Cartesian coords should be unit vectors (on unit sphere)
        norms = np.linalg.norm(pa.cartesian_coords, axis=-1)
        assert np.allclose(norms, 1.0, atol=1e-10)

    def test_pitch_angles_in_valid_range(self):
        """Pitch angles should be in [0, 180] degrees."""
        er = prepare_synthetic_er()
        pa = PitchAngle(er)

        assert np.all(pa.pitch_angles >= 0)
        assert np.all(pa.pitch_angles <= 180)

    def test_pitch_angles_shape_matches_data(self):
        """Pitch angles array shape matches (n_rows, n_channels)."""
        er = prepare_synthetic_er()
        pa = PitchAngle(er)

        n_rows = len(er.data)
        assert pa.pitch_angles.shape == (n_rows, config.CHANNELS)

    def test_pitch_angles_polarity_zero_is_nan(self):
        """Polarity=0 should mark pitch angles as NaN."""
        er = prepare_synthetic_er()
        polarity = np.zeros(len(er.data), dtype=int)
        pa = PitchAngle(er, polarity=polarity)

        assert np.isnan(pa.unit_magnetic_field).all()
        assert np.isnan(pa.pitch_angles).all()


# ==============================================================================
# LossConeFitter Tests
# ==============================================================================


class TestLossConeFitterNormalization:
    """Tests for flux normalization modes."""

    @pytest.fixture
    def fitter(self):
        """Create fitter with synthetic data."""
        er = prepare_synthetic_er()
        return LossConeFitter(
            er,
            normalization_mode="ratio",
        )

    def test_normalization_mode_ratio(self, fitter):
        """Ratio mode: divides by per-energy incident flux."""
        norm2d = fitter.build_norm2d(0)

        # Should have shape (SWEEP_ROWS, CHANNELS)
        assert norm2d.shape == (config.SWEEP_ROWS, config.CHANNELS)
        # Values should be finite where data exists
        finite_mask = np.isfinite(norm2d)
        assert finite_mask.sum() > 0

    def test_normalization_mode_ratio2(self):
        """Ratio2 mode: pairwise mirror normalization."""
        er = prepare_synthetic_er()
        fitter = LossConeFitter(
            er,
            normalization_mode="ratio2",
        )

        norm2d = fitter.build_norm2d(0)

        # Incident angles should be exactly 1.0 (except NaN)
        assert norm2d.shape == (config.SWEEP_ROWS, config.CHANNELS)

    def test_normalization_mode_ratio_rescaled(self):
        """Ratio_rescaled: per-energy ratio then global [0,1] scaling."""
        er = prepare_synthetic_er()
        fitter = LossConeFitter(
            er,
            normalization_mode="ratio_rescaled",
        )

        norm2d = fitter.build_norm2d(0)

        # All finite values should be in [0, 1]
        finite_vals = norm2d[np.isfinite(norm2d)]
        if len(finite_vals) > 0:
            assert np.all(finite_vals >= 0)
            assert np.all(finite_vals <= 1.0 + 1e-10)

    def test_normalization_mode_global(self):
        """Global mode: divides by maximum incident flux."""
        er = prepare_synthetic_er()
        fitter = LossConeFitter(
            er,
            normalization_mode="global",
        )

        norm2d = fitter.build_norm2d(0)

        assert norm2d.shape == (config.SWEEP_ROWS, config.CHANNELS)

    def test_invalid_normalization_mode_raises(self):
        """Unknown normalization mode raises ValueError."""
        er = prepare_synthetic_er()

        with pytest.raises(ValueError, match="Unknown normalization_mode"):
            LossConeFitter(
                er,
                normalization_mode="invalid_mode",
            )

    def test_invalid_incident_flux_stat_raises(self):
        """Unknown incident_flux_stat raises ValueError."""
        er = prepare_synthetic_er()

        with pytest.raises(ValueError, match="Unknown incident_flux_stat"):
            LossConeFitter(
                er,
                incident_flux_stat="median",  # Invalid
            )

    def test_invalid_fit_method_raises(self):
        """Unknown fit_method raises ValueError."""
        er = prepare_synthetic_er()

        with pytest.raises(ValueError, match="Unknown fit_method"):
            LossConeFitter(
                er,
                fit_method="unknown",
            )

    def test_invalid_loss_cone_background_raises(self):
        """Non-positive loss_cone_background raises ValueError."""
        er = prepare_synthetic_er()

        with pytest.raises(ValueError, match="loss_cone_background must be positive"):
            LossConeFitter(
                er,
                loss_cone_background=-1.0,
            )


def test_build_lillis_mask_thresholds():
    """Lillis mask should exclude low/zero, near-max, and NaN bins."""
    raw_flux = np.array([[1.0, 10.0, 2.0, 0.1, np.nan]])
    pitches = np.array([[30.0, 150.0, 60.0, 120.0, 45.0]])
    mask = build_lillis_mask(raw_flux, pitches)
    expected = np.array([[True, False, False, False, False]])
    np.testing.assert_array_equal(mask, expected)


class TestLossConeFitterFitting:
    """Tests for surface potential fitting."""

    def test_fit_returns_valid_results(self):
        """Full fitting returns array with expected shape."""
        er = prepare_synthetic_er()
        fitter = LossConeFitter(er)

        results = fitter.fit_surface_potential()

        # Should have [U_surface, bs_over_bm, beam_amp, chi2, chunk_index]
        assert results.shape[1] == 5
        # At least one chunk should exist
        assert results.shape[0] >= 1

    def test_fit_single_chunk(self):
        """Fitting a single chunk returns valid values."""
        er = prepare_synthetic_er()
        fitter = LossConeFitter(er)

        U_surface, bs_over_bm, beam_amp, chi2 = fitter._fit_surface_potential(0)

        # bs_over_bm should be in bounds
        if not np.isnan(bs_over_bm):
            assert 0.1 <= bs_over_bm <= 1.1

    def test_fit_handles_out_of_range_chunk(self):
        """Fitting a chunk beyond data range returns NaN."""
        er = prepare_synthetic_er()
        fitter = LossConeFitter(er)

        # Request chunk that doesn't exist
        U_surface, bs_over_bm, beam_amp, chi2 = fitter._fit_surface_potential(999)

        assert np.isnan(U_surface)
        assert np.isnan(bs_over_bm)


class TestLossConeFitterLatinHypercube:
    """Tests for Latin Hypercube sampling."""

    def test_lhs_shape(self):
        """LHS sample has correct shape (400 samples, 3 params)."""
        er = prepare_synthetic_er()
        fitter = LossConeFitter(er)

        assert fitter.lhs.shape == (400, 3)

    def test_lhs_bounds(self):
        """LHS samples are within specified bounds."""
        er = prepare_synthetic_er()
        fitter = LossConeFitter(er)

        # U_surface: [-1000, 1000]
        assert np.all(fitter.lhs[:, 0] >= -1000)
        assert np.all(fitter.lhs[:, 0] <= 1000)
        # bs_over_bm: [0.1, 1.1]
        assert np.all(fitter.lhs[:, 1] >= 0.1)
        assert np.all(fitter.lhs[:, 1] <= 1.1)
