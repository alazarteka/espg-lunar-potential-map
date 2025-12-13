"""Tests for temporal basis function fitting."""

import numpy as np
import pytest

from src.temporal.basis import (
    AVAILABLE_BASES,
    SIDEREAL_PERIOD_DAYS,
    BasisFitResult,
    BasisFunction,
    build_temporal_design,
    fit_temporal_basis,
    parse_basis_spec,
    reconstruct_at_times,
    save_basis_result,
    _get_basis_func_by_name,
    _make_sidereal_basis,
    _make_synodic_basis,
    _make_synodic2_basis,
)
from src.temporal.coefficients import DEFAULT_SYNODIC_PERIOD_DAYS


class TestBasisFunction:
    """Tests for BasisFunction dataclass."""

    def test_basis_function_creation(self):
        """Test creating a basic basis function."""
        func = lambda t: np.ones_like(t)
        basis = BasisFunction(name="constant", func=func)
        assert basis.name == "constant"
        assert callable(basis.func)

    def test_basis_function_evaluation(self):
        """Test evaluating a basis function."""
        func = lambda t: 2 * t
        basis = BasisFunction(name="linear", func=func)
        t = np.array([0.0, 1.0, 2.0])
        result = basis.func(t)
        np.testing.assert_array_equal(result, np.array([0.0, 2.0, 4.0]))


class TestBasisFactories:
    """Tests for basis function factory methods."""

    def test_make_synodic_basis_returns_two_functions(self):
        """Synodic basis should return cos and sin components."""
        bases = _make_synodic_basis()
        assert len(bases) == 2
        assert bases[0].name == "synodic_cos"
        assert bases[1].name == "synodic_sin"

    def test_make_synodic_basis_with_custom_period(self):
        """Test synodic basis with custom period."""
        custom_period = 30.0  # days
        bases = _make_synodic_basis(period_days=custom_period)
        assert len(bases) == 2

        # Evaluate at one period - should complete full cycle
        period_hours = custom_period * 24.0
        t = np.array([0.0, period_hours / 4, period_hours / 2, period_hours])

        cos_vals = bases[0].func(t)
        sin_vals = bases[1].func(t)

        # cos should be [1, 0, -1, 1]
        np.testing.assert_allclose(cos_vals, [1, 0, -1, 1], atol=1e-10)
        # sin should be [0, 1, 0, 0]
        np.testing.assert_allclose(sin_vals, [0, 1, 0, 0], atol=1e-10)

    def test_make_sidereal_basis_returns_two_functions(self):
        """Sidereal basis should return cos and sin components."""
        bases = _make_sidereal_basis()
        assert len(bases) == 2
        assert bases[0].name == "sidereal_cos"
        assert bases[1].name == "sidereal_sin"

    def test_make_synodic2_basis_double_frequency(self):
        """Synodic2 basis should oscillate at 2x frequency."""
        bases = _make_synodic2_basis()
        assert len(bases) == 2
        assert bases[0].name == "synodic2_cos"
        assert bases[1].name == "synodic2_sin"

        # At half the base period, synodic2 should complete one full cycle
        half_period_hours = DEFAULT_SYNODIC_PERIOD_DAYS * 24.0 / 2
        t = np.array([0.0, half_period_hours])

        cos_vals = bases[0].func(t)
        # Should return to initial value after half the base period
        np.testing.assert_allclose(cos_vals[0], cos_vals[1], atol=1e-10)

    def test_constant_basis_is_ones(self):
        """Constant basis should return ones."""
        bases = AVAILABLE_BASES["constant"]()
        assert len(bases) == 1
        assert bases[0].name == "constant"

        t = np.array([0.0, 10.0, 100.0, 1000.0])
        result = bases[0].func(t)
        np.testing.assert_array_equal(result, np.ones(4))

    def test_linear_basis_normalized(self):
        """Linear basis should be normalized by max time."""
        bases = AVAILABLE_BASES["linear"]()
        assert len(bases) == 1
        assert bases[0].name == "linear"

        t = np.array([0.0, 50.0, 100.0])
        result = bases[0].func(t)
        # Should be normalized to [0, 0.5, 1.0]
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0])

    def test_linear_basis_handles_zero_max(self):
        """Linear basis should handle case where max(t) = 0."""
        bases = AVAILABLE_BASES["linear"]()
        t = np.array([0.0])
        result = bases[0].func(t)
        # Should not divide by zero
        assert np.isfinite(result[0])


class TestParseBasisSpec:
    """Tests for parsing basis specifications."""

    def test_parse_single_basis(self):
        """Test parsing a single basis name."""
        bases = parse_basis_spec("constant")
        assert len(bases) == 1
        assert bases[0].name == "constant"

    def test_parse_multiple_bases(self):
        """Test parsing multiple comma-separated bases."""
        bases = parse_basis_spec("constant,synodic")
        assert len(bases) == 3  # constant + synodic_cos + synodic_sin
        assert bases[0].name == "constant"
        assert bases[1].name == "synodic_cos"
        assert bases[2].name == "synodic_sin"

    def test_parse_with_whitespace(self):
        """Test parsing with whitespace around commas."""
        bases = parse_basis_spec(" constant , synodic , linear ")
        assert len(bases) == 4  # constant + synodic_cos + synodic_sin + linear

    def test_parse_case_insensitive(self):
        """Test parsing is case-insensitive."""
        bases = parse_basis_spec("CONSTANT,Synodic")
        assert len(bases) == 3

    def test_parse_unknown_basis_raises_error(self):
        """Test that unknown basis names raise ValueError."""
        with pytest.raises(ValueError, match="Unknown basis 'invalid'"):
            parse_basis_spec("constant,invalid")

    def test_parse_all_available_bases(self):
        """Test that all available bases can be parsed."""
        all_bases = ",".join(AVAILABLE_BASES.keys())
        bases = parse_basis_spec(all_bases)
        # Should get all basis functions
        assert len(bases) > 0


class TestGetBasisFuncByName:
    """Tests for getting basis functions by expanded name."""

    def test_get_constant_basis(self):
        """Test retrieving constant basis function."""
        func = _get_basis_func_by_name("constant")
        t = np.array([0.0, 10.0, 100.0])
        result = func(t)
        np.testing.assert_array_equal(result, np.ones(3))

    def test_get_linear_basis(self):
        """Test retrieving linear basis function."""
        func = _get_basis_func_by_name("linear")
        t = np.array([0.0, 50.0, 100.0])
        result = func(t)
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0])

    def test_get_synodic_cos_basis(self):
        """Test retrieving synodic cosine basis."""
        func = _get_basis_func_by_name("synodic_cos")
        period_hours = DEFAULT_SYNODIC_PERIOD_DAYS * 24.0
        t = np.array([0.0, period_hours / 4, period_hours / 2])
        result = func(t)
        np.testing.assert_allclose(result, [1, 0, -1], atol=1e-10)

    def test_get_synodic_sin_basis(self):
        """Test retrieving synodic sine basis."""
        func = _get_basis_func_by_name("synodic_sin")
        period_hours = DEFAULT_SYNODIC_PERIOD_DAYS * 24.0
        t = np.array([0.0, period_hours / 4])
        result = func(t)
        np.testing.assert_allclose(result, [0, 1], atol=1e-10)

    def test_get_sidereal_basis(self):
        """Test retrieving sidereal basis functions."""
        func_cos = _get_basis_func_by_name("sidereal_cos")
        func_sin = _get_basis_func_by_name("sidereal_sin")

        period_hours = SIDEREAL_PERIOD_DAYS * 24.0
        t = np.array([0.0, period_hours])

        # Should complete full cycle
        cos_vals = func_cos(t)
        np.testing.assert_allclose(cos_vals[0], cos_vals[1], atol=1e-10)

    def test_get_unknown_basis_raises_error(self):
        """Test that unknown basis names raise ValueError."""
        with pytest.raises(ValueError, match="Unknown expanded basis name"):
            _get_basis_func_by_name("invalid_basis")


class TestBuildTemporalDesign:
    """Tests for building temporal design matrices."""

    def test_build_design_single_basis(self):
        """Test building design matrix with single basis."""
        bases = [BasisFunction("constant", lambda t: np.ones_like(t))]
        t = np.array([0.0, 1.0, 2.0])

        design = build_temporal_design(t, bases)

        assert design.shape == (3, 1)
        np.testing.assert_array_equal(design[:, 0], np.ones(3))

    def test_build_design_multiple_bases(self):
        """Test building design matrix with multiple bases."""
        bases = [
            BasisFunction("constant", lambda t: np.ones_like(t)),
            BasisFunction("linear", lambda t: t),
        ]
        t = np.array([0.0, 1.0, 2.0])

        design = build_temporal_design(t, bases)

        assert design.shape == (3, 2)
        np.testing.assert_array_equal(design[:, 0], [1, 1, 1])
        np.testing.assert_array_equal(design[:, 1], [0, 1, 2])

    def test_build_design_preserves_dtype(self):
        """Test that design matrix has correct dtype."""
        bases = [BasisFunction("constant", lambda t: np.ones_like(t))]
        t = np.array([0.0, 1.0, 2.0])

        design = build_temporal_design(t, bases)

        assert design.dtype == np.float64


class TestFitTemporalBasis:
    """Tests for temporal basis fitting."""

    def test_fit_constant_potential_recovers_correct_value(self):
        """Test fitting a spatially uniform, temporally constant potential."""
        # Create synthetic data: uniform potential of 5.0 V everywhere
        n_points = 100
        np.random.seed(42)

        lat = np.random.uniform(-90, 90, n_points)
        lon = np.random.uniform(-180, 180, n_points)
        potential = np.full(n_points, 5.0)

        # Create UTC timestamps
        base_time = np.datetime64("1998-01-01T00:00:00")
        utc = base_time + np.arange(n_points) * np.timedelta64(1, "h")

        result = fit_temporal_basis(
            utc=utc,
            lat=lat,
            lon=lon,
            potential=potential,
            lmax=2,
            basis_spec="constant",
        )

        assert result.lmax == 2
        assert result.n_samples == n_points
        assert len(result.basis_names) == 1
        assert result.basis_names[0] == "constant"

        # RMS residual should be very small for perfect fit
        assert result.rms_residual < 0.1

    def test_fit_with_synodic_basis(self):
        """Test fitting with synodic temporal variation."""
        n_points = 200
        np.random.seed(42)

        lat = np.random.uniform(-90, 90, n_points)
        lon = np.random.uniform(-180, 180, n_points)

        # Create temporal variation with synodic period
        base_time = np.datetime64("1998-01-01T00:00:00")
        utc = base_time + np.arange(n_points) * np.timedelta64(1, "h")
        t_hours = np.arange(n_points, dtype=float)
        period_hours = DEFAULT_SYNODIC_PERIOD_DAYS * 24.0

        # Simple sinusoidal variation
        temporal_component = 2.0 * np.cos(2 * np.pi * t_hours / period_hours)
        potential = 5.0 + temporal_component

        result = fit_temporal_basis(
            utc=utc,
            lat=lat,
            lon=lon,
            potential=potential,
            lmax=1,
            basis_spec="constant,synodic",
        )

        assert len(result.basis_names) == 3  # constant, synodic_cos, synodic_sin
        # Should fit reasonably well
        assert result.rms_residual < 1.0

    def test_fit_with_l2_penalty(self):
        """Test fitting with L2 regularization."""
        n_points = 50
        np.random.seed(42)

        lat = np.random.uniform(-90, 90, n_points)
        lon = np.random.uniform(-180, 180, n_points)
        potential = np.random.uniform(0, 10, n_points)

        base_time = np.datetime64("1998-01-01T00:00:00")
        utc = base_time + np.arange(n_points) * np.timedelta64(1, "h")

        # Fit with and without penalty
        result_no_penalty = fit_temporal_basis(
            utc=utc, lat=lat, lon=lon, potential=potential, lmax=2,
            basis_spec="constant", l2_penalty=0.0
        )

        result_with_penalty = fit_temporal_basis(
            utc=utc, lat=lat, lon=lon, potential=potential, lmax=2,
            basis_spec="constant", l2_penalty=1.0
        )

        # With penalty, coefficients should have smaller magnitude
        coeff_magnitude_no_penalty = np.linalg.norm(result_no_penalty.basis_coeffs)
        coeff_magnitude_with_penalty = np.linalg.norm(result_with_penalty.basis_coeffs)

        assert coeff_magnitude_with_penalty < coeff_magnitude_no_penalty

    def test_fit_multiple_basis_functions(self):
        """Test fitting with multiple basis functions."""
        n_points = 150
        np.random.seed(42)

        lat = np.random.uniform(-90, 90, n_points)
        lon = np.random.uniform(-180, 180, n_points)
        potential = np.random.uniform(0, 10, n_points)

        base_time = np.datetime64("1998-01-01T00:00:00")
        utc = base_time + np.arange(n_points) * np.timedelta64(1, "h")

        result = fit_temporal_basis(
            utc=utc,
            lat=lat,
            lon=lon,
            potential=potential,
            lmax=2,
            basis_spec="constant,linear,synodic",
        )

        # Should have constant + linear + synodic_cos + synodic_sin = 4 bases
        assert len(result.basis_names) == 4
        assert "constant" in result.basis_names
        assert "linear" in result.basis_names
        assert "synodic_cos" in result.basis_names
        assert "synodic_sin" in result.basis_names


class TestReconstructAtTimes:
    """Tests for reconstructing coefficients at specific times."""

    def test_reconstruct_constant_basis(self):
        """Test reconstruction with constant basis returns same coeffs."""
        # Create a simple fit result
        basis_names = ["constant"]
        lmax = 1
        n_coeffs = (lmax + 1) ** 2
        basis_coeffs = np.random.randn(1, n_coeffs) + 1j * np.random.randn(1, n_coeffs)

        result = BasisFitResult(
            basis_names=basis_names,
            basis_coeffs=basis_coeffs,
            lmax=lmax,
            n_samples=100,
            rms_residual=1.0,
        )

        ref_time = np.datetime64("1998-01-01T00:00:00")
        times = ref_time + np.array([0, 24, 48]) * np.timedelta64(1, "h")

        reconstructed = reconstruct_at_times(result, times, ref_time)

        assert len(reconstructed) == 3
        # All should have same coefficients (constant basis)
        for harm_coeff in reconstructed:
            np.testing.assert_array_almost_equal(harm_coeff.coeffs, basis_coeffs[0])

    def test_reconstruct_linear_basis(self):
        """Test reconstruction with linear basis.

        Note: Linear basis normalizes by max(t), so when evaluating at a single
        time point, it returns t / max(t, 1.0), which is 0 for t=0 and 1 otherwise.
        """
        basis_names = ["linear"]
        lmax = 1
        n_coeffs = (lmax + 1) ** 2
        basis_coeffs = np.ones((1, n_coeffs), dtype=complex)

        result = BasisFitResult(
            basis_names=basis_names,
            basis_coeffs=basis_coeffs,
            lmax=lmax,
            n_samples=100,
            rms_residual=1.0,
        )

        ref_time = np.datetime64("1998-01-01T00:00:00")
        times = ref_time + np.array([0, 50, 100]) * np.timedelta64(1, "h")

        reconstructed = reconstruct_at_times(result, times, ref_time)

        # At t=0, basis evaluates to 0
        assert np.allclose(np.abs(reconstructed[0].coeffs), 0.0, atol=1e-10)
        # For t > 0, basis normalizes by its own max, giving 1.0
        assert np.allclose(np.abs(reconstructed[1].coeffs), 1.0, atol=1e-2)
        assert np.allclose(np.abs(reconstructed[2].coeffs), 1.0, atol=1e-2)

    def test_reconstruct_preserves_metadata(self):
        """Test that reconstruction preserves result metadata."""
        basis_names = ["constant"]
        lmax = 2
        n_coeffs = (lmax + 1) ** 2
        basis_coeffs = np.ones((1, n_coeffs), dtype=complex)
        n_samples = 150
        rms_residual = 2.5

        result = BasisFitResult(
            basis_names=basis_names,
            basis_coeffs=basis_coeffs,
            lmax=lmax,
            n_samples=n_samples,
            rms_residual=rms_residual,
        )

        ref_time = np.datetime64("1998-01-01T00:00:00")
        times = ref_time + np.array([0]) * np.timedelta64(1, "h")

        reconstructed = reconstruct_at_times(result, times, ref_time)

        assert reconstructed[0].lmax == lmax
        assert reconstructed[0].n_samples == n_samples
        assert reconstructed[0].rms_residual == rms_residual


class TestSaveBasisResult:
    """Tests for saving basis fit results."""

    def test_save_and_load_roundtrip(self, tmp_path):
        """Test saving and loading a basis fit result."""
        basis_names = ["constant", "synodic_cos", "synodic_sin"]
        lmax = 2
        n_coeffs = (lmax + 1) ** 2
        basis_coeffs = np.random.randn(3, n_coeffs) + 1j * np.random.randn(3, n_coeffs)

        result = BasisFitResult(
            basis_names=basis_names,
            basis_coeffs=basis_coeffs,
            lmax=lmax,
            n_samples=200,
            rms_residual=3.5,
        )

        output_path = tmp_path / "test_result.npz"
        save_basis_result(result, output_path)

        assert output_path.exists()

        # Load and verify
        loaded = np.load(output_path)
        np.testing.assert_array_equal(loaded["basis_names"], basis_names)
        np.testing.assert_array_almost_equal(loaded["basis_coeffs"], basis_coeffs)
        assert loaded["lmax"] == lmax
        assert loaded["n_samples"] == 200
        assert loaded["rms_residual"] == 3.5

    def test_save_creates_parent_directory(self, tmp_path):
        """Test that save_basis_result creates parent directories."""
        result = BasisFitResult(
            basis_names=["constant"],
            basis_coeffs=np.ones((1, 4), dtype=complex),
            lmax=1,
            n_samples=100,
            rms_residual=1.0,
        )

        output_path = tmp_path / "subdir1" / "subdir2" / "result.npz"
        save_basis_result(result, output_path)

        assert output_path.exists()
        assert output_path.parent.exists()
