"""Tests for src/utils/spice_ops.py"""

import numpy as np
import pytest

from src import config
from src.utils import spice_ops


def test_get_lp_position_wrt_moon_batch_with_mock(monkeypatch):
    """Test batch function matches scalar function with mocked SPICE calls."""

    # Mock SPICE to return predictable values
    call_log = []

    def mock_spkpos(target, time, frame, abcorr, observer):
        call_log.append((target, time, frame, abcorr, observer))
        # Return position that depends on time
        return np.array([time * 100, time * 10, time]), None

    monkeypatch.setattr("spiceypy.spkpos", mock_spkpos)

    # Test batch function
    times = np.array([0.0, 1.0, 2.0, 5.0])
    batch_result = spice_ops.get_lp_position_wrt_moon_batch(times)
    assert all(
        target == spice_ops.LP
        and frame == spice_ops.IAU_MOON
        and abcorr == "NONE"
        and observer == spice_ops.MOON
        for target, _time, frame, abcorr, observer in call_log
    )

    # Compare with individual scalar calls
    call_log.clear()
    scalar_results = []
    for t in times:
        result = spice_ops.get_lp_position_wrt_moon(t)
        scalar_results.append(result)
    scalar_results = np.array(scalar_results)

    # Should match
    np.testing.assert_allclose(batch_result, scalar_results, rtol=1e-10)
    assert batch_result.shape == (4, 3)


def test_get_lp_vector_to_sun_batch_with_mock(monkeypatch):
    """Test batch LP->Sun vector function matches scalar."""

    call_log = []

    def mock_spkpos(target, time, frame, abcorr, observer):
        call_log.append((target, frame, abcorr, observer))
        # Sun vector depends on time
        return np.array([time * 50, time * 5, -time]), None

    monkeypatch.setattr("spiceypy.spkpos", mock_spkpos)

    times = np.array([0.5, 1.5, 3.0])
    batch_result = spice_ops.get_lp_vector_to_sun_in_lunar_frame_batch(times)
    assert all(
        target == spice_ops.SUN
        and frame == spice_ops.IAU_MOON
        and abcorr == "NONE"
        and observer == spice_ops.LP
        for target, frame, abcorr, observer in call_log
    )

    scalar_results = []
    for t in times:
        result = spice_ops.get_lp_vector_to_sun_in_lunar_frame(t)
        scalar_results.append(result)
    scalar_results = np.array(scalar_results)

    np.testing.assert_allclose(batch_result, scalar_results, rtol=1e-10)


def test_get_sun_vector_wrt_moon_batch_with_mock(monkeypatch):
    """Test batch Moon->Sun vector function matches scalar."""

    call_log = []

    def mock_spkpos(target, time, frame, abcorr, observer):
        call_log.append((target, frame, abcorr, observer))
        return np.array([time * 1000, time * 100, time * 10]), None

    monkeypatch.setattr("spiceypy.spkpos", mock_spkpos)

    times = np.array([1.0, 2.0, 4.0, 8.0])
    batch_result = spice_ops.get_sun_vector_wrt_moon_batch(times)
    assert all(
        target == spice_ops.SUN
        and frame == spice_ops.IAU_MOON
        and abcorr == "NONE"
        and observer == spice_ops.MOON
        for target, frame, abcorr, observer in call_log
    )

    scalar_results = []
    for t in times:
        result = spice_ops.get_sun_vector_wrt_moon(t)
        scalar_results.append(result)
    scalar_results = np.array(scalar_results)

    np.testing.assert_allclose(batch_result, scalar_results, rtol=1e-10)


def test_get_lp_vector_to_sun_in_eclipj2000_requests_attitude_frame(
    monkeypatch,
):
    """The SCD-defining Sun vector must share the PDS attitude frame."""
    calls = []

    def mock_spkpos(target, time, frame, abcorr, observer):
        calls.append((target, time, frame, abcorr, observer))
        return np.array([1.0, 2.0, 3.0]), None

    monkeypatch.setattr("spiceypy.spkpos", mock_spkpos)

    result = spice_ops.get_lp_vector_to_sun_in_eclipj2000_batch(np.array([1.0, 2.0]))

    np.testing.assert_allclose(result, [[1.0, 2.0, 3.0]] * 2)
    assert calls == [
        (spice_ops.SUN, 1.0, spice_ops.ECLIPJ2000, "NONE", spice_ops.LP),
        (spice_ops.SUN, 2.0, spice_ops.ECLIPJ2000, "NONE", spice_ops.LP),
    ]


def test_get_j2000_iau_moon_transform_matrix_batch_with_mock(monkeypatch):
    """Test batch transform matrix function matches scalar."""

    def mock_pxform(from_frame, to_frame, time):
        # Return rotation matrix that depends on time
        angle = time * 0.1
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    monkeypatch.setattr("spiceypy.pxform", mock_pxform)

    times = np.array([0.0, 1.0, 2.0])
    batch_result = spice_ops.get_j2000_iau_moon_transform_matrix_batch(times)

    scalar_results = []
    for t in times:
        result = spice_ops.get_j2000_iau_moon_transform_matrix(t)
        scalar_results.append(result)
    scalar_results = np.array(scalar_results)

    np.testing.assert_allclose(batch_result, scalar_results, rtol=1e-10)
    assert batch_result.shape == (3, 3, 3)


def test_get_eclipj2000_iau_moon_transform_requests_named_frames(monkeypatch):
    calls = []

    def mock_pxform(from_frame, to_frame, time):
        calls.append((from_frame, to_frame, time))
        return np.eye(3)

    monkeypatch.setattr("spiceypy.pxform", mock_pxform)

    result = spice_ops.get_eclipj2000_iau_moon_transform_matrix_batch(
        np.array([1.0, 2.0])
    )

    np.testing.assert_allclose(result, np.tile(np.eye(3), (2, 1, 1)))
    assert calls == [
        (spice_ops.ECLIPJ2000, spice_ops.IAU_MOON, 1.0),
        (spice_ops.ECLIPJ2000, spice_ops.IAU_MOON, 2.0),
    ]


def test_batch_functions_handle_errors(monkeypatch):
    """Test that batch functions return NaN for failed calls."""

    def mock_spkpos_with_errors(target, time, frame, abcorr, observer):
        if time == 2.0:
            raise RuntimeError("SPICE error")
        return np.array([time, time, time]), None

    monkeypatch.setattr("spiceypy.spkpos", mock_spkpos_with_errors)

    times = np.array([1.0, 2.0, 3.0])
    result = spice_ops.get_lp_position_wrt_moon_batch(times)

    # First and third should be valid
    assert np.all(np.isfinite(result[0]))
    assert np.all(np.isfinite(result[2]))
    # Second should be NaN
    assert np.all(np.isnan(result[1]))


@pytest.mark.skip_ci
def test_batch_spice_ops_with_real_kernels():
    """
    Integration test with real SPICE kernels.
    Requires SPICE kernels to be loaded.
    """
    # Note: This test requires kernels to be available
    # It will be skipped in CI but can run locally

    from src.potential_mapper.spice import load_spice_files

    # Check if kernels exist before trying to load
    spice_dir = config.SPICE_KERNELS_DIR
    patterns = [f"*{config.EXT_BSP}", f"*{config.EXT_TPC}", f"*{config.EXT_TLS}"]
    has_kernels = False
    if spice_dir.exists():
        for pattern in patterns:
            if list(spice_dir.glob(pattern)):
                has_kernels = True
                break

    if not has_kernels:
        pytest.skip("No SPICE kernels found")

    # Load kernels
    load_spice_files()

    # Use real times from LP mission (1998)
    # J2000 epoch: 2000-01-01 12:00:00
    # LP mission roughly -63158400 to -31536000 (1998-01 to 1999-01)
    times = np.array([-60000000.0, -55000000.0, -50000000.0])

    # Test all batch functions against scalar
    batch_pos = spice_ops.get_lp_position_wrt_moon_batch(times)
    batch_lp_sun = spice_ops.get_lp_vector_to_sun_in_lunar_frame_batch(times)
    batch_lp_sun_eclip = spice_ops.get_lp_vector_to_sun_in_eclipj2000_batch(times)
    batch_moon_sun = spice_ops.get_sun_vector_wrt_moon_batch(times)
    batch_mats = spice_ops.get_j2000_iau_moon_transform_matrix_batch(times)
    batch_eclip_mats = spice_ops.get_eclipj2000_iau_moon_transform_matrix_batch(times)

    for i, t in enumerate(times):
        scalar_pos = spice_ops.get_lp_position_wrt_moon(t)
        scalar_lp_sun = spice_ops.get_lp_vector_to_sun_in_lunar_frame(t)
        scalar_moon_sun = spice_ops.get_sun_vector_wrt_moon(t)
        scalar_mat = spice_ops.get_j2000_iau_moon_transform_matrix(t)
        lp_j2000, _ = spice_ops.spice.spkpos(
            spice_ops.LP, t, spice_ops.J2000, "NONE", spice_ops.MOON
        )
        lp_expected = spice_ops.spice.pxform(
            spice_ops.J2000, spice_ops.IAU_MOON, t
        ) @ np.asarray(lp_j2000)
        lp_sun_eclip, _ = spice_ops.spice.spkpos(
            spice_ops.SUN, t, spice_ops.ECLIPJ2000, "NONE", spice_ops.LP
        )

        # All should match
        np.testing.assert_allclose(batch_pos[i], scalar_pos, rtol=1e-10)
        np.testing.assert_allclose(batch_pos[i], lp_expected, rtol=1e-10)
        np.testing.assert_allclose(batch_lp_sun[i], scalar_lp_sun, rtol=1e-10)
        np.testing.assert_allclose(batch_lp_sun_eclip[i], lp_sun_eclip, rtol=1e-10)
        np.testing.assert_allclose(batch_moon_sun[i], scalar_moon_sun, rtol=1e-10)
        np.testing.assert_allclose(batch_mats[i], scalar_mat, rtol=1e-10)
        np.testing.assert_allclose(
            batch_eclip_mats[i],
            spice_ops.spice.pxform(spice_ops.ECLIPJ2000, spice_ops.IAU_MOON, t),
            rtol=1e-10,
        )
