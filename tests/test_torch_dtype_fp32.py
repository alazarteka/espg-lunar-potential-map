"""Regression tests pinning float32 as the automatic torch dtype.

Physics review finding F1 (``scratch/review-2026-07-25/01-physics.md``): both
``src/losscone/torch/fitter.py._auto_detect_dtype`` and
``src/kappa_torch.py._auto_detect_dtype`` used to return ``torch.float16`` on any
CUDA device with compute capability >= 7.0. In fp16 the pitch grid and
``180 - alpha_c`` quantize to ~0.125 deg, shifting the loss-cone boundary by a
whole pitch bin: the Halekas chi2 measured 18.078 in fp16 against 9.108 in both
fp32 and fp64. Since the boundary position *is* the fitted surface potential,
production results were silently wrong and hardware-dependent.

Production reaches these via ``dtype="auto"`` (``potential_mapper/pipeline.py``
constructs ``LossConeFitterTorch`` with no ``dtype``), so "auto" must never
select fp16 again on any device.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src import config  # noqa: E402
from src.utils.synthetic import prepare_synthetic_er  # noqa: E402

# (major, minor) compute capabilities spanning pre-Volta through Blackwell.
# Every one of these must yield float32; the old code returned float16 for
# anything with major >= 7.
_CAPABILITIES = [(6, 1), (7, 0), (7, 5), (8, 0), (8, 6), (9, 0), (10, 0)]


@pytest.mark.parametrize("major, minor", _CAPABILITIES)
@pytest.mark.parametrize("device_type", ["cpu", "cuda"])
def test_losscone_auto_detect_dtype_is_float32(
    monkeypatch, device_type: str, major: int, minor: int
) -> None:
    """The loss-cone fitter's auto dtype is float32 for every mocked device."""
    from src.losscone.torch import fitter as fitter_mod

    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(major=major, minor=minor),
        raising=False,
    )
    device = torch.device(device_type)
    assert fitter_mod._auto_detect_dtype(device) is torch.float32


@pytest.mark.parametrize("major, minor", _CAPABILITIES)
@pytest.mark.parametrize("device_type", ["cpu", "cuda"])
def test_kappa_auto_detect_dtype_is_float32(
    monkeypatch, device_type: str, major: int, minor: int
) -> None:
    """The kappa fitter's auto dtype is float32 for every mocked device.

    fp16 is doubly fatal here: ``fit_batch`` casts electron densities of order
    1e6-1e8 m^-3 to this dtype and the fp16 maximum is 65504, so every density
    would overflow to inf.
    """
    import src.kappa_torch as kappa_torch_mod

    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(major=major, minor=minor),
        raising=False,
    )
    device = torch.device(device_type)
    assert kappa_torch_mod._auto_detect_dtype(device) is torch.float32


def test_auto_detect_dtype_never_calls_cuda_properties(monkeypatch) -> None:
    """Auto dtype no longer branches on device capability at all."""
    from src.losscone.torch import fitter as fitter_mod

    def _boom(_device):  # pragma: no cover - must not be reached
        raise AssertionError("auto dtype must not inspect device properties")

    monkeypatch.setattr(torch.cuda, "get_device_properties", _boom, raising=False)
    assert fitter_mod._auto_detect_dtype(torch.device("cuda")) is torch.float32


def test_fp16_density_overflow_justifies_the_fix() -> None:
    """Document why fp16 cannot carry this pipeline's electron densities."""
    density_m3 = 1.0e6  # a modest lunar-plasma electron density
    assert torch.tensor(density_m3, dtype=torch.float16).isinf()
    assert torch.isfinite(torch.tensor(density_m3, dtype=torch.float32))


def _synthetic_fitter_inputs() -> tuple[object, SimpleNamespace]:
    """Minimal ERData + pitch-angle stub for constructing a torch fitter."""
    er = prepare_synthetic_er()
    pitch_1d = np.linspace(0.0, 180.0, config.CHANNELS, dtype=np.float64)
    pitches = np.broadcast_to(pitch_1d[None, :], (config.SWEEP_ROWS, config.CHANNELS))
    return er, SimpleNamespace(pitch_angles=pitches)


def test_losscone_fitter_torch_default_dtype_resolves_to_float32() -> None:
    """A default-constructed LossConeFitterTorch runs in float32.

    This is the exact construction production uses (no ``dtype`` argument, i.e.
    ``dtype="auto"``).
    """
    from src.losscone_torch import LossConeFitterTorch

    er, pitch_angle = _synthetic_fitter_inputs()
    fitter = LossConeFitterTorch(er, pitch_angle=pitch_angle, device="cpu")
    assert fitter.dtype is torch.float32


def test_losscone_fitter_torch_explicit_auto_matches_default() -> None:
    """Passing dtype="auto" explicitly gives the same float32."""
    from src.losscone_torch import LossConeFitterTorch

    er, pitch_angle = _synthetic_fitter_inputs()
    fitter = LossConeFitterTorch(
        er, pitch_angle=pitch_angle, device="cpu", dtype="auto"
    )
    assert fitter.dtype is torch.float32


def test_losscone_fitter_torch_explicit_float64_still_honored() -> None:
    """The fp32 default must not clobber an explicit high-precision request."""
    from src.losscone_torch import LossConeFitterTorch

    er, pitch_angle = _synthetic_fitter_inputs()
    fitter = LossConeFitterTorch(
        er, pitch_angle=pitch_angle, device="cpu", dtype="float64"
    )
    assert fitter.dtype is torch.float64


def test_kappa_fitter_torch_default_dtype_resolves_to_float32() -> None:
    """A default-constructed KappaFitterTorch runs in float32."""
    from src.kappa_torch import KappaFitterTorch

    assert KappaFitterTorch(device="cpu").dtype is torch.float32


def test_kappa_fitter_torch_rejects_unknown_dtype() -> None:
    """An unrecognized dtype string used to fall through to float16 silently."""
    from src.kappa_torch import KappaFitterTorch

    with pytest.raises(ValueError, match="dtype must be"):
        KappaFitterTorch(device="cpu", dtype="f32")
