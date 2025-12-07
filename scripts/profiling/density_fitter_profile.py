"""Profile execution of the density estimation helper."""

import cProfile
import pstats
import time
from io import StringIO

import src.config as config
from src.flux import ERData
from src.kappa import Kappa
from src.utils.units import ureg


def profile_kappa_fitting():
    """Profile the kappa fitting process from the notebook."""

    sample_data_path = config.DATA_DIR / "1998/091_120APR/3D980401.TAB"
    er_data = ERData(str(sample_data_path))

    pr = cProfile.Profile()
    pr.enable()

    start_time = time.time()

    densities = []

    spec_numbers = er_data.data[
        "spec_no"
    ].unique()  # Limit to first 50 spectra for profiling

    for i in spec_numbers:
        kappa = Kappa(er_data, spec_no=i)
        # density = kappa._get_density_estimate().to(ureg.particle / ureg.m**3).magnitude
        densities.append(kappa.density_estimate.to(ureg.particle / ureg.m**3).magnitude)

    end_time = time.time()

    pr.disable()
    profile_path = (
        config.PROJECT_ROOT / "scratch" / "profiles" / "density_fitting_profile.prof"
    )
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    pr.dump_stats(profile_path)

    s = StringIO()
    ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats("cumulative")
    ps.print_stats(30)

    print(f"Processing time: {end_time - start_time:.2f} seconds")
    print(f"Processed {len(spec_numbers)} spectra")
    print(
        f"Average time per spectrum: {(end_time - start_time) / len(spec_numbers):.3f} seconds"
    )
    print("\nTop functions by cumulative time:")
    print(s.getvalue())

    return densities


if __name__ == "__main__":
    profile_kappa_fitting()
