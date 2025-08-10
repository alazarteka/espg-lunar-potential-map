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
    kappas = []
    thetas = []

    spec_numbers = er_data.data["spec_no"].unique()

    for i in spec_numbers:
        try:
            kappa = Kappa(er_data, spec_no=i)
            density = kappa.density_estimate.to(ureg.particle / ureg.m**3).magnitude
            densities.append(density)

            if not kappa.is_data_valid:
                continue

            fit_results = kappa.fit(n_starts=10)
            if fit_results is None:
                continue

            kappas.append(fit_results.params.kappa)
            thetas.append(
                fit_results.params.theta.to(ureg.meter / ureg.second).magnitude
            )

        except Exception:
            pass

    end_time = time.time()

    pr.disable()
    pr.dump_stats(config.PROJECT_ROOT / "temp" / "kappa_fitting_profile.prof")

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

    return densities, kappas, thetas


if __name__ == "__main__":
    profile_kappa_fitting()
