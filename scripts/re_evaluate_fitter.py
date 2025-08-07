
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import src.config as config
from src.flux import ERData
from src.kappa import Kappa
from src.utils.units import ureg

# Hardcode a data file for analysis
root = config.PROJECT_ROOT
data_files = glob.glob(str(config.DATA_DIR / "199*" / "*" / "*"))
sample_data_path = data_files[0]

er_data = ERData(sample_data_path)

# Run the fitting process for a single spectrum
spec_no = er_data.data["spec_no"].unique()[0]
kappa = Kappa(er_data, spec_no=spec_no)
density = kappa.density_estimate.to(ureg.particle / ureg.m**3).magnitude
fitted, error = kappa.fit(n_starts=10, use_fast=True, use_weights=True)

# Print the results
print(f"Results for spectrum {spec_no} in file {sample_data_path}:")
print(f"  - Density: {density:.4e} particles/m^3")
print(f"  - Kappa: {fitted.kappa:.4f}")
print(f"  - Theta: {fitted.theta.to(ureg.meter / ureg.second).magnitude:.4e} m/s")
print(f"  - Fitting Error (chi-squared): {error:.4f}")
