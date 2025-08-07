

import glob
import matplotlib.pyplot as plt
import numpy as np
import src.config as config
from src.flux import ERData
from src.kappa import Kappa

# Get all data files
data_files = glob.glob(str(config.DATA_DIR / "199*" / "*" / "*.TAB"))

fitting_errors = []

# Process each data file
for data_file in data_files:
    print(f"Processing file: {data_file}")
    try:
        er_data = ERData(data_file)
        # Process each spectrum in the file
        for spec_no in er_data.data["spec_no"].unique():
            try:
                kappa = Kappa(er_data, spec_no=spec_no)
                if kappa.is_data_valid:
                    fitted, error = kappa.fit(n_starts=10, use_fast=True, use_weights=True)
                    if error is not None:
                        fitting_errors.append(error)
            except Exception as e:
                print(f"  Error processing spectrum {spec_no}: {e}")
    except Exception as e:
        print(f"Error processing file {data_file}: {e}")

# Plot the histogram of fitting errors
plt.figure(figsize=(10, 6))
plt.hist(fitting_errors, bins=100, log=True)
plt.title("Distribution of Fitting Errors (Chi-Squared)")
plt.xlabel("Chi-Squared")
plt.ylabel("Frequency (log scale)")
plt.grid(True)
plt.savefig("temp/error_distribution.png")

# Print some statistics
fitting_errors = np.array(fitting_errors)
print("\n--- Error Distribution Statistics ---")
print(f"Number of successful fits: {len(fitting_errors)}")
print(f"Mean error: {np.mean(fitting_errors):.2f}")
print(f"Median error: {np.median(fitting_errors):.2f}")
print(f"95th percentile: {np.percentile(fitting_errors, 95):.2f}")
print(f"99th percentile: {np.percentile(fitting_errors, 99):.2f}")

