# Lunar Prospector Plasma Analysis

## Description

This project analyzes data from the Lunar Prospector mission to model plasma flux and map lunar surface potential. It utilizes data from the NASA Planetary Data System and SPICE kernels for trajectory and orientation information.

Currently in active development—major revisions ongoing.

## Project Structure

```
.
├── README.md               # This file
├── data.py                 # Handles data and SPICE kernel acquisition
├── flux.py                 # Flux and potential calculation logic
├── model.py                # Core modeling components
├── potential_mapper.py     # Potential mapping logic
├── utils.py                # Utility functions
├── environment-base.yml    # Conda environment definition
├── data/                   # Processed data (created/populated by data.py)
└── spice_kernels/          # SPICE kernel files (created/populated by data.py)
```

## Installation

1.  **Set up the environment:**
    Ensure you have Anaconda installed. 
    ```bash
    conda env create -f environment-base.yml
    conda activate urp-mapping
    ```

2.  **Download necessary data:**
    *   With the conda environment active, run `data.py`
    ```bash
    python data.py
    ```


## Data

*   Data required by the core modules are expected in `data/`.
*   SPICE kernels are expected in `spice_kernels/`.
*   Running `data.py` will download the files in the appropriate directory if they are not locally available.

## Usage

After setting up and downloading data, run the desired module. For example:
```bash
python potential_mapper.py
```