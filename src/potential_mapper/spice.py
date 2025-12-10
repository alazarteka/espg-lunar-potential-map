import logging

import spiceypy as spice

import src.config as config


def load_spice_files() -> None:
    """
    Load SPICE kernels from `config.SPICE_KERNELS_DIR` (bsp/tpc/tls patterns).

    Safe to call multiple times in a process; relies on SPICE to deduplicate
    loaded kernels. Raises FileNotFoundError if directory or files are missing.
    """
    spice_dir = config.SPICE_KERNELS_DIR
    if not spice_dir.exists():
        raise FileNotFoundError(f"SPICE directory {spice_dir} not found")

    patterns = [
        f"*{config.EXT_BSP}",
        f"*{config.EXT_TPC}",
        f"*{config.EXT_TLS}",
    ]
    spice_files = []
    for pattern in patterns:
        spice_files.extend(spice_dir.glob(pattern))

    if not spice_files:
        raise FileNotFoundError(f"No SPICE files found in {spice_dir}")

    logging.debug(f"Found {len(spice_files)} SPICE files in {spice_dir}. Loading...")
    for spice_file in spice_files:
        try:
            logging.debug(f"Loading SPICE file: {spice_file}")
            spice.furnsh(str(spice_file))
        except Exception as e:
            logging.error(f"Error loading SPICE file {spice_file}: {e}")
            raise

    # Let Python handle SPICE errors via exceptions
    # (erract RETURN makes SPICE return control to spiceypy which raises SpiceyError)
    spice.erract("SET", 10, "RETURN")
