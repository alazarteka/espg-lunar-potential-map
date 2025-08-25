import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from . import config

logger = logging.getLogger(__name__)

# Create a session for connection reuse
session = requests.Session()
adapter = requests.adapters.HTTPAdapter(
    pool_connections=config.CONNECTION_POOL_SIZE,
    pool_maxsize=config.CONNECTION_POOL_SIZE,
    max_retries=3,
)
session.mount("http://", adapter)
session.mount("https://", adapter)


def solid_angle_from_thetas(base_dir: Path) -> None:
    """
    Save the solid angle for a given latitude in degrees.

    Args:
        base_dir (Path): Path to the directory where the solid angles will be saved.

    We could not find a file in the PDS that contains this information in a
    format we can use, so we hardcode the values for the four latitudes.

    Source: https://pds-ppi.igpp.ucla.edu/data/lp-er-calibrated/document/data-3deleflux-desc.txt
    The values can be found in the "Parameters" section.

    The formatting is identical to the one used in the thetas.tab file.
    """

    latitude_to_area = {
        78.75: 0.119570,
        56.25: 0.170253,
        33.75: 0.127401,
        11.25: 0.150279,
    }

    thetas = np.loadtxt(base_dir / config.THETA_FILE, dtype=float)
    solid_angles = list(map(lambda x: latitude_to_area[abs(x)], thetas))
    np.savetxt(
        base_dir / config.SOLID_ANGLES_FILE, solid_angles, fmt="%.6f", delimiter=" "
    )


class DataManager:
    """
    Manage downloading and organizing data files by scraping directory indexes.

    Features:
    - List remote directories and files
    - Download all .TAB (or specified extension) files in a given directory
    - Skip existing downloads
    """

    def __init__(self, base_dir: str, base_url: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = base_url.rstrip("/")

    def ensure_dir(self, *subdirs) -> Path:
        path = self.base_dir.joinpath(*subdirs)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def list_remote_dirs(self, remote_path: str = "") -> list[str]:
        """
        Return a list of subdirectory names under base_url/remote_path/.
        """
        url = f"{self.base_url}/{remote_path.rstrip('/')}/"
        logger.debug(f"Listing remote directories at {url}")
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        dirs = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            # directory links end with '/'
            if href.endswith("/") and href not in ("../", "./"):
                dirs.append(href.rstrip("/"))

        if len(dirs):
            dirs = dirs[1:]
        return dirs

    def list_remote_files(
        self, remote_path: str, ext: str = config.EXT_TAB
    ) -> list[str]:
        """
        Return filenames under base_url/remote_path/ matching ext.
        """
        url = f"{self.base_url}/{remote_path.rstrip('/')}/"
        logger.debug(f"Listing remote files at {url}")
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        files = [
            a["href"]
            for a in soup.find_all("a", href=True)
            if a["href"].upper().endswith(ext.upper())
        ]
        return files

    def download_file(
        self, url: str, dest: Path, chunk_size: int = config.CHUNK_SIZE_BYTES
    ) -> Path:
        """
        Stream-download a file if not already present.
        """
        dest = Path(dest)
        temp_dest = dest.with_suffix(dest.suffix + ".part")

        if dest.exists():
            logger.debug(f"Skipping existing file: {dest}")
            return dest

        if temp_dest.exists():
            logger.debug("Partial download found, resuming...")
        else:
            logger.debug(f"Starting download: {url} -> {temp_dest}")

        try:
            resp = session.get(url, stream=True, timeout=60)
            resp.raise_for_status()
            with open(temp_dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size):
                    if chunk:
                        f.write(chunk)
            temp_dest.rename(dest)
            logger.debug(f"Downloaded: {dest}")
            return dest
        except Exception as e:
            if temp_dest.exists():
                temp_dest.unlink()
            raise e

    def fetch_directory(
        self, remote_path: str, ext: str = config.EXT_TAB
    ) -> list[Path]:
        """
        Download all files with the given extension in base_url/remote_path/.
        """
        files = self.list_remote_files(remote_path, ext)
        local_dir = self.ensure_dir(*remote_path.split("/"))

        urls_and_dests = [
            (f"{self.base_url}/{remote_path}/{fname}", local_dir / fname)
            for fname in files
        ]
        self.download_files_in_parallel(
            urls_and_dests, folder_desc=f"Downloading {remote_path}"
        )
        return [dest for url, dest in urls_and_dests]

    def collect_all_download_tasks(
        self, years: list[str], ext: str = config.EXT_TAB
    ) -> list[tuple[str, Path]]:
        """
        Collect all download tasks across all years and julian days without downloading.
        This allows for better parallelization.
        """
        all_tasks = []

        for year in tqdm(years, desc="Collecting download tasks"):
            julian_dirs = self.list_remote_dirs(year)
            for julian in julian_dirs:
                remote_path = f"{year}/{julian}"
                try:
                    files = self.list_remote_files(remote_path, ext)
                    local_dir = self.ensure_dir(*remote_path.split("/"))

                    year_julian_tasks = [
                        (f"{self.base_url}/{remote_path}/{fname}", local_dir / fname)
                        for fname in files
                    ]
                    all_tasks.extend(year_julian_tasks)
                except Exception as e:
                    logger.warning(f"Failed to list files in {remote_path}: {e}")
                    continue

        return all_tasks

    def download_files_in_parallel(
        self,
        urls_and_dests: list[tuple[str, Path]],
        max_workers: int = config.MAX_DOWNLOAD_WORKERS,
        folder_desc: str = "Downloading files",
    ) -> None:
        """
        Download multiple (url, dest) pairs in parallel.
        """
        # Filter out already existing files
        remaining_tasks = [
            (url, dest) for url, dest in urls_and_dests if not dest.exists()
        ]

        if not remaining_tasks:
            logger.info("All files already exist, skipping downloads")
            return

        existing = len(urls_and_dests) - len(remaining_tasks)
        logger.info(
            "Downloading %d files (%d already exist)",
            len(remaining_tasks),
            existing,
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            with tqdm(
                total=len(remaining_tasks), desc=folder_desc, unit="file"
            ) as pbar:
                futures = [
                    executor.submit(self.download_file, url, dest)
                    for url, dest in remaining_tasks
                ]
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Download failed: {e}")
                    finally:
                        pbar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Lunar Prospector data files."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Increase output verbosity to DEBUG.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    spice_mgr = DataManager(
        base_dir=str(config.SPICE_KERNELS_DIR),
        base_url="https://naif.jpl.nasa.gov/pub/naif/LPM/kernels/spk/",
    )
    generic_mgr = DataManager(
        base_dir=str(config.SPICE_KERNELS_DIR),
        base_url="https://naif.jpl.nasa.gov/pub/naif/generic_kernels/",
    )
    lpephemu_mgr = DataManager(
        base_dir=str(config.SPICE_KERNELS_DIR),
        base_url="https://pds-geosciences.wustl.edu/missions/lunarp/spice/",
    )
    attitude_mgr = DataManager(
        base_dir=str(config.DATA_DIR),
        base_url="https://pds-geosciences.wustl.edu/lunar/prospectorcd/lp_0019/geometry/",
    )
    data_mgr = DataManager(
        base_dir=str(config.DATA_DIR),
        base_url="https://pds-ppi.igpp.ucla.edu/data/lp-er-calibrated/data-3deleflux/",
    )
    theta_mgr = DataManager(
        base_dir=str(config.DATA_DIR),
        base_url="https://pds-ppi.igpp.ucla.edu/data/lp-er-calibrated/data-3deleflux-anc/theta/",
    )

    # download specific spice kernels
    logger.info("Downloading specific SPICE kernels...")
    spice_files = [
        "lp_ask_980111-980531.bsp",
        "lp_ask_980601-981031.bsp",
        "lp_ask_981101-990331.bsp",
        "lp_ask_990401-990730.bsp",
    ]
    spice_tasks = [
        (f"{spice_mgr.base_url}/{fname}", spice_mgr.base_dir / fname)
        for fname in spice_files
    ]

    # download generic kernels
    logger.info("Downloading generic kernels...")
    generic_files = [
        "lsk/latest_leapseconds.tls",
        "pck/pck00011.tpc",
    ]
    generic_tasks = [
        (f"{generic_mgr.base_url}/{fname}", spice_mgr.base_dir / Path(fname).name)
        for fname in generic_files
    ]

    # download lpephemu
    logger.info("Downloading lpephemu kernel...")
    lpephemu_tasks = [
        (f"{lpephemu_mgr.base_url}/lpephemu.bsp", spice_mgr.base_dir / "lpephemu.bsp")
    ]

    # download attitude table
    logger.info("Downloading attitude table...")
    attitude_tasks = [
        (
            f"{attitude_mgr.base_url}/{config.ATTITUDE_FILE}",
            data_mgr.base_dir / config.ATTITUDE_FILE,
        )
    ]

    # download theta files
    logger.info("Downloading theta file...")
    theta_tasks = [
        (
            f"{theta_mgr.base_url}/{config.THETA_FILE}",
            data_mgr.base_dir / config.THETA_FILE,
        )
    ]

    # Combine all initial downloads and run in parallel
    initial_tasks = (
        spice_tasks + generic_tasks + lpephemu_tasks + attitude_tasks + theta_tasks
    )
    spice_mgr.download_files_in_parallel(
        initial_tasks, folder_desc="Downloading initial files"
    )

    # Generate solid angles after theta file is downloaded
    logger.info("Generating solid angles...")
    solid_angle_from_thetas(data_mgr.base_dir)

    # download data files
    logger.info("Collecting all 3D electron flux data download tasks...")
    years = data_mgr.list_remote_dirs()
    logger.info(f"Found {len(years)} years of data")

    # Collect all download tasks first
    all_download_tasks = data_mgr.collect_all_download_tasks(years)
    logger.info(f"Found {len(all_download_tasks)} total files to potentially download")

    # Download all files in parallel across all years/julian days
    data_mgr.download_files_in_parallel(
        all_download_tasks,
        max_workers=config.MAX_DOWNLOAD_WORKERS,
        folder_desc="Downloading all 3D electron flux data",
    )

    logger.info("All downloads completed.")
