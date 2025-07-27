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
        resp = requests.get(url)
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

    def list_remote_files(self, remote_path: str, ext: str = ".TAB") -> list[str]:
        """
        Return filenames under base_url/remote_path/ matching ext.
        """
        url = f"{self.base_url}/{remote_path.rstrip('/')}/"
        logger.debug(f"Listing remote files at {url}")
        resp = requests.get(url)
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

        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        with open(temp_dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size):
                if chunk:
                    f.write(chunk)
        temp_dest.rename(dest)
        logger.debug(f"Downloaded: {dest}")
        return dest

    def fetch_directory(self, remote_path: str, ext: str = ".TAB") -> list[Path]:
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

    def download_files_in_parallel(
        self,
        urls_and_dests: list[tuple[str, Path]],
        max_workers: int = config.MAX_DOWNLOAD_WORKERS,
        folder_desc: str = "Downloading files",
    ) -> None:
        """
        Download multiple (url, dest) pairs in parallel.
        """
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            with tqdm(total=len(urls_and_dests), desc=folder_desc, leave=False) as pbar:
                futures = [
                    executor.submit(self.download_file, url, dest)
                    for url, dest in urls_and_dests
                ]
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Downloading files",
                    unit="file",
                ):
                    future.result()
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
        base_dir=config.KERNELS_DIR,
        base_url="https://naif.jpl.nasa.gov/pub/naif/LPM/kernels/spk/",
    )
    generic_mgr = DataManager(
        base_dir=config.KERNELS_DIR,
        base_url="https://naif.jpl.nasa.gov/pub/naif/generic_kernels/",
    )
    lpephemu_mgr = DataManager(
        base_dir=config.KERNELS_DIR,
        base_url="https://pds-geosciences.wustl.edu/missions/lunarp/spice/",
    )
    attitude_mgr = DataManager(
        base_dir=config.DATA_DIR,
        base_url="https://pds-geosciences.wustl.edu/lunar/prospectorcd/lp_0019/geometry/",
    )
    data_mgr = DataManager(
        base_dir=config.DATA_DIR,
        base_url="https://pds-ppi.igpp.ucla.edu/data/lp-er-calibrated/data-3deleflux/",
    )
    theta_mgr = DataManager(
        base_dir=config.DATA_DIR,
        base_url="https://pds-ppi.igpp.ucla.edu/data/lp-er-calibrated/data-3deleflux-anc/theta/",
    )

    # download specific spice kernels
    logger.info("Downloading specific SPICE kernels...")
    for fname in [
        "lp_ask_980111-980531.bsp",
        "lp_ask_980601-981031.bsp",
        "lp_ask_981101-990331.bsp",
        "lp_ask_990401-990730.bsp",
    ]:

        spice_mgr.download_file(
            f"{spice_mgr.base_url}/{fname}", spice_mgr.base_dir / fname
        )

    # download generic kernels
    logger.info("Downloading generic kernels...")
    for fname in [
        "lsk/latest_leapseconds.tls",
        "pck/pck00011.tpc",
    ]:

        generic_mgr.download_file(
            f"{generic_mgr.base_url}/{fname}", spice_mgr.base_dir / Path(fname).name
        )

    # download lpephemu
    logger.info("Downloading lpephemu kernel...")
    lpephemu_mgr.download_file(
        f"{lpephemu_mgr.base_url}/lpephemu.bsp", spice_mgr.base_dir / "lpephemu.bsp"
    )

    # download attitude table
    logger.info("Downloading attitude table...")
    attitude_mgr.download_file(
        f"{attitude_mgr.base_url}/{config.ATTITUDE_FILE}",
        data_mgr.base_dir / config.ATTITUDE_FILE,
    )

    # download theta files
    logger.info("Downloading theta file and saving solid angles...")
    theta_mgr.download_file(
        f"{theta_mgr.base_url}/{config.THETA_FILE}",
        data_mgr.base_dir / config.THETA_FILE,
    )
    solid_angle_from_thetas(data_mgr.base_dir)

    # download data files
    logger.info("Downloading 3D electron flux data files...")
    years = data_mgr.list_remote_dirs()
    for year in tqdm(years, desc="Years"):
        # list all julian-date subdirectories in that year
        julian_dirs = data_mgr.list_remote_dirs(year)
        for julian in tqdm(julian_dirs, desc=f"Julian days ({year})", leave=False):
            remote_path = f"{year}/{julian}"
            downloaded_files = data_mgr.fetch_directory(remote_path)

    logger.info("All downloads completed.")
