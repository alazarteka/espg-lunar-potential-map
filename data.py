from tqdm import tqdm
import os
import logging
import requests
from pathlib import Path
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class FluxDataManager:
    """
    Manage downloading and organizing PDS flux data files by scraping directory indexes.

    Features:
    - List remote directories and files
    - Download all .TAB (or specified extension) files in a given directory
    - Skip existing downloads
    """
    def __init__(self, base_dir: str, base_url: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = base_url.rstrip('/')

    def ensure_dir(self, *subdirs) -> Path:
        path = self.base_dir.joinpath(*subdirs)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def list_remote_dirs(self, remote_path: str = '') -> list[str]:
        """
        Return a list of subdirectory names under base_url/remote_path/.
        """
        url = f"{self.base_url}/{remote_path.rstrip('/')}/"
        logger.info(f"Listing remote directories at {url}")
        resp = requests.get(url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        dirs = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            # directory links end with '/'
            if href.endswith('/') and href not in ('../', './'):
                dirs.append(href.rstrip('/'))

        if len(dirs):
            dirs = dirs[1:]
        return dirs

    def list_remote_files(self, remote_path: str, ext: str = '.TAB') -> list[str]:
        """
        Return filenames under base_url/remote_path/ matching ext.
        """
        url = f"{self.base_url}/{remote_path.rstrip('/')}/"
        logger.info(f"Listing remote files at {url}")
        resp = requests.get(url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        files = [a['href'] for a in soup.find_all('a', href=True)
                 if a['href'].upper().endswith(ext.upper())]
        return files

    def download_file(self, url: str, dest: Path, chunk_size: int = 4 * 1024 * 1024) -> Path:
        """
        Stream-download a file if not already present.
        """
        dest = Path(dest)
        if dest.exists():
            # logger.info(f"Skipping existing file: {dest}")
            tqdm.write(f"Skipping existing file: {dest}")
            return dest
        # logger.info(f"Downloading {url} -> {dest}")
        tqdm.write(f"Downloading {url} -> {dest}")
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        with open(dest, 'wb') as f:
            for chunk in resp.iter_content(chunk_size):
                if chunk:
                    f.write(chunk)
        # logger.info(f"Downloaded {dest}")
        tqdm.write(f"Downloaded {dest}")
        return dest

    def fetch_directory(self, remote_path: str, ext: str = '.TAB') -> list[Path]:
        """
        Download all files with the given extension in base_url/remote_path/.
        """
        files = self.list_remote_files(remote_path, ext)
        local_dir = self.ensure_dir(*remote_path.split('/'))
        paths = []
        for fname in files:
            url = f"{self.base_url}/{remote_path}/{fname}"
            dest = local_dir / fname
            paths.append(self.download_file(url, dest))
        return paths

if __name__ == '__main__':
    base_url = 'https://pds-ppi.igpp.ucla.edu/data/lp-er-calibrated/data-3deleflux/'

    mgr = FluxDataManager(base_dir='data', base_url=base_url)

    years = mgr.list_remote_dirs()
    for year in tqdm(years, desc="Years"):
        # list all julian-date subdirectories in that year
        julian_dirs = mgr.list_remote_dirs(year)
        for julian in tqdm(julian_dirs, desc=f"Julian days ({year})", leave=False):
            remote_path = f"{year}/{julian}"
            downloaded_files = mgr.fetch_directory(remote_path)
            for f in downloaded_files:
                # print(f"Downloaded: {f}")
                tqdm.write(f"Downloaded: {f}")
