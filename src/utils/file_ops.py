import logging
import os


def list_files(directory: str) -> list[str]:
    """
    List all files in the specified directory.

    Args:
        directory: Path to the directory to list.

    Returns:
        list[str]: List of filenames in the directory.
    """
    try:
        files = os.listdir(directory)
        return [f for f in files if os.path.isfile(os.path.join(directory, f))]
    except FileNotFoundError:
        logging.error(f"Directory {directory} not found.")
        return []


def list_folder_files(directory: str) -> list[str]:
    """
    List all subdirectories in the specified directory.

    Args:
        directory: Path to the directory to list.

    Returns:
        list[str]: List of subdirectory names in the directory.
    """
    try:
        files = os.listdir(directory)
        return [f for f in files if os.path.isdir(os.path.join(directory, f))]
    except FileNotFoundError:
        logging.error(f"Directory {directory} not found.")
        return []
