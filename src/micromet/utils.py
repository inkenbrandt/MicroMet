import logging
from pathlib import Path
from typing import Dict
import yaml

def load_yaml(path: Path | str) -> Dict:
    """
    Load a YAML file and return its contents as a dictionary.

    Parameters
    ----------
    path : Path or str
        The path to the YAML file.

    Returns
    -------
    dict
        The contents of the YAML file as a dictionary.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open() as fp:
        return yaml.safe_load(fp)


def logger_check(logger: logging.Logger | None) -> logging.Logger:
    """
    Initialize and return a logger instance if none is provided.

    This function checks if a logger object is provided. If not, it
    creates and configures a new logger.

    Parameters
    ----------
    logger : logging.Logger or None
        An existing logger instance.

    Returns
    -------
    logging.Logger
        A configured logger instance.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.WARNING)
        ch = logging.StreamHandler()
        ch.setFormatter(
            logging.Formatter(
                fmt="%(levelname)s [%(asctime)s] %(name)s – %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(ch)
    else:
        logger = logger
    return logger
