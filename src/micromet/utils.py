import logging
from pathlib import Path
from typing import Dict
import yaml

def load_yaml(path: Path | str) -> Dict:
    """Load a YAML file and return as dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open() as fp:
        return yaml.safe_load(fp)


def logger_check(logger: logging.Logger | None) -> logging.Logger:
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
