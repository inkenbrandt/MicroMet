"""
Utility functions for the micromet package.
"""
import logging
import configparser
from pathlib import Path
from typing import Dict, Tuple
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


# ============================================================================
# Site Configuration Functions
# ============================================================================


def read_site_config(
    site_id: str,
    config_dir: Path | str = "src/micromet/data"
) -> Dict[str, float | str]:
    """
    Read site configuration from an .ini file.

    Parameters
    ----------
    site_id : str
        The site identifier (e.g., 'US-CdM', 'US-UTW').
    config_dir : Path or str, optional
        Directory containing the .ini files. Defaults to 'src/micromet/data'.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'site_lat': float - Station latitude
        - 'site_lon': float - Station longitude
        - 'site_utc_offset': float - UTC offset in hours
        - 'site_elevation': float - Station elevation in meters
        - 'site_name': str - Full station name
        - 'site_id': str - Station identifier

    Raises
    ------
    FileNotFoundError
        If the .ini file for the site is not found.
    KeyError
        If required metadata fields are missing.

    Examples
    --------
    >>> config = read_site_config('US-CdM')
    >>> config['site_lat']
    37.5241
    >>> config['site_utc_offset']
    -7.0
    """
    config_dir = Path(config_dir)
    ini_file = config_dir / f"{site_id}.ini"
    
    if not ini_file.exists():
        # Try to provide helpful error message
        available_files = list(config_dir.glob("US-*.ini"))
        available_sites = [f.stem for f in available_files]
        raise FileNotFoundError(
            f"Configuration file not found: {ini_file}\n"
            f"Available sites: {', '.join(available_sites)}"
        )
    
    parser = configparser.ConfigParser()
    parser.read(ini_file)
    
    try:
        metadata = parser['METADATA']
        return {
            'site_lat': float(metadata['station_latitude']),
            'site_lon': float(metadata['station_longitude']),
            'site_utc_offset': float(metadata['utc_offset']),
            'site_elevation': float(metadata['station_elevation']),
            'site_name': metadata['site_name'],
            'site_id': site_id,
        }
    except KeyError as e:
        raise KeyError(
            f"Missing required metadata field in {ini_file}: {e}"
        ) from e
    except ValueError as e:
        raise ValueError(
            f"Invalid numeric value in {ini_file}: {e}"
        ) from e


def get_all_site_configs(
    config_dir: Path | str = "src/micromet/data"
) -> Dict[str, Dict[str, float | str]]:
    """
    Read all site configurations from .ini files in a directory.

    Parameters
    ----------
    config_dir : Path or str, optional
        Directory containing the .ini files. Defaults to 'src/micromet/data'.

    Returns
    -------
    dict
        Dictionary mapping site_id to configuration dictionaries.

    Examples
    --------
    >>> all_configs = get_all_site_configs()
    >>> all_configs['US-CdM']['site_lat']
    37.5241
    >>> list(all_configs.keys())
    ['US-CdM', 'US-UTB', 'US-UTD', ...]
    """
    config_dir = Path(config_dir)
    all_configs = {}
    
    for ini_file in sorted(config_dir.glob("US-*.ini")):
        site_id = ini_file.stem
        try:
            all_configs[site_id] = read_site_config(site_id, config_dir)
        except (KeyError, ValueError) as e:
            logging.warning(f"Could not read config for {site_id}: {e}")
    
    return all_configs


def extract_config_for_reformatter(
    site_id: str,
    config_dir: Path | str = "src/micromet/data"
) -> Tuple[float, float, float]:
    """
    Extract only the values needed for Reformatter from a site config.

    This is a convenience function that returns just the three values
    needed to initialize a Reformatter with timestamp checking.

    Parameters
    ----------
    site_id : str
        The site identifier (e.g., 'US-CdM').
    config_dir : Path or str, optional
        Directory containing the .ini files.

    Returns
    -------
    tuple[float, float, float]
        A tuple of (site_lat, site_lon, site_utc_offset).

    Examples
    --------
    >>> lat, lon, utc = extract_config_for_reformatter('US-CdM')
    >>> lat, lon, utc
    (37.5241, -109.7471, -7.0)
    """
    config = read_site_config(site_id, config_dir)
    return (
        config['site_lat'],
        config['site_lon'],
        config['site_utc_offset']
    ) # type: ignore


def create_reformatter_from_site(
    site_id: str,
    config_dir: Path | str = "src/micromet/data",
    check_timestamps: bool = True,
    **reformatter_kwargs
):
    """
    Create a Reformatter instance with site configuration loaded from .ini file.

    This is a convenience factory function that reads the site configuration
    and creates a properly configured Reformatter instance.

    Parameters
    ----------
    site_id : str
        The site identifier (e.g., 'US-CdM', 'US-UTW').
    config_dir : Path or str, optional
        Directory containing the .ini files. Defaults to 'src/micromet/data'.
    check_timestamps : bool, optional
        Whether to enable timestamp checking. Defaults to True.
    **reformatter_kwargs
        Additional keyword arguments passed to Reformatter
        (e.g., drop_soil, var_limits_csv).

    Returns
    -------
    Reformatter
        A configured Reformatter instance.

    Examples
    --------
    >>> reformatter = create_reformatter_from_site('US-CdM')
    >>> df_clean, report, ts_results = reformatter.process(raw_data)
    
    >>> # With additional options
    >>> reformatter = create_reformatter_from_site(
    ...     'US-UTW',
    ...     drop_soil=False,
    ...     check_timestamps=True
    ... )
    
    >>> # Disable timestamp checking for speed
    >>> reformatter = create_reformatter_from_site(
    ...     'US-UTB',
    ...     check_timestamps=False
    ... )
    """
    from micromet.format.reformatter import Reformatter
    
    lat, lon, utc = extract_config_for_reformatter(site_id, config_dir)
    
    return Reformatter(
        site_lat=lat,
        site_lon=lon,
        site_utc_offset=utc,
        check_timestamps=check_timestamps,
        **reformatter_kwargs
    )


__all__ = [
    'load_yaml',
    'logger_check',
    'read_site_config',
    'get_all_site_configs',
    'extract_config_for_reformatter',
    'create_reformatter_from_site',
]