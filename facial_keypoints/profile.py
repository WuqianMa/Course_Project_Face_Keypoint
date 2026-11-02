"""Load the experiment's calibrated cohort thresholds only when needed."""

import json
from functools import lru_cache

from .paths import PROFILE_PATH


@lru_cache(maxsize=1)
def get_profile_config():
    """Read the original EDA thresholds; derive the full-data cutoff if absent."""
    try:
        with open(PROFILE_PATH, encoding="utf-8") as stream:
            config = json.load(stream)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Cohort thresholds are required at {PROFILE_PATH}. "
            "Restore profile_thresholds.json from the experiment or set "
            "FACE_KEYPOINTS_PROFILE_PATH. See README.md for its schema."
        ) from error
    if "data_full" not in config:
        config["data_full"] = {
            "profile_cutoff_px": config["data_small"]["profile_cutoff_px"],
        }
    return config
