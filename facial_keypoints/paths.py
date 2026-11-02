"""Project paths, overridable through environment variables.

Importing this module never creates directories or reads dataset files.
"""

import os
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
DATA_DIR = os.path.abspath(
    os.path.expanduser(
        os.environ.get("FACE_KEYPOINTS_DATA_DIR", os.path.join(PROJECT_ROOT, "Data"))
    )
)
CLEAN_DIR = os.path.join(DATA_DIR, "Clean Data")
RAW_CSV = os.path.join(DATA_DIR, "facial-keypoints-detection", "training.csv")
RESULTS_DIR = os.path.abspath(
    os.path.expanduser(
        os.environ.get(
            "FACE_KEYPOINTS_RESULTS_DIR", os.path.join(PROJECT_ROOT, "results")
        )
    )
)
EDA_DIR = os.path.join(RESULTS_DIR, "eda")
PROFILE_PATH = os.path.abspath(
    os.path.expanduser(
        os.environ.get(
            "FACE_KEYPOINTS_PROFILE_PATH",
            os.path.join(EDA_DIR, "profile_thresholds.json"),
        )
    )
)
FIGURE_DIR = os.path.join(PROJECT_ROOT, "figures")
TABLE_DIR = os.path.join(PROJECT_ROOT, "tables")
AGGREGATE_DIR = os.path.join(PROJECT_ROOT, "data")
MASTER_LOG = os.path.join(PROJECT_ROOT, "logs", "master.log")
ANALYSIS_DATA_DIR = os.path.abspath(
    os.path.expanduser(os.environ.get("PSL_DL_DATA_ROOT", CLEAN_DIR))
)
