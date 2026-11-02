"""Cohorts for the facial keypoint study."""

import numpy as np

from .profile import get_profile_config

COHORT_G1_P = {"frontal": 0.5, "asym": 0.1, "hard": 0.0}

COHORT_G2_THETA = {"frontal": 12.0, "asym": 12.0, "hard": 5.0}

COHORT_RETRY = {"frontal": 3, "asym": 3, "hard": 5}


def _classify_cohort(is_asym, is_hard):
    if is_hard:
        return "hard"
    if is_asym:
        return "asym"
    return "frontal"


def compute_profile_flags(df_split, ds_name):
    cfg = get_profile_config()[ds_name]
    n = len(df_split)
    if ds_name == "data_large":
        eye_mean_x = (
            df_split["left_eye_center_x"].values + df_split["right_eye_center_x"].values
        ) / 2.0
        face_asym = np.abs(eye_mean_x - df_split["nose_tip_x"].values)
        inter_eye_dx = np.abs(
            df_split["left_eye_center_x"].values - df_split["right_eye_center_x"].values
        )
        is_asym = face_asym > cfg["tau_asym_px"]
        is_hard = (face_asym > cfg["tau_hard_px"]) & (inter_eye_dx < cfg["eps_dx_px"])
    elif ds_name == "data_full":
        has_both_eyes = (df_split["mask_left_eye_center_x"].values > 0.5) & (
            df_split["mask_right_eye_center_x"].values > 0.5
        )
        inter_eye_dx = np.abs(
            df_split["left_eye_center_x"].values - df_split["right_eye_center_x"].values
        )
        is_asym = has_both_eyes & (inter_eye_dx < cfg["profile_cutoff_px"])
        is_hard = np.zeros(n, dtype=bool)
    else:  # data_small
        inter_eye_dx = np.abs(
            df_split["left_eye_center_x"].values - df_split["right_eye_center_x"].values
        )
        is_asym = inter_eye_dx < cfg["profile_cutoff_px"]
        is_hard = np.zeros(n, dtype=bool)
    return is_asym.astype(bool), is_hard.astype(bool)
