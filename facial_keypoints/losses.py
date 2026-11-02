"""Losses for the facial keypoint study."""

import torch.nn as nn


def plain_mse():
    return nn.MSELoss()


def masked_mse(pred, y, m):
    """sum(m * (pred - y)^2) / clamp(sum(m), min=1).

    NaN entries in y must be pre-sanitized (see build_tensors): mask=0
    suppresses contribution to numerator and denominator, but a single
    NaN multiplied by 0 still produces NaN in floating point.
    """
    diff2 = (pred - y).pow(2) * m
    denom = m.sum().clamp(min=1.0)
    return diff2.sum() / denom
