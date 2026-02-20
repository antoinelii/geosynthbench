# src/geosynthbench/pipeline/metrics.py
from __future__ import annotations

import numpy as np


def accuracy(pred: str, gt: str) -> float:
    return float(pred == gt)


def binary_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    inter = np.logical_and(pred_b, gt_b).sum()
    union = np.logical_or(pred_b, gt_b).sum()
    return float(inter) / float(max(union, 1))
