"""
preprocessing.py

Data-preparation utilities for BrainGPT-mini: normalization now, patch tokenization /
positional encoding helpers to follow as the project grows.
"""

import numpy as np

# Small epsilon added to std before dividing, so a flat/dead channel (std == 0) never gives division by zero error
EPS = np.finfo(np.float32).eps


def zscore_per_channel(data: np.ndarray, eps: float = EPS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score normalize a (channels, samples) array along the time axis, per channel,
    using that recording's own statistics (per-subject normalization).

    Why per-channel: different EEG electrodes have systematically different baseline
    amplitude (frontal vs. occipital, etc.), so a single scalar mean/std across all
    channels would under-correct some channels and over-correct others.

    Why per-subject (stats computed from this array alone, not pooled globally):
    raw EEG amplitude varies subject-to-subject due to electrode impedance and scalp
    contact quality. Normalizing each subject against their own stats removes this
    nuisance variable while preserving the relative shape of that subject's signal.

    Args:
        data: array of shape (num_channels, num_samples)
        eps: numerical stability floor added to std, guards against dead/flat channels

    Returns:
        normalized: same shape as `data`, float32, mean ~0 / std ~1 per channel
        mean: shape (num_channels, 1) -- per-channel mean used (keep for reproducibility)
        std:  shape (num_channels, 1) -- per-channel std used (keep for reproducibility)
    """
    if data.ndim != 2:
        raise ValueError(f"expected (channels, samples), got shape {data.shape}")

    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True)

    normalized = (data - mean) / (std + eps)
    return normalized.astype(np.float32), mean, std


def normalize_epochs_by_subject(filtered_dict: dict, eps: float = EPS) -> dict:
    """
    Per-subject, per-channel z-score normalization for epoched EEG data, as produced
    by `get_filtered_data` (Ingest_V2.py).

    Input format (per subject key, e.g. "s001"):
        {
            "epochs": array (n_epochs, n_channels, n_times),
            "labels": array (n_epochs,),
            "run_id": array (n_epochs,),
            "data_type": "train" or "test",
            "tag": e.g. "lr" or "ff",
        }

    Normalization scope: mean/std are computed per channel, pooled across BOTH the
    epoch axis and the time axis (axis=(0, 2)) -- i.e. one mean/std per channel per
    subject, using every trial and every timepoint that subject has. This matches
    the earlier per-subject scope decision: the dominant amplitude difference in
    this dataset is between subjects (electrode impedance, scalp contact), not
    within a single recording session, so stats are computed per subject rather
    than per run or globally across all subjects.

    NOTE: this operates natively on the 3D (n_epochs, n_channels, n_times) shape
    via axis=(0, 2), rather than reshaping to 2D and reusing `zscore_per_channel`.
    This avoids an implicit transpose+reshape copy and the associated risk of
    getting axis order wrong during the reshape/un-reshape round trip.

    Args:
        filtered_dict: output of `get_filtered_data`, keyed by subject id
        eps: numerical stability floor added to std, guards against dead/flat channels

    Returns:
        normalized_dict: same structure as `filtered_dict`, but with:
            - "epochs" replaced by the per-subject z-scored, float32 array
            - "mean" and "std" added (each shape (1, n_channels, 1)), kept for
              reproducibility and for de-normalizing predictions back to raw
              amplitude units later if needed
            - "labels", "run_id", "data_type", "tag" passed through unchanged

        Does not mutate `filtered_dict` -- a new dict is built and returned.
    """
    normalized_dict = {}

    for sub, items in filtered_dict.items():
        epochs_arr = items["epochs"]
        labels = items["labels"]
        run_id = items["run_id"]
        data_type = items["data_type"]
        tag = items["tag"]

        # Pool over epochs (axis 0) and time (axis 2), keep channels (axis 1) separate.
        # keepdims=True keeps the shapes (1, n_channels, 1) so they broadcast directly
        # against epochs_arr (n_epochs, n_channels, n_times) with no manual reshaping.
        mean = epochs_arr.mean(axis=(0, 2), keepdims=True)
        std = epochs_arr.std(axis=(0, 2), keepdims=True)

        normalized = (epochs_arr - mean) / (std + eps)
        normalized = normalized.astype(np.float32)

        normalized_dict[sub] = {
            "epochs": normalized,
            "mean": mean,
            "std": std,
            "labels": labels,
            "run_id": run_id,
            "data_type": data_type,
            "tag": tag,
        }

    return normalized_dict