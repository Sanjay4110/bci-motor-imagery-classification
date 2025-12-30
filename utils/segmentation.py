# utils/segmentation.py
"""
Segmentation utilities for BCI project.

Provides:
- segment_edf_file(file_path, ...): return (segment_array, seg_info, debug)
- batch_segment_files(folder, out_dir, ...): process all EDFs in folder and save segments + metadata
"""

import os
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import mne
from scipy.signal import welch

# Local helper: compute band power for 1D signal using Welch
def _compute_bandpower(signal_1d: np.ndarray, sfreq: float, band: Tuple[float, float]) -> float:
    """Compute band power (sum of PSD in band) for 1D signal using Welch."""
    if len(signal_1d) < 4:
        return 0.0
    nperseg = min(int(sfreq * 2), len(signal_1d))
    freqs, psd = welch(signal_1d, fs=sfreq, nperseg=nperseg)
    idx = np.logical_and(freqs >= band[0], freqs <= band[1])
    return float(np.trapz(psd[idx], freqs[idx]) if np.any(idx) else 0.0)

def _extract_direction_from_filename(filename: str) -> Optional[str]:
    """Try to parse LR or RL from filename (case-insensitive)."""
    name = Path(filename).name.lower()
    if "_lr_" in name or name.endswith("_lr.edf") or name.endswith("_lr"):
        return "LR"
    if "_rl_" in name or name.endswith("_rl.edf") or name.endswith("_rl"):
        return "RL"
    return None

def compute_sliding_erd(raw: mne.io.Raw,
                        window_length: float = 3.0,
                        step_size: float = 0.25,
                        remove_start: float = 0.5) -> Dict[str, Any]:
    """
    Compute sliding ERD (mu & beta) across the recording.

    Returns a dict with:
      - window_times: list of window center times (s)
      - mu_erd_scores: per-window mu ERD (mean across motor channels)
      - beta_erd_scores: per-window beta ERD
      - combined_erd: combined metric (mean of mu & beta ERD)
      - best_window_idx: index of best window
      - best_window_time: (tmin, tmax) of best window
      - motor_channels: list of selected motor channel names
    """
    from utils.eeg_processing import get_motor_channels, compute_psd, calculate_signal_quality

    sfreq = raw.info.get('sfreq', 250.0)
    duration = raw.times[-1] if hasattr(raw, 'times') else raw.n_times / sfreq

    # Choose motor channels (fallback to first 3 channels)
    motor_chs = get_motor_channels(raw)
    if not motor_chs:
        motor_chs = raw.ch_names[:3]

    # Extract baseline from start (0 .. remove_start)
    baseline_tmin = 0.0
    baseline_tmax = min(remove_start, duration)
    if baseline_tmax <= baseline_tmin:
        baseline_tmax = min(0.5, duration)  # fallback

    # extract baseline data
    baseline_data, _ = raw.copy().pick_channels(motor_chs).get_data(return_times=False), sfreq
    # baseline samples indices
    b_start = int(baseline_tmin * sfreq)
    b_end = int(baseline_tmax * sfreq)
    if b_end <= b_start:
        b_end = min(b_start + int(0.5 * sfreq), baseline_data.shape[1])
    baseline_segment = baseline_data[:, b_start:b_end]

    # compute baseline band power per channel
    eps = 1e-10
    mu_band = (8.0, 12.0)
    beta_band = (13.0, 30.0)

    baseline_mu = np.array([_compute_bandpower(ch, sfreq, mu_band) for ch in baseline_segment])
    baseline_beta = np.array([_compute_bandpower(ch, sfreq, beta_band) for ch in baseline_segment])
    # if baseline has zero length, avoid zeros
    baseline_mu_mean = np.mean(baseline_mu) if baseline_mu.size else eps
    baseline_beta_mean = np.mean(baseline_beta) if baseline_beta.size else eps

    # sliding windows
    window_starts = np.arange(remove_start, max(duration - window_length + 1e-6, remove_start), step_size)
    window_times = []
    mu_erd_scores = []
    beta_erd_scores = []
    combined_erd = []

    picked_raw = raw.copy().pick_channels(motor_chs)
    data = picked_raw.get_data()  # shape (n_channels x samples)

    for ws in window_starts:
        we = ws + window_length
        if we > duration:
            break
        s_idx = int(ws * sfreq)
        e_idx = int(we * sfreq)
        if e_idx <= s_idx or e_idx > data.shape[1]:
            continue
        window_seg = data[:, s_idx:e_idx]

        # compute per-channel bandpower in window and mean across selected channels
        win_mu = np.array([_compute_bandpower(ch, sfreq, mu_band) for ch in window_seg])
        win_beta = np.array([_compute_bandpower(ch, sfreq, beta_band) for ch in window_seg])

        # mean across channels
        win_mu_mean = np.mean(win_mu)
        win_beta_mean = np.mean(win_beta)

        # ERD = (Baseline - Trial) / Baseline * 100
        mu_erd = (baseline_mu_mean - win_mu_mean) / (baseline_mu_mean + eps) * 100.0
        beta_erd = (baseline_beta_mean - win_beta_mean) / (baseline_beta_mean + eps) * 100.0

        # store
        center_time = ws + window_length / 2.0
        window_times.append(center_time)
        mu_erd_scores.append(float(mu_erd))
        beta_erd_scores.append(float(beta_erd))
        combined_erd.append(float((mu_erd + beta_erd) / 2.0))

    if len(combined_erd) == 0:
        # fail-safe: single window at end
        window_times = [max(0.0, duration - window_length) + window_length / 2.0]
        mu_erd_scores = [0.0]
        beta_erd_scores = [0.0]
        combined_erd = [0.0]

    best_idx = int(np.argmax(combined_erd))
    best_center = window_times[best_idx]
    best_tmin = best_center - window_length / 2.0
    best_tmax = best_center + window_length / 2.0

    seg_result = {
        'window_times': window_times,
        'mu_erd_scores': mu_erd_scores,
        'beta_erd_scores': beta_erd_scores,
        'combined_erd': combined_erd,
        'best_window_idx': best_idx,
        'best_window_time': (best_tmin, best_tmax),
        'motor_channels': motor_chs,
        'sfreq': sfreq
    }

    return seg_result

def extract_best_segment(raw: mne.io.Raw, seg_result: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Given raw and seg_result from compute_sliding_erd, extract the chosen window
    and create seg_info metadata dict.
    Returns (segment_array (channels x samples), seg_info)
    """
    from utils.eeg_processing import calculate_signal_quality

    sfreq = seg_result.get('sfreq', raw.info.get('sfreq', 250.0))
    tmin, tmax = seg_result['best_window_time']
    # clamp times
    tmin = max(0.0, tmin)
    tmax = min(raw.times[-1], tmax)

    start_idx = int(tmin * sfreq)
    end_idx = int(tmax * sfreq)
    if end_idx <= start_idx:
        # fallback to last possible window
        end_idx = min(start_idx + int(3.0 * sfreq), raw.n_times)
    picked = raw.copy().pick_channels(seg_result.get('motor_channels', raw.ch_names[:3]))
    data = picked.get_data()  # channels x samples
    segment = data[:, start_idx:end_idx]

    # compute ERD scores reported in seg_result
    mu_erd = seg_result['mu_erd_scores'][seg_result['best_window_idx']]
    beta_erd = seg_result['beta_erd_scores'][seg_result['best_window_idx']]
    combined = seg_result['combined_erd'][seg_result['best_window_idx']]

    # signal quality on the extracted segment
    quality = calculate_signal_quality(segment, sfreq)

    seg_info = {
        'tmin': float(tmin),
        'tmax': float(tmax),
        'erd_score': float(combined),
        'mu_erd': float(mu_erd),
        'beta_erd': float(beta_erd),
        'sfreq': float(sfreq),
        'motor_channels': seg_result.get('motor_channels', []),
        'quality_score': float(quality.get('quality_score', 0.0)),
        'direction': _extract_direction_from_filename(getattr(raw, 'filenames', [raw])[0]) if hasattr(raw, 'filenames') else None
    }

    return segment, seg_info

def segment_edf_file(file_path: str,
                     window_length: float = 3.0,
                     step_size: float = 0.25,
                     remove_start: float = 0.5,
                     preprocess: bool = True) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Complete segmentation pipeline for a single EDF file

    Returns:
      - segment: Best window as numpy array (channels x samples)
      - seg_info: dict with metadata (tmin,tmax,erd_score,quality_score,...)
      - seg_result: full segmentation result for plotting/debug
    """
    from utils.eeg_processing import load_edf, preprocess_raw

    raw, metadata = load_edf(file_path, verbose=False)
    if raw is None:
        return None, None, None

    if preprocess:
        try:
            raw = preprocess_raw(raw)
        except Exception:
            # if preprocess fails, proceed with raw
            pass

    # Compute sliding ERD
    seg_result = compute_sliding_erd(raw, window_length=window_length, step_size=step_size, remove_start=remove_start)

    # Extract best segment
    segment, seg_info = extract_best_segment(raw, seg_result)

    # Merge metadata from filename
    if metadata:
        seg_info.update(metadata)

    return segment, seg_info, seg_result

def batch_segment_files(
    input_folder: str,
    output_folder: str = "data/processed",
    window_len: float = 3.0,
    step: float = 0.25,
    overwrite: bool = False,
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """
    Process all .edf files in input_folder, save segments and metadata to output_folder.

    Saves:
      - {base}_segment.npy
      - {base}_metadata.json

    Returns list of metadata dictionaries for processed files.
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    edf_files = sorted(list(input_folder.glob("*.edf")))
    results = []

    for f in edf_files:
        base = f.stem
        seg_path = output_folder / f"{base}_segment.npy"
        meta_path = output_folder / f"{base}_metadata.json"

        if seg_path.exists() and meta_path.exists() and not overwrite:
            if verbose:
                print(f"[batch] Skipping {f.name} (already exists)")
            with open(meta_path, "r") as fd:
                meta = json.load(fd)
            results.append(meta)
            continue

        # NOTE: pass correct arg names to segment_edf_file
        segment, seg_info, debug = segment_edf_file(
            str(f),
            window_length=window_len,
            step_size=step,
            remove_start=0.5,
            preprocess=True
        )

        if segment is None:
            if verbose:
                print(f"[batch] Failed to segment {f.name}")
            continue

        # save the numpy segment and metadata
        np.save(seg_path, segment)
        with open(meta_path, "w") as fd:
            json.dump(seg_info, fd, indent=2)

        if verbose:
            print(f"[batch] Saved: {seg_path} + {meta_path}")

        seg_info["segment_path"] = str(seg_path)
        results.append(seg_info)

    return results
