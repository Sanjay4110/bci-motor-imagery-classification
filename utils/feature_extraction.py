# utils/feature_extraction.py
"""
Feature Extraction Module
Extracts frequency, time, and spatial features from EEG segments
"""

import numpy as np
from scipy import signal
from scipy.stats import skew, kurtosis
from typing import Dict, Tuple, List
import pandas as pd

def compute_band_power(data: np.ndarray, 
                       sfreq: float, 
                       band: Tuple[float, float]) -> np.ndarray:
    """Compute power in specific frequency band for each channel"""
    freqs, psd = signal.welch(data, fs=sfreq, nperseg=int(2*sfreq))
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    return np.trapz(psd[:, band_mask], freqs[band_mask], axis=1)

def extract_frequency_features(segment: np.ndarray, sfreq: float) -> Dict[str, float]:
    """
    Extract frequency domain features
    """
    features = {}
    # Define frequency bands
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'mu': (8, 12),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }

    # Compute power for each band and channel
    for band_name, (fmin, fmax) in bands.items():
        band_power = compute_band_power(segment, sfreq, (fmin, fmax))
        for ch_idx, power in enumerate(band_power):
            features[f'{band_name}_power_ch{ch_idx}'] = power
        features[f'{band_name}_power_mean'] = np.mean(band_power)
        features[f'{band_name}_power_std'] = np.std(band_power)

    # total power
    total_power = compute_band_power(segment, sfreq, (1, 45))
    for ch_idx, power in enumerate(total_power):
        features[f'total_power_ch{ch_idx}'] = power
    features['total_power_mean'] = np.mean(total_power)

    # ratios
    mu_power = compute_band_power(segment, sfreq, bands['mu'])
    beta_power = compute_band_power(segment, sfreq, bands['beta'])
    gamma_power = compute_band_power(segment, sfreq, bands['gamma'])
    alpha_power = compute_band_power(segment, sfreq, bands['alpha'])

    with np.errstate(divide='ignore', invalid='ignore'):
        features['mu_beta_ratio'] = np.mean(mu_power / (beta_power + 1e-10))
        features['beta_gamma_ratio'] = np.mean(beta_power / (gamma_power + 1e-10))
        features['mu_alpha_ratio'] = np.mean(mu_power / (alpha_power + 1e-10))
        features['alpha_beta_ratio'] = np.mean(alpha_power / (beta_power + 1e-10))

    return features

def extract_time_features(segment: np.ndarray) -> Dict[str, float]:
    """
    Extract time domain features
    """
    features = {}
    for ch_idx, ch_data in enumerate(segment):
        features[f'mean_ch{ch_idx}'] = np.mean(ch_data)
        features[f'std_ch{ch_idx}'] = np.std(ch_data)
        features[f'var_ch{ch_idx}'] = np.var(ch_data)
        features[f'skewness_ch{ch_idx}'] = skew(ch_data)
        features[f'kurtosis_ch{ch_idx}'] = kurtosis(ch_data)

        zero_crossings = np.where(np.diff(np.sign(ch_data)))[0]
        features[f'zcr_ch{ch_idx}'] = len(zero_crossings) / len(ch_data)

        features[f'energy_ch{ch_idx}'] = np.sum(ch_data ** 2)

    features['mean_all'] = np.mean(segment)
    features['std_all'] = np.std(segment)
    features['max_amplitude'] = np.max(np.abs(segment))
    features['min_amplitude'] = np.min(segment)

    hjorth = compute_hjorth_parameters(segment)
    for ch_idx, (activity, mobility, complexity) in enumerate(hjorth):
        features[f'hjorth_activity_ch{ch_idx}'] = activity
        features[f'hjorth_mobility_ch{ch_idx}'] = mobility
        features[f'hjorth_complexity_ch{ch_idx}'] = complexity

    features['hjorth_activity_mean'] = np.mean([h[0] for h in hjorth])
    features['hjorth_mobility_mean'] = np.mean([h[1] for h in hjorth])
    features['hjorth_complexity_mean'] = np.mean([h[2] for h in hjorth])

    return features

def compute_hjorth_parameters(segment: np.ndarray) -> list:
    results = []
    for ch_data in segment:
        d1 = np.diff(ch_data)
        d2 = np.diff(d1)
        activity = np.var(ch_data)
        mobility = np.sqrt(np.var(d1) / (activity + 1e-10))
        complexity = np.sqrt(np.var(d2) / (np.var(d1) + 1e-10)) / (mobility + 1e-10)
        results.append((activity, mobility, complexity))
    return results

def extract_spatial_features(segment: np.ndarray,
                             channel_names: list = None) -> Dict[str, float]:
    features = {}
    n_channels = segment.shape[0]

    c3_idx, c4_idx = None, None
    if channel_names:
        for i, name in enumerate(channel_names):
            if name.upper() == 'C3':
                c3_idx = i
            elif name.upper() == 'C4':
                c4_idx = i

    if c3_idx is not None and c4_idx is not None:
        c3_power = np.var(segment[c3_idx])
        c4_power = np.var(segment[c4_idx])

        features['c3_power'] = c3_power
        features['c4_power'] = c4_power
        features['c3_c4_diff'] = c3_power - c4_power

        with np.errstate(divide='ignore', invalid='ignore'):
            features['c3_c4_asymmetry'] = (c3_power - c4_power) / (c3_power + c4_power + 1e-10)

        features['c3_c4_correlation'] = np.corrcoef(segment[c3_idx], segment[c4_idx])[0, 1]

    if n_channels > 1:
        corr_matrix = np.corrcoef(segment)
        upper_tri = corr_matrix[np.triu_indices(n_channels, k=1)]
        features['mean_correlation'] = np.mean(upper_tri)
        features['std_correlation'] = np.std(upper_tri)
        features['max_correlation'] = np.max(upper_tri)
        features['min_correlation'] = np.min(upper_tri)

    channel_powers = np.var(segment, axis=1)
    features['power_range'] = np.max(channel_powers) - np.min(channel_powers)
    features['power_std'] = np.std(channel_powers)
    features['dominant_channel'] = int(np.argmax(channel_powers))

    return features

def extract_all_features(segment: np.ndarray,
                         sfreq: float,
                         channel_names: list = None) -> Dict[str, float]:
    features = {}
    features.update(extract_frequency_features(segment, sfreq))
    features.update(extract_time_features(segment))
    features.update(extract_spatial_features(segment, channel_names))
    return features

def build_feature_dataset(segment_info_list: list,
                          segments_dir: str,
                          use_filename_labels: bool = True) -> pd.DataFrame:
    """
    Build feature dataset from segmented files.

    Args:
        segment_info_list: List of segment info dictionaries (must include filename, sfreq, label optionally)
        segments_dir: Directory containing saved segments (.npy)
        use_filename_labels: If True, include only samples that have label 0/1 in metadata.
                             If False, include segments even if label missing (label will be -1).
    Returns:
        DataFrame with features and labels
    """
    import os
    from pathlib import Path

    dataset = []
    segments_dir = Path(segments_dir)

    for info in segment_info_list:
        # if use_filename_labels is True, require valid label 0 or 1
        if use_filename_labels:
            if 'label' not in info or info.get('label') not in (0, 1):
                # skip unlabeled samples
                print(f"Skipping unlabeled file: {info.get('filename')}")
                continue

        # determine segment path
        if 'segment_path' in info and info['segment_path']:
            segment_path = info['segment_path']
        else:
            base_name = Path(info['filename']).stem
            segment_path = str(segments_dir / f"{base_name}_segment.npy")

        if not os.path.exists(segment_path):
            print(f"Warning: Segment not found: {segment_path}")
            continue

        segment = np.load(segment_path)

        # extract features
        features = extract_all_features(
            segment,
            sfreq=info.get('sfreq', 250.0),
            channel_names=info.get('motor_channels', None)
        )

        # add metadata
        features['filename'] = info.get('filename', Path(segment_path).name)
        features['subject'] = info.get('subject', 'unknown')
        features['method'] = info.get('method', 'unknown')
        features['direction'] = info.get('direction', 'unknown')
        features['pattern'] = info.get('pattern', 'unknown')
        features['iteration'] = info.get('iteration', '0')
        # label: prefer explicit label if present, otherwise -1
        features['label'] = info.get('label', -1)
        features['erd_score'] = info.get('erd_score', 0)
        features['quality_score'] = info.get('quality_score', 0)

        dataset.append(features)

    df = pd.DataFrame(dataset)
    print(f"Feature dataset created: {len(df)} samples, {len(df.columns)} features")
    return df

def select_important_features(df: pd.DataFrame,
                              label_col: str = 'label',
                              method: str = 'variance',
                              n_features: int = 50) -> list:
    from sklearn.feature_selection import VarianceThreshold, mutual_info_classif

    metadata_cols = ['filename', 'subject', 'method', 'direction', 'pattern', 
                     'iteration', 'label', 'erd_score', 'quality_score']
    feature_cols = [col for col in df.columns if col not in metadata_cols]

    X = df[feature_cols].values
    y = df[label_col].values

    if method == 'variance':
        selector = VarianceThreshold(threshold=0.01)
        selector.fit(X)
        selected = [feature_cols[i] for i in range(len(feature_cols)) 
                   if selector.get_support()[i]]
    elif method == 'mutual_info':
        mi_scores = mutual_info_classif(X, y, random_state=42)
        top_indices = np.argsort(mi_scores)[-n_features:]
        selected = [feature_cols[i] for i in top_indices]
    else:
        selected = feature_cols

    return selected[:n_features]
