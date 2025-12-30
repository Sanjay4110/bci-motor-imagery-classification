"""
EEG Signal Processing Utilities
Handles loading, preprocessing, and basic analysis of EEG data
"""

import mne
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal
from typing import Tuple, Dict, Optional
import re


def parse_filename(filename: str) -> Dict[str, str]:
    """
    Parse EDF filename to extract metadata
    
    Format: <subject>_<method>_<direction>_<pattern>_<iteration>.edf
    Example: pradeep_move_LR_lower_1.edf
    
    Args:
        filename: Name of the EDF file
        
    Returns:
        Dictionary with metadata fields
    """
    # Remove .edf extension
    name = Path(filename).stem
    
    # Split by underscore
    parts = name.split('_')
    
    if len(parts) >= 5:
        return {
            'filename': filename,
            'subject': parts[0],
            'method': parts[1],  # move or watch
            'direction': parts[2],  # LR or RL
            'pattern': parts[3],  # straight, upper, lower
            'iteration': parts[4],
            'label': 1 if parts[2] == 'LR' else 0  # LR=1, RL=0
        }
    else:
        # Fallback for non-standard names
        return {
            'filename': filename,
            'subject': 'unknown',
            'method': 'unknown',
            'direction': 'unknown',
            'pattern': 'unknown',
            'iteration': '0',
            'label': -1
        }


def load_edf(file_path: str, verbose: bool = False) -> Tuple[mne.io.Raw, Dict]:
    """
    Load EDF file and return raw data with metadata
    
    Args:
        file_path: Path to EDF file
        verbose: Print loading information
        
    Returns:
        Tuple of (raw MNE object, metadata dict)
    """
    try:
        # Load EDF file
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        
        # Parse filename
        metadata = parse_filename(Path(file_path).name)
        metadata['file_path'] = file_path
        metadata['n_channels'] = len(raw.ch_names)
        metadata['sfreq'] = raw.info['sfreq']
        metadata['duration'] = raw.times[-1]
        
        if verbose:
            print(f"Loaded: {Path(file_path).name}")
            print(f"  Channels: {raw.ch_names}")
            print(f"  Sampling rate: {raw.info['sfreq']} Hz")
            print(f"  Duration: {raw.times[-1]:.2f} seconds")
            print(f"  Direction: {metadata['direction']}")
        
        return raw, metadata
        
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None


def preprocess_raw(raw: mne.io.Raw, 
                   l_freq: float = 1.0,
                   h_freq: float = 45.0,
                   notch_freq: float = 50.0) -> mne.io.Raw:
    """
    Preprocess raw EEG data
    
    Args:
        raw: MNE Raw object
        l_freq: High-pass filter frequency (Hz)
        h_freq: Low-pass filter frequency (Hz)
        notch_freq: Notch filter frequency for line noise (Hz)
        
    Returns:
        Preprocessed Raw object
    """
    # Make a copy to avoid modifying original
    raw_processed = raw.copy()
    
    # Apply bandpass filter (1-45 Hz)
    raw_processed.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
    
    # Apply notch filter for line noise (50 Hz or 60 Hz)
    raw_processed.notch_filter(freqs=notch_freq, verbose=False)
    
    # Re-reference to average (optional but recommended)
    raw_processed.set_eeg_reference('average', projection=False, verbose=False)
    
    return raw_processed


def extract_channel_data(raw: mne.io.Raw, 
                         channel_names: list = None) -> Tuple[np.ndarray, float]:
    """
    Extract data from specific channels
    
    Args:
        raw: MNE Raw object
        channel_names: List of channel names (e.g., ['C3', 'C4'])
        
    Returns:
        Tuple of (data array [channels x samples], sampling frequency)
    """
    if channel_names is None:
        # Default to motor cortex channels
        channel_names = ['C3', 'Cz', 'C4']
    
    # Get available channels (case-insensitive)
    available_channels = []
    for ch in channel_names:
        matches = [name for name in raw.ch_names if name.upper() == ch.upper()]
        if matches:
            available_channels.append(matches[0])
    
    if not available_channels:
        # Fallback: use all channels
        available_channels = raw.ch_names
    
    # Extract data
    data, times = raw[available_channels, :]
    
    return data, raw.info['sfreq']


def compute_psd(data: np.ndarray, 
                sfreq: float,
                fmin: float = 1.0,
                fmax: float = 45.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Power Spectral Density using Welch's method
    
    Args:
        data: EEG data [channels x samples]
        sfreq: Sampling frequency
        fmin: Minimum frequency
        fmax: Maximum frequency
        
    Returns:
        Tuple of (frequencies, psd [channels x frequencies])
    """
    n_channels = data.shape[0]
    
    # Compute PSD for each channel
    psd_list = []
    for ch_data in data:
        freqs, psd = signal.welch(ch_data, fs=sfreq, nperseg=int(2*sfreq))
        psd_list.append(psd)
    
    psd = np.array(psd_list)
    
    # Restrict to frequency range
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    
    return freqs[freq_mask], psd[:, freq_mask]


def compute_band_power(data: np.ndarray,
                       sfreq: float,
                       band: Tuple[float, float]) -> np.ndarray:
    """
    Compute power in a specific frequency band
    
    Args:
        data: EEG data [channels x samples]
        sfreq: Sampling frequency
        band: Tuple of (low_freq, high_freq)
        
    Returns:
        Band power for each channel
    """
    freqs, psd = compute_psd(data, sfreq, fmin=band[0]-1, fmax=band[1]+1)
    
    # Find indices for band
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    
    # Integrate power in band
    band_power = np.trapz(psd[:, band_mask], freqs[band_mask], axis=1)
    
    return band_power


def compute_erd(baseline_power: np.ndarray,
                trial_power: np.ndarray) -> np.ndarray:
    """
    Compute Event-Related Desynchronization (ERD)
    
    ERD = (Baseline - Trial) / Baseline * 100
    Positive ERD indicates power decrease (desynchronization)
    
    Args:
        baseline_power: Power during baseline period
        trial_power: Power during trial period
        
    Returns:
        ERD percentage for each channel
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        erd = (baseline_power - trial_power) / baseline_power * 100
        erd = np.nan_to_num(erd, nan=0.0, posinf=0.0, neginf=0.0)
    
    return erd


def extract_time_window(raw: mne.io.Raw,
                        tmin: float,
                        tmax: float) -> np.ndarray:
    """
    Extract data from a specific time window
    
    Args:
        raw: MNE Raw object
        tmin: Start time (seconds)
        tmax: End time (seconds)
        
    Returns:
        Data array [channels x samples] for the time window
    """
    data, times = raw[:, :]
    sfreq = raw.info['sfreq']
    
    # Convert times to sample indices
    start_idx = int(tmin * sfreq)
    end_idx = int(tmax * sfreq)
    
    return data[:, start_idx:end_idx]


def get_motor_channels(raw: mne.io.Raw) -> list:
    """
    Identify motor cortex channels (C3, Cz, C4) from channel names
    
    Args:
        raw: MNE Raw object
        
    Returns:
        List of available motor channel names
    """
    motor_channels = ['C3', 'CZ', 'C4', 'FC3', 'FCZ', 'FC4', 'CP3', 'CPZ', 'CP4']
    
    available = []
    for ch in motor_channels:
        matches = [name for name in raw.ch_names 
                  if name.upper() == ch.upper()]
        if matches:
            available.append(matches[0])
    
    return available if available else raw.ch_names[:3]  # Fallback


def calculate_signal_quality(data: np.ndarray, sfreq: float) -> Dict[str, float]:
    """
    Calculate signal quality metrics
    
    Args:
        data: EEG data [channels x samples]
        sfreq: Sampling frequency
        
    Returns:
        Dictionary of quality metrics
    """
    metrics = {}
    
    # Signal-to-Noise Ratio (simple estimate)
    signal_power = np.mean(data ** 2)
    noise_estimate = np.median(np.abs(np.diff(data, axis=1))) / 0.6745
    snr = 10 * np.log10(signal_power / (noise_estimate ** 2 + 1e-10))
    metrics['snr_db'] = snr
    
    # Check for flatlines
    flatline_ratio = np.mean(np.std(data, axis=1) < 0.1)
    metrics['flatline_ratio'] = flatline_ratio
    
    # Check for extreme values
    extreme_ratio = np.mean(np.abs(data) > 200)  # > 200 µV
    metrics['extreme_ratio'] = extreme_ratio
    
    # Overall quality score (0-1)
    quality_score = (1 - flatline_ratio) * (1 - extreme_ratio) * np.clip(snr / 20, 0, 1)
    metrics['quality_score'] = quality_score
    
    return metrics