"""
Visualization Utilities
Create plots for EEG data, segmentation, features, and model evaluation
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_raw_eeg(raw_data: np.ndarray,
                sfreq: float,
                channel_names: List[str] = None,
                duration: float = None) -> plt.Figure:
    """
    Plot raw EEG time series
    
    Args:
        raw_data: EEG data [channels x samples]
        sfreq: Sampling frequency
        channel_names: Channel names
        duration: Duration to plot (None for all)
        
    Returns:
        Matplotlib figure
    """
    n_channels, n_samples = raw_data.shape
    times = np.arange(n_samples) / sfreq
    
    if duration:
        end_idx = int(duration * sfreq)
        raw_data = raw_data[:, :end_idx]
        times = times[:end_idx]
    
    if channel_names is None:
        channel_names = [f'Ch{i+1}' for i in range(n_channels)]
    
    fig, axes = plt.subplots(n_channels, 1, figsize=(14, 2*n_channels), sharex=True)
    
    if n_channels == 1:
        axes = [axes]
    
    for i, (ax, ch_name) in enumerate(zip(axes, channel_names)):
        ax.plot(times, raw_data[i], linewidth=0.5, color='steelblue')
        ax.set_ylabel(ch_name, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        if i == n_channels - 1:
            ax.set_xlabel('Time (s)', fontweight='bold')
    
    plt.suptitle('Raw EEG Signal', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    return fig


def plot_segmentation_results(segmentation_result: Dict,
                              raw_data: np.ndarray = None,
                              sfreq: float = None) -> plt.Figure:
    """
    Plot ERD scores and highlight best segment
    
    Args:
        segmentation_result: Output from compute_sliding_erd
        raw_data: Optional raw EEG data to overlay
        sfreq: Sampling frequency
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    window_times = segmentation_result['window_times']
    mu_erd = segmentation_result['mu_erd_scores']
    beta_erd = segmentation_result['beta_erd_scores']
    combined = segmentation_result['combined_erd']
    best_idx = segmentation_result['best_window_idx']
    best_time = segmentation_result['best_window_time']
    
    # Mu ERD
    axes[0].plot(window_times, mu_erd, 'b-', linewidth=2, label='Mu ERD (8-12 Hz)')
    axes[0].axvline(window_times[best_idx], color='red', linestyle='--', linewidth=2, label='Best Window')
    axes[0].fill_between([best_time[0], best_time[1]], 
                         axes[0].get_ylim()[0], axes[0].get_ylim()[1],
                         alpha=0.2, color='red')
    axes[0].set_ylabel('Mu ERD (%)', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_title('Event-Related Desynchronization Analysis', fontweight='bold', fontsize=12)
    
    # Beta ERD
    axes[1].plot(window_times, beta_erd, 'g-', linewidth=2, label='Beta ERD (13-30 Hz)')
    axes[1].axvline(window_times[best_idx], color='red', linestyle='--', linewidth=2, label='Best Window')
    axes[1].fill_between([best_time[0], best_time[1]], 
                         axes[1].get_ylim()[0], axes[1].get_ylim()[1],
                         alpha=0.2, color='red')
    axes[1].set_ylabel('Beta ERD (%)', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Combined ERD
    axes[2].plot(window_times, combined, 'purple', linewidth=2, label='Combined ERD')
    axes[2].axvline(window_times[best_idx], color='red', linestyle='--', linewidth=2, label='Best Window')
    axes[2].fill_between([best_time[0], best_time[1]], 
                         axes[2].get_ylim()[0], axes[2].get_ylim()[1],
                         alpha=0.2, color='red')
    axes[2].scatter(window_times[best_idx], combined[best_idx], 
                   s=200, color='red', marker='*', zorder=5, label=f'Max ERD: {combined[best_idx]:.1f}%')
    axes[2].set_ylabel('Combined ERD (%)', fontweight='bold')
    axes[2].set_xlabel('Time (s)', fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    
    return fig


def plot_selected_segment(segment: np.ndarray,
                         sfreq: float,
                         channel_names: List[str] = None,
                         segment_info: Dict = None) -> plt.Figure:
    """
    Plot the selected 3-second segment
    
    Args:
        segment: Segmented EEG data [channels x samples]
        sfreq: Sampling frequency
        channel_names: Channel names
        segment_info: Optional metadata
        
    Returns:
        Matplotlib figure
    """
    n_channels, n_samples = segment.shape
    times = np.arange(n_samples) / sfreq
    
    if channel_names is None:
        channel_names = [f'Ch{i+1}' for i in range(n_channels)]
    
    fig, axes = plt.subplots(n_channels, 1, figsize=(12, 2*n_channels), sharex=True)
    
    if n_channels == 1:
        axes = [axes]
    
    for i, (ax, ch_name) in enumerate(zip(axes, channel_names)):
        ax.plot(times, segment[i], linewidth=1, color='darkgreen')
        ax.set_ylabel(ch_name, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        if i == n_channels - 1:
            ax.set_xlabel('Time (s)', fontweight='bold')
    
    title = 'Selected Motor Imagery Window (3 seconds)'
    if segment_info:
        title += f" | ERD: {segment_info.get('erd_score', 0):.1f}% | Quality: {segment_info.get('quality_score', 0):.2f}"
    
    plt.suptitle(title, fontsize=12, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    return fig


def plot_power_spectrum(segment: np.ndarray,
                       sfreq: float,
                       channel_names: List[str] = None) -> plt.Figure:
    """
    Plot power spectrum for each channel
    
    Args:
        segment: EEG data [channels x samples]
        sfreq: Sampling frequency
        channel_names: Channel names
        
    Returns:
        Matplotlib figure
    """
    from scipy import signal as scipy_signal
    
    n_channels = segment.shape[0]
    
    if channel_names is None:
        channel_names = [f'Ch{i+1}' for i in range(n_channels)]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, ch_name in enumerate(channel_names):
        freqs, psd = scipy_signal.welch(segment[i], fs=sfreq, nperseg=int(2*sfreq))
        ax.semilogy(freqs, psd, label=ch_name, linewidth=2)
    
    # Highlight frequency bands
    ax.axvspan(8, 12, alpha=0.2, color='blue', label='Mu (8-12 Hz)')
    ax.axvspan(13, 30, alpha=0.2, color='green', label='Beta (13-30 Hz)')
    ax.axvspan(30, 45, alpha=0.2, color='orange', label='Gamma (30-45 Hz)')
    
    ax.set_xlabel('Frequency (Hz)', fontweight='bold')
    ax.set_ylabel('Power Spectral Density (μV²/Hz)', fontweight='bold')
    ax.set_title('Power Spectrum by Channel', fontweight='bold')
    ax.set_xlim(1, 45)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    return fig


def plot_feature_distributions(df: pd.DataFrame,
                              features: List[str],
                              label_col: str = 'label') -> plt.Figure:
    """
    Plot distributions of selected features by class
    
    Args:
        df: Feature DataFrame
        features: List of features to plot
        label_col: Label column name
        
    Returns:
        Matplotlib figure
    """
    n_features = min(len(features), 9)
    features = features[:n_features]
    
    n_rows = (n_features + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4*n_rows))
    axes = axes.flatten()
    
    for i, feature in enumerate(features):
        lr_data = df[df[label_col] == 1][feature]
        rl_data = df[df[label_col] == 0][feature]
        
        axes[i].hist(rl_data, bins=20, alpha=0.6, label='RL', color='blue', density=True)
        axes[i].hist(lr_data, bins=20, alpha=0.6, label='LR', color='red', density=True)
        axes[i].set_xlabel(feature, fontsize=9)
        axes[i].set_ylabel('Density')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Feature Distributions by Direction', fontweight='bold', fontsize=14)
    plt.tight_layout()
    
    return fig


def plot_confusion_matrix(cm: np.ndarray,
                         class_names: List[str] = ['RL', 'LR']) -> plt.Figure:
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix
        class_names: Class labels
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, ax=ax)
    
    ax.set_xlabel('Predicted Direction', fontweight='bold')
    ax.set_ylabel('Actual Direction', fontweight='bold')
    ax.set_title('Confusion Matrix', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    
    return fig


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float) -> plt.Figure:
    """
    Plot ROC curve
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        roc_auc: Area under curve
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random Classifier')
    
    ax.set_xlabel('False Positive Rate', fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontweight='bold')
    ax.set_title('ROC Curve', fontweight='bold', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig


def plot_model_comparison(results: Dict) -> plt.Figure:
    """
    Compare performance of multiple models
    
    Args:
        results: Dictionary of model results
        
    Returns:
        Matplotlib figure
    """
    metrics = ['cv_mean', 'test_accuracy', 'precision', 'recall', 'f1_score']
    metric_names = ['CV Accuracy', 'Test Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    model_names = [name for name in results.keys() if 'error' not in results[name]]
    
    data = {metric: [] for metric in metrics}
    
    for model_name in model_names:
        for metric in metrics:
            data[metric].append(results[model_name][metric])
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 4))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
    
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        axes[i].bar(range(len(model_names)), data[metric], color=colors)
        axes[i].set_xticks(range(len(model_names)))
        axes[i].set_xticklabels(model_names, rotation=45, ha='right')
        axes[i].set_ylabel('Score')
        axes[i].set_title(metric_name, fontweight='bold')
        axes[i].set_ylim(0, 1.1)
        axes[i].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for j, v in enumerate(data[metric]):
            axes[i].text(j, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)
    
    plt.suptitle('Model Performance Comparison', fontweight='bold', fontsize=14)
    plt.tight_layout()
    
    return fig


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 20) -> plt.Figure:
    """
    Plot top feature importances
    
    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to show
        
    Returns:
        Matplotlib figure
    """
    top_features = importance_df.head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.barh(range(len(top_features)), top_features['importance'].values, color='steelblue')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'].values)
    ax.set_xlabel('Importance', fontweight='bold')
    ax.set_title(f'Top {top_n} Most Important Features', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    return fig


def create_interactive_eeg_plot(raw_data: np.ndarray,
                               sfreq: float,
                               channel_names: List[str] = None) -> go.Figure:
    """
    Create interactive EEG plot using Plotly
    
    Args:
        raw_data: EEG data [channels x samples]
        sfreq: Sampling frequency
        channel_names: Channel names
        
    Returns:
        Plotly figure
    """
    n_channels, n_samples = raw_data.shape
    times = np.arange(n_samples) / sfreq
    
    if channel_names is None:
        channel_names = [f'Ch{i+1}' for i in range(n_channels)]
    
    fig = make_subplots(rows=n_channels, cols=1,
                       subplot_titles=channel_names,
                       shared_xaxes=True,
                       vertical_spacing=0.02)
    
    for i, ch_name in enumerate(channel_names):
        fig.add_trace(
            go.Scatter(x=times, y=raw_data[i], mode='lines', name=ch_name,
                      line=dict(width=1)),
            row=i+1, col=1
        )
    
    fig.update_xaxes(title_text="Time (s)", row=n_channels, col=1)
    fig.update_layout(height=200*n_channels, showlegend=False,
                     title_text="Interactive EEG Signal")
    
    return fig