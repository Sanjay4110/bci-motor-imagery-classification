"""
Machine Learning Models Module
Train and evaluate classifiers for LR/RL direction detection
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc
import joblib
import json
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def prepare_data(df: pd.DataFrame,
                label_col: str = 'label',
                test_size: float = 0.2,
                random_state: int = 42) -> Tuple:
    """
    Prepare data for training
    
    Args:
        df: Feature DataFrame
        label_col: Name of label column
        test_size: Proportion for test set
        random_state: Random seed
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names, scaler)
    """
    # Separate features and labels
    metadata_cols = ['filename', 'subject', 'method', 'direction', 'pattern', 
                    'iteration', 'label', 'erd_score', 'quality_score']
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    
    X = df[feature_cols].values
    y = df[label_col].values
    
    # Handle any NaN or inf values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, feature_cols, scaler


def train_single_model(model_name: str,
                      X_train: np.ndarray,
                      y_train: np.ndarray,
                      X_test: np.ndarray,
                      y_test: np.ndarray,
                      cv_folds: int = 5) -> Dict:
    """
    Train and evaluate a single model
    
    Args:
        model_name: Name of model ('LDA', 'SVM_Linear', 'SVM_RBF', 'RF', 'XGBoost', 'LogReg')
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        cv_folds: Number of cross-validation folds
        
    Returns:
        Dictionary with model and metrics
    """
    # Initialize model
    if model_name == 'LDA':
        model = LinearDiscriminantAnalysis()
    elif model_name == 'SVM_Linear':
        model = SVC(kernel='linear', probability=True, random_state=42)
    elif model_name == 'SVM_RBF':
        model = SVC(kernel='rbf', probability=True, random_state=42)
    elif model_name == 'RF':
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    elif model_name == 'XGBoost':
        if XGBOOST_AVAILABLE:
            model = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
        else:
            return {'error': 'XGBoost not available'}
    elif model_name == 'LogReg':
        model = LogisticRegression(max_iter=1000, random_state=42)
    else:
        return {'error': f'Unknown model: {model_name}'}
    
    # Cross-validation on training set
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    
    # Train on full training set
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Probabilities
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_test_pred, average='binary'
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    
    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
    roc_auc = auc(fpr, tpr)
    
    return {
        'model': model,
        'model_name': model_name,
        'cv_scores': cv_scores.tolist(),
        'cv_mean': float(cv_scores.mean()),
        'cv_std': float(cv_scores.std()),
        'train_accuracy': float(train_acc),
        'test_accuracy': float(test_acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'roc_auc': float(roc_auc),
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist()
    }


def train_all_models(X_train: np.ndarray,
                    y_train: np.ndarray,
                    X_test: np.ndarray,
                    y_test: np.ndarray,
                    cv_folds: int = 5) -> Dict:
    """
    Train all available models and compare
    
    Returns:
        Dictionary with results for all models
    """
    models_to_train = ['LDA', 'SVM_Linear', 'SVM_RBF', 'RF', 'LogReg']
    
    if XGBOOST_AVAILABLE:
        models_to_train.append('XGBoost')
    
    results = {}
    
    print("Training models...")
    for model_name in models_to_train:
        print(f"  Training {model_name}...")
        result = train_single_model(
            model_name, X_train, y_train, X_test, y_test, cv_folds
        )
        results[model_name] = result
    
    return results


def select_best_model(results: Dict) -> Tuple[str, Dict]:
    """
    Select best model based on test accuracy
    
    Args:
        results: Dictionary of model results
        
    Returns:
        Tuple of (best_model_name, best_model_result)
    """
    best_name = None
    best_acc = 0
    
    for name, result in results.items():
        if 'error' not in result:
            if result['test_accuracy'] > best_acc:
                best_acc = result['test_accuracy']
                best_name = name
    
    return best_name, results[best_name]


def save_model(model, scaler, feature_names: List[str],
              model_path: str, scaler_path: str, 
              metadata_path: str):
    """
    Save trained model and preprocessing objects
    
    Args:
        model: Trained model
        scaler: Fitted StandardScaler
        feature_names: List of feature names
        model_path: Path to save model
        scaler_path: Path to save scaler
        metadata_path: Path to save metadata
    """
    # Save model
    joblib.dump(model, model_path)
    
    # Save scaler
    joblib.dump(scaler, scaler_path)
    
    # Save metadata
    metadata = {
        'feature_names': feature_names,
        'n_features': len(feature_names),
        'model_type': type(model).__name__
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model saved to: {model_path}")


def load_model(model_path: str, scaler_path: str, 
              metadata_path: str) -> Tuple:
    """
    Load trained model and preprocessing objects
    
    Returns:
        Tuple of (model, scaler, metadata)
    """
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return model, scaler, metadata


def predict_single_sample(segment: np.ndarray,
                         sfreq: float,
                         model,
                         scaler,
                         feature_names: List[str],
                         channel_names: List[str] = None) -> Dict:
    """
    Predict direction for a single EEG segment
    
    Args:
        segment: EEG data [channels x samples]
        sfreq: Sampling frequency
        model: Trained model
        scaler: Fitted scaler
        feature_names: List of feature names used in training
        channel_names: Optional channel names
        
    Returns:
        Dictionary with prediction and confidence
    """
    from utils.feature_extraction import extract_all_features
    
    # Extract features
    features = extract_all_features(segment, sfreq, channel_names)
    
    # Ensure features match training features
    feature_vector = []
    for fname in feature_names:
        if fname in features:
            feature_vector.append(features[fname])
        else:
            feature_vector.append(0.0)
    
    X = np.array(feature_vector).reshape(1, -1)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Standardize
    X_scaled = scaler.transform(X)
    
    # Predict
    prediction = model.predict(X_scaled)[0]
    probabilities = model.predict_proba(X_scaled)[0]
    
    # Convert to direction
    direction = 'LR' if prediction == 1 else 'RL'
    confidence = probabilities[prediction]
    
    return {
        'prediction': int(prediction),
        'direction': direction,
        'confidence': float(confidence),
        'probabilities': {
            'RL': float(probabilities[0]),
            'LR': float(probabilities[1])
        }
    }


def get_feature_importance(model, feature_names: List[str]) -> pd.DataFrame:
    """
    Get feature importance from trained model
    
    Args:
        model: Trained model
        feature_names: List of feature names
        
    Returns:
        DataFrame with feature importances sorted by importance
    """
    if hasattr(model, 'feature_importances_'):
        # Tree-based models
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # Linear models
        importances = np.abs(model.coef_[0])
    else:
        # Models without feature importance
        return pd.DataFrame()
    
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    df = df.sort_values('importance', ascending=False)
    
    return df


def generate_classification_report(results: Dict,
                                   best_model_name: str,
                                   output_path: str = None) -> str:
    """
    Generate detailed classification report
    
    Args:
        results: Dictionary of all model results
        best_model_name: Name of best model
        output_path: Optional path to save report
        
    Returns:
        Report string
    """
    report = []
    report.append("=" * 80)
    report.append("BCI MOTOR IMAGERY CLASSIFICATION REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Summary table
    report.append("MODEL COMPARISON")
    report.append("-" * 80)
    report.append(f"{'Model':<15} {'CV Accuracy':<15} {'Test Accuracy':<15} {'F1 Score':<15}")
    report.append("-" * 80)
    
    for name, result in results.items():
        if 'error' not in result:
            report.append(
                f"{name:<15} "
                f"{result['cv_mean']:.4f}±{result['cv_std']:.4f}   "
                f"{result['test_accuracy']:.4f}        "
                f"{result['f1_score']:.4f}"
            )
    
    report.append("")
    report.append(f"BEST MODEL: {best_model_name}")
    report.append("=" * 80)
    
    # Detailed metrics for best model
    best = results[best_model_name]
    report.append("")
    report.append("DETAILED METRICS (Best Model)")
    report.append("-" * 80)
    report.append(f"  Training Accuracy: {best['train_accuracy']:.4f}")
    report.append(f"  Test Accuracy:     {best['test_accuracy']:.4f}")
    report.append(f"  Precision:         {best['precision']:.4f}")
    report.append(f"  Recall:            {best['recall']:.4f}")
    report.append(f"  F1 Score:          {best['f1_score']:.4f}")
    report.append(f"  ROC AUC:           {best['roc_auc']:.4f}")
    report.append("")
    
    # Confusion matrix
    cm = np.array(best['confusion_matrix'])
    report.append("CONFUSION MATRIX")
    report.append("-" * 80)
    report.append(f"              Predicted RL    Predicted LR")
    report.append(f"  Actual RL:  {cm[0,0]:<14}  {cm[0,1]:<14}")
    report.append(f"  Actual LR:  {cm[1,0]:<14}  {cm[1,1]:<14}")
    report.append("")
    
    report_text = "\n".join(report)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_text)
        print(f"Report saved to: {output_path}")
    
    return report_text