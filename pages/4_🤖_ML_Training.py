"""
Page 4: Machine Learning Model Training
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json

sys.path.append(str(Path(__file__).parent.parent))

from utils.ml_models import (
    prepare_data, train_all_models, select_best_model,
    save_model, get_feature_importance, generate_classification_report
)
from utils.visualization import (
    plot_confusion_matrix, plot_roc_curve,
    plot_model_comparison, plot_feature_importance
)

st.set_page_config(page_title="ML Training", page_icon="🤖", layout="wide")

st.title("🤖 Machine Learning Training")
st.markdown("Train and evaluate classifiers for LR/RL direction detection")

# Check for feature dataset
features_dir = Path("data/features")
dataset_file = features_dir / "dataset_features.csv"

if not dataset_file.exists():
    st.warning("⚠️ Feature dataset not found. Please extract features first.")
    st.stop()

# Load dataset
df = pd.read_csv(dataset_file)

st.success(f"✅ Loaded dataset: {len(df)} samples")

# Dataset info
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Samples", len(df))
with col2:
    lr_count = len(df[df['label'] == 1])
    st.metric("LR Samples", lr_count)
with col3:
    rl_count = len(df[df['label'] == 0])
    st.metric("RL Samples", rl_count)
with col4:
    metadata_cols = ['filename', 'subject', 'method', 'direction', 'pattern', 
                    'iteration', 'label', 'erd_score', 'quality_score']
    n_features = len([col for col in df.columns if col not in metadata_cols])
    st.metric("Features", n_features)

st.markdown("---")

# Training parameters
st.header("⚙️ Training Configuration")

col1, col2 = st.columns(2)

with col1:
    test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)

with col2:
    cv_folds = st.slider("Cross-validation folds", 3, 10, 5, 1)

# Train button
if st.button("▶️ Train All Models", type="primary"):
    with st.spinner("Preparing data..."):
        X_train, X_test, y_train, y_test, feature_names, scaler = prepare_data(
            df, test_size=test_size
        )
    
    st.success(f"✅ Data prepared: {len(X_train)} training, {len(X_test)} testing samples")
    
    # Train models
    with st.spinner("Training models... This may take a few minutes"):
        results = train_all_models(X_train, y_train, X_test, y_test, cv_folds)
    
    st.success("✅ Training complete!")
    
    # Save results to session state
    st.session_state['training_results'] = results
    st.session_state['X_test'] = X_test
    st.session_state['y_test'] = y_test
    st.session_state['feature_names'] = feature_names
    st.session_state['scaler'] = scaler
    
    # Select best model
    best_name, best_result = select_best_model(results)
    st.session_state['best_model_name'] = best_name
    st.session_state['best_model'] = best_result['model']
    
    st.balloons()

# Display results if available
if 'training_results' in st.session_state:
    results = st.session_state['training_results']
    best_name = st.session_state['best_model_name']
    
    st.markdown("---")
    st.header("📊 Training Results")
    
    # Model comparison
    st.subheader("Model Performance Comparison")
    fig = plot_model_comparison(results)
    st.pyplot(fig)
    
    # Best model highlight
    st.success(f"🏆 **Best Model:** {best_name} (Test Accuracy: {results[best_name]['test_accuracy']:.4f})")
    
    # Detailed metrics
    st.subheader("Detailed Metrics (Best Model)")
    
    best_result = results[best_name]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Test Accuracy", f"{best_result['test_accuracy']:.4f}")
    with col2:
        st.metric("Precision", f"{best_result['precision']:.4f}")
    with col3:
        st.metric("Recall", f"{best_result['recall']:.4f}")
    with col4:
        st.metric("F1 Score", f"{best_result['f1_score']:.4f}")
    
    # Confusion matrix and ROC
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Confusion Matrix")
        cm = np.array(best_result['confusion_matrix'])
        fig = plot_confusion_matrix(cm)
        st.pyplot(fig)
    
    with col2:
        st.subheader("ROC Curve")
        fig = plot_roc_curve(
            np.array(best_result['fpr']),
            np.array(best_result['tpr']),
            best_result['roc_auc']
        )
        st.pyplot(fig)
    
    # Feature importance
    if hasattr(st.session_state['best_model'], 'feature_importances_') or \
       hasattr(st.session_state['best_model'], 'coef_'):
        st.subheader("🎯 Feature Importance")
        
        importance_df = get_feature_importance(
            st.session_state['best_model'],
            st.session_state['feature_names']
        )
        
        top_n = st.slider("Number of top features to show", 10, 50, 20, 5)
        
        fig = plot_feature_importance(importance_df, top_n)
        st.pyplot(fig)
    
    # Save model
    st.markdown("---")
    st.subheader("💾 Save Model")
    
    if st.button("Save Best Model"):
        models_dir = Path("models")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = models_dir / "best_model.joblib"
        scaler_path = models_dir / "scaler.joblib"
        metadata_path = models_dir / "model_metadata.json"
        
        save_model(
            st.session_state['best_model'],
            st.session_state['scaler'],
            st.session_state['feature_names'],
            str(model_path),
            str(scaler_path),
            str(metadata_path)
        )
        
        # Save additional info
        info = {
            'model_name': best_name,
            'test_accuracy': best_result['test_accuracy'],
            'training_date': pd.Timestamp.now().isoformat(),
            'n_samples': len(df),
            'n_features': len(st.session_state['feature_names'])
        }
        
        info_path = models_dir / "training_info.json"
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        st.success("✅ Model saved successfully!")
        st.json(info)
    
    # Generate report
    st.markdown("---")
    st.subheader("📄 Classification Report")
    
    report = generate_classification_report(results, best_name)
    st.text(report)
    
    # Download report
    st.download_button(
        label="📥 Download Report",
        data=report,
        file_name="classification_report.txt",
        mime="text/plain"
    )

else:
    st.info("👆 Click 'Train All Models' to start training")

st.markdown("---")
st.info("💡 **Next Step:** Go to '🔍 Test Prediction' to test the model on new samples")