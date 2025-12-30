# pages/combined_demo.py
"""
Combined Demo Page (fixed animation + PDF report download)
- Uses saved model if available (won't retrain each time).
- If model missing: runs a one-time auto-train (batch segmentation -> features -> train -> save).
- Upload any EDF -> auto-segment -> extract features -> predict -> show plots -> cursor demo -> download PDF report.
"""

import streamlit as st
import sys
from pathlib import Path
import json
import time
import io
import numpy as np
import pandas as pd

# PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

sys.path.append(str(Path(__file__).parent.parent))

# project utils
from utils.segmentation import batch_segment_files, segment_edf_file
from utils.feature_extraction import extract_all_features, build_feature_dataset
from utils.ml_models import (
    load_model, predict_single_sample, prepare_data,
    train_all_models, select_best_model, save_model
)
from utils.visualization import plot_selected_segment, plot_power_spectrum

st.set_page_config(page_title="BCI Demo - Predict + Features + Cursor", page_icon="🧠", layout="wide")
st.title("BCI Demo — Upload a sample, get features, prediction & cursor demo")
st.markdown("Upload an EDF (any name). If no model exists the app will run a one-time training from `data/raw/` EDFs.")

# Paths
ROOT = Path(".")
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
FEAT_DIR = ROOT / "data" / "features"
MODELS_DIR = ROOT / "models"

MODEL_FILE = MODELS_DIR / "best_model.joblib"
SCALER_FILE = MODELS_DIR / "scaler.joblib"
META_FILE = MODELS_DIR / "model_metadata.json"
TRAIN_INFO = MODELS_DIR / "training_info.json"

# ensure directories exist
PROC_DIR.mkdir(parents=True, exist_ok=True)
FEAT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Auto-train helper ----------
def auto_train_once():
    raw_files = list(RAW_DIR.glob("*.edf"))
    if len(raw_files) == 0:
        raise RuntimeError(f"No EDF files found in {RAW_DIR}. Please add training EDFs.")

    st.info("Auto-train: running batch segmentation on training EDFs...")
    batch_segment_files(str(RAW_DIR), str(PROC_DIR), window_len=3.0, step=0.25, overwrite=False, verbose=False)
    st.success("Segmentation complete.")

    # build feature dataset
    segment_info_list = []
    for meta in sorted(PROC_DIR.glob("*_metadata.json")):
        try:
            with open(meta, "r") as f:
                m = json.load(f)
            base = Path(meta).stem.replace("_metadata", "")
            seg_path = PROC_DIR / f"{base}_segment.npy"
            if seg_path.exists():
                m['segment_path'] = str(seg_path)
                segment_info_list.append(m)
        except Exception:
            continue

    if len(segment_info_list) == 0:
        raise RuntimeError("No processed metadata found after segmentation.")

    df = build_feature_dataset(segment_info_list, str(PROC_DIR), use_filename_labels=True)
    if df is None or len(df) == 0:
        raise RuntimeError("Feature dataset empty after build.")

    feat_csv = FEAT_DIR / "dataset_features.csv"
    df.to_csv(feat_csv, index=False)
    st.success(f"Feature dataset created: {feat_csv} ({len(df)} samples)")

    # train models
    st.info("Training ML models (this can take a few minutes)...")
    X_train, X_test, y_train, y_test, feature_names, scaler = prepare_data(df, test_size=0.2)
    results = train_all_models(X_train, y_train, X_test, y_test, cv_folds=5)
    best_name, best_result = select_best_model(results)
    if best_name is None:
        raise RuntimeError("Training failed.")

    save_model(best_result['model'], scaler, feature_names, str(MODEL_FILE), str(SCALER_FILE), str(META_FILE))

    info = {
        'model_name': best_name,
        'test_accuracy': best_result.get('test_accuracy', 0.0),
        'training_date': time.strftime("%Y-%m-%d %H:%M:%S"),
        'n_samples': len(df),
        'n_features': len(feature_names)
    }
    with open(TRAIN_INFO, 'w') as f:
        json.dump(info, f, indent=2)

    st.success(f"Auto-train finished. Best model: {best_name} (test_acc={info['test_accuracy']:.3f})")
    return best_name, best_result

# auto-train if model missing
if not MODEL_FILE.exists() or not SCALER_FILE.exists() or not META_FILE.exists():
    try:
        with st.spinner("No saved model found. Starting one-time auto-train..."):
            auto_train_once()
    except Exception as e:
        st.error(f"Auto-train failed: {e}")
        st.stop()

# load model
try:
    model, scaler, metadata = load_model(str(MODEL_FILE), str(SCALER_FILE), str(META_FILE))
    training_info = {}
    if TRAIN_INFO.exists():
        training_info = json.load(open(TRAIN_INFO))
    st.success("Model loaded — ready for inference.")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# ---------- Upload & process single sample ----------
st.markdown("---")
st.header("1) Upload EEG sample (any filename)")

uploaded = st.file_uploader("Upload EDF file for analysis", type=["edf"], key="uploader")

if uploaded is None:
    st.info("Upload an EDF file to analyze (filename need not contain direction).")
    st.stop()

tmp_path = Path("temp_sample.edf")
with open(tmp_path, "wb") as f:
    f.write(uploaded.getbuffer())
st.success(f"Saved uploaded file: {uploaded.name}")

# buttons: Process & Predict
if st.button("Process & Predict", key="process_button"):
    # do segmentation -> features -> predict and store in session_state
    try:
        segment, seg_info, seg_result = segment_edf_file(str(tmp_path), preprocess=True)
        if segment is None:
            st.error("Segmentation failed.")
            st.stop()
    except Exception as e:
        st.error(f"Segmentation error: {e}")
        st.stop()

    try:
        features = extract_all_features(segment, seg_info['sfreq'], seg_info.get('motor_channels', None))
    except Exception as e:
        st.error(f"Feature extraction failed: {e}")
        st.stop()

    try:
        prediction = predict_single_sample(
            segment,
            seg_info['sfreq'],
            model,
            scaler,
            metadata['feature_names'],
            seg_info.get('motor_channels', None)
        )
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # store in session_state so reruns don't lose them
    st.session_state['last_segment'] = segment
    st.session_state['last_seg_info'] = seg_info
    st.session_state['last_features'] = features
    st.session_state['last_prediction'] = prediction

    # ensure cursor state exists
    if "cursor_pos" not in st.session_state:
        st.session_state.cursor_pos = 50  # center

    # clear any prior run flag
    st.session_state.run_demo = False

    st.success("Processing complete — scroll down for results.")

# If we've already processed (session state), show results and controls
if 'last_prediction' in st.session_state:
    prediction = st.session_state['last_prediction']
    features = st.session_state['last_features']
    seg_info = st.session_state['last_seg_info']
    segment = st.session_state['last_segment']

    # Prediction summary
    st.markdown("### Prediction")
    direction = prediction['direction']
    confidence = prediction['confidence']
    color = "🔴" if direction == "LR" else "🔵"

    st.markdown(f"""
    <div style='text-align:center; padding:16px; background:linear-gradient(135deg,#667eea 0%,#764ba2 100%); border-radius:10px;'>
        <h2 style='color:white; margin:0;'>{color} {direction}</h2>
        <h4 style='color:white; margin:6px 0;'>Direction: {"Left → Right" if direction=="LR" else "Right → Left"}</h4>
        <h3 style='color:white; margin:6px 0;'>Confidence: {confidence:.1%}</h3>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("RL Probability", f"{prediction['probabilities']['RL']:.1%}")
    with col2:
        st.metric("LR Probability", f"{prediction['probabilities']['LR']:.1%}")

    # Features sample table
    st.markdown("### Extracted Features (sample)")
    feat_items = {k: v for k, v in features.items() if isinstance(v, (int, float))}
    feat_df = pd.DataFrame(list(feat_items.items()), columns=["feature", "value"]).sort_values("feature")
    st.dataframe(feat_df.head(60), use_container_width=True)

    # Plots
    st.markdown("### Signal Plots")
    try:
        fig1 = plot_selected_segment(segment, seg_info['sfreq'], seg_info.get('motor_channels', None), seg_info)
        st.pyplot(fig1)
    except Exception:
        st.warning("Couldn't plot selected segment.")

    try:
        fig2 = plot_power_spectrum(segment, seg_info['sfreq'], seg_info.get('motor_channels', None))
        st.pyplot(fig2)
    except Exception:
        st.warning("Couldn't plot power spectrum.")

    # ---------- Cursor demo (interactive, stable) ----------
    st.markdown("### Cursor Demo (simulated)")
    st.write("Click **Run Cursor Demo** to animate the cursor based on prediction confidence. Cursor position persists between demos.")

    # make sure keys exist
    if "cursor_pos" not in st.session_state:
        st.session_state.cursor_pos = 50
    if "run_demo" not in st.session_state:
        st.session_state.run_demo = False

    # compute movement target
    base_pos = float(st.session_state.cursor_pos)
    move = int(np.clip(confidence * 40, 3, 40))
    if direction == "LR":
        target_pos = min(100, base_pos + move)
        caption_text = "Moving RIGHT →"
    else:
        target_pos = max(0, base_pos - move)
        caption_text = "← Moving LEFT"

    # placeholder for animation
    canvas_placeholder = st.empty()

    def render_cursor(pos_percent: float, caption: str = ""):
        html = f"""
        <div style='width:100%; height:140px; background:linear-gradient(90deg,#e3f2fd 0%,#bbdefb 50%,#90caf9 100%);
                    border-radius:8px; position:relative; border:2px solid #1976d2;'>
            <div style='position:absolute; left:{pos_percent}%; top:50%; transform:translate(-50%,-50%);
                        width:36px; height:36px; background:#f44336; border-radius:50%; border:3px solid white; box-shadow:0 4px 8px rgba(0,0,0,0.25);'></div>
            <div style='position:absolute; left:10px; top:10px; color:#1976d2; font-weight:bold;'>← RL</div>
            <div style='position:absolute; right:10px; top:10px; color:#1976d2; font-weight:bold;'>LR →</div>
            <div style='position:absolute; left:50%; bottom:8px; transform:translateX(-50%); color:#666; font-style:italic;'>{caption}</div>
        </div>
        """
        canvas_placeholder.markdown(html, unsafe_allow_html=True)

    # show static
    render_cursor(st.session_state.cursor_pos, caption="Current position")

    # callbacks to set run_demo or reset
    def start_demo_cb():
        st.session_state.run_demo = True

    def reset_cursor_cb():
        st.session_state.cursor_pos = 50

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.button("▶️ Run Cursor Demo", on_click=start_demo_cb, key="run_demo_button")
    with col_b:
        st.button("⟲ Reset Cursor to Center", on_click=reset_cursor_cb, key="reset_button")

    # perform animation if flag set (this runs during the rerun)
    if st.session_state.get("run_demo", False):
        steps = 25
        start = float(st.session_state.cursor_pos)
        for i in range(steps + 1):
            interp = start + (target_pos - start) * (i / steps)
            frame_caption = caption_text if i == steps else ""
            render_cursor(interp, caption=frame_caption)
            time.sleep(0.03)
        st.session_state.cursor_pos = float(target_pos)
        st.session_state.run_demo = False
        st.success("✅ Movement complete!")

    # ---------- Report: only download button (PDF) ----------
    st.markdown("### Download Single-sample Report (PDF)")

    def build_report_pdf_bytes(filename: str, direction: str, confidence: float, seg_info: dict, top_features: dict):
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter
        x = 40
        y = height - 40
        c.setFont("Helvetica-Bold", 14)
        c.drawString(x, y, "BCI Single-sample Report")
        y -= 24
        c.setFont("Helvetica", 10)
        c.drawString(x, y, f"File: {filename}")
        y -= 14
        c.drawString(x, y, f"Predicted direction: {direction} (confidence {confidence:.3f})")
        y -= 14
        c.drawString(x, y, f"ERD score: {seg_info.get('erd_score', 'N/A')}")
        y -= 14
        c.drawString(x, y, f"Quality score: {seg_info.get('quality_score', 'N/A')}")
        y -= 20
        c.setFont("Helvetica-Bold", 12)
        c.drawString(x, y, "Top features (sample):")
        y -= 16
        c.setFont("Helvetica", 9)
        count = 0
        for feat, val in list(top_features.items())[:40]:
            if y < 60:
                c.showPage()
                y = height - 40
                c.setFont("Helvetica", 9)
            c.drawString(x, y, f"{feat}: {val:.6g}")
            y -= 12
            count += 1
        c.showPage()
        c.save()
        buffer.seek(0)
        return buffer.getvalue()

    pdf_bytes = None
    if st.button("📥 Generate PDF Report"):
        # build bytes and show download button
        pdf_bytes = build_report_pdf_bytes(
            uploaded.name,
            direction,
            confidence,
            seg_info,
            feat_items
        )
        st.success("PDF generated below — click to download")
        st.download_button("Download PDF report", data=pdf_bytes, file_name=f"bci_report_{Path(uploaded.name).stem}.pdf", mime="application/pdf")

st.markdown("---")
st.info("Tip: Model trained from labeled EDFs in data/raw/ (one-time).")
