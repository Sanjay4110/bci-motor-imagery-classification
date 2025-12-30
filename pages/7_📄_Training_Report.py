# pages/7_📄_Training_Report.py
"""
Training Report (single PDF) — quality_score removed
Generates one consolidated PDF containing:
 - System summary
 - Dataset statistics
 - Model performance
 - Detailed analysis (aggregate ERD only; quality removed)
"""

import streamlit as st
from pathlib import Path
import json
import pandas as pd
import io
import time
from datetime import datetime

# PDF libs
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

st.set_page_config(page_title="Training Report", page_icon="📄", layout="wide")
st.title("📄 Training Report")
st.markdown("One consolidated PDF with summary, dataset stats and model performance. Quality metrics removed from report.")

# Paths
FEATURES_FILE = Path("data/features/dataset_features.csv")
PROCESSED_DIR = Path("data/processed")
TRAIN_INFO = Path("models/training_info.json")

# Load available data
df = None
if FEATURES_FILE.exists():
    try:
        df = pd.read_csv(FEATURES_FILE)
    except Exception:
        df = None

n_segments = len(list(PROCESSED_DIR.glob("*_segment.npy"))) if PROCESSED_DIR.exists() else 0

training_info = {}
if TRAIN_INFO.exists():
    try:
        training_info = json.loads(TRAIN_INFO.read_text())
    except Exception:
        training_info = {}

# Top-line metrics on page
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("📁 Feature samples", len(df) if df is not None else 0)
with col2:
    st.metric("✂️ Processed segments", n_segments)
with col3:
    st.metric("🤖 Trained model", training_info.get("model_name", "None"))

st.markdown("---")


# Helper: build PDF bytes (no quality_score anywhere)
def build_training_pdf_bytes():
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    margin = 30
    x = margin
    y = height - margin

    def new_page():
        nonlocal x, y
        c.showPage()
        x = margin
        y = height - margin

    def write_heading(text):
        nonlocal y
        c.setFont("Helvetica-Bold", 14)
        c.drawString(x, y, text)
        y -= 18

    def write_line(text, indent=0, fontsize=10):
        nonlocal y
        c.setFont("Helvetica", fontsize)
        c.drawString(x + indent, y, text)
        y -= (12 if fontsize == 10 else 14)

    # Cover
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(width / 2, y, "BCI Motor Imagery — Training Report")
    y -= 30
    c.setFont("Helvetica", 10)
    c.drawCentredString(width / 2, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 30

    # SYSTEM SUMMARY
    write_heading("SYSTEM SUMMARY")
    write_line("-" * 80)
    write_line(f"Total feature samples: {len(df) if df is not None else 0}")
    write_line(f"Processed segments: {n_segments}")
    write_line(f"Trained model: {training_info.get('model_name', 'N/A')}")
    write_line(f"Test accuracy: {training_info.get('test_accuracy', 'N/A')}")
    write_line(f"Training date: {training_info.get('training_date', 'N/A')}")
    y -= 6

    if y < 120:
        new_page()

    # DATASET STATISTICS
    write_heading("DATASET STATISTICS")
    write_line("-" * 80)
    if df is None:
        write_line("No feature dataset found (data/features/dataset_features.csv).")
    else:
        try:
            total = len(df)
            write_line(f"Total samples: {total}")
            # class distribution
            write_line("Class distribution (label):")
            counts = df['label'].value_counts().to_dict()
            for k, v in counts.items():
                pct = v / total * 100 if total else 0
                write_line(f"  Label {k}: {v} ({pct:.1f}%)", indent=10)
            y -= 4

            # per-subject counts
            write_line("Per-subject sample counts:")
            subjects = df['subject'].value_counts().to_dict()
            for subj, cnt in subjects.items():
                if y < 80:
                    new_page()
                write_line(f"  {subj}: {cnt}", indent=10)
        except Exception:
            write_line("Failed to compute dataset statistics (invalid CSV).")
    y -= 6

    if y < 120:
        new_page()

    # MODEL PERFORMANCE
    write_heading("MODEL PERFORMANCE")
    write_line("-" * 80)
    if training_info:
        write_line(f"Model name: {training_info.get('model_name','N/A')}")
        write_line(f"Test accuracy: {training_info.get('test_accuracy','N/A')}")
        write_line(f"Number of training samples: {training_info.get('n_samples','N/A')}")
        write_line(f"Number of features: {training_info.get('n_features','N/A')}")
        write_line(f"Training date: {training_info.get('training_date','N/A')}")
    else:
        write_line("No training information found (models/training_info.json).")
    y -= 6

    if y < 120:
        new_page()

    # DETAILED ANALYSIS (ERD only if present)
    write_heading("DETAILED ANALYSIS")
    write_line("-" * 80)
    if df is not None and 'erd_score' in df.columns:
        try:
            avg_erd = df['erd_score'].mean()
            write_line(f"Average ERD score: {avg_erd:.3f}%")
            y -= 4
            # List top subjects by ERD
            write_line("Subjects by average ERD (top 10):")
            subj_erd = df.groupby('subject')['erd_score'].mean().sort_values(ascending=False)
            for subj, q in subj_erd.head(10).items():
                if y < 80:
                    new_page()
                write_line(f"  {subj}: {q:.3f}", indent=10)
        except Exception:
            write_line("Could not compute ERD metrics.")
    else:
        write_line("No ERD metrics available in feature dataset (column 'erd_score' missing).")

    # final notes / end
    y -= 10
    if y < 120:
        new_page()
    write_line("=" * 60)
    write_line("End of Report")
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()

# Generate & download button
if st.button("📥 Generate & Download Training Report (PDF)"):
    try:
        pdf_bytes = build_training_pdf_bytes()
        st.success("PDF generated — click below to download.")
        st.download_button("Download Training Report (PDF)", data=pdf_bytes, file_name=f"training_report_{int(time.time())}.pdf", mime="application/pdf")
    except Exception as e:
        st.error(f"Failed to generate PDF: {e}")

st.markdown("---")
st.info("This report compiles dataset, processed segments and model training info into one PDF for sharing.")
