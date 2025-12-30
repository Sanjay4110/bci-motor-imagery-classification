# app.py
import streamlit as st
from pathlib import Path
import json
import pandas as pd



st.set_page_config(
    page_title="BCI Motor Imagery Classifier",
    page_icon="🧠",
    layout="wide"
)

# ------------------ HEADER ------------------
st.title("🧠 BCI Motor Imagery Direction Classifier")
st.markdown("""
### **Real-time Left ↔ Right Motor Imagery Prediction**
This application processes raw EEG signals, extracts neural features, and predicts the **intended cursor direction**  
using a pre-trained machine learning model.

It is designed for brain–computer interface (BCI) systems where motor imagery  
(mentally imagining left or right hand movement) is used to control digital interfaces.

---
""")

# ------------------ PROJECT OVERVIEW ------------------
st.subheader("🎯 Project Overview")
st.markdown("""
This BCI system automatically:

- 🧩 **Segments EEG signals** to isolate the most meaningful 3-second motor-imagery window  
- 📈 **Extracts 40+ temporal, spectral, and ERD-based features**  
- 🤖 **Classifies the user’s mental command** as **Left→Right (LR)** or **Right→Left (RL)**  
- 🎮 **Displays a real-time cursor demo** driven by the prediction  
- 📄 **Generates structured PDF reports** for demonstrations and analysis  

All training is done **offline** using 98 labeled EEG samples.  
During testing, **any EDF file** can be uploaded — the model predicts direction purely from the signal.
""")

st.markdown("---")

# ------------------ SYSTEM STATUS ------------------
st.subheader("📊 System Status")

# Check available data
raw_count = len(list(Path("data/raw").glob("*.edf")))
proc_count = len(list(Path("data/processed").glob("*_segment.npy")))
feat_count = len(list(Path("data/features").glob("*.csv")))
model_count = len(list(Path("models").glob("*.joblib")))

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📁 Raw EDF Files", raw_count)
with col2:
    st.metric("✂️ Segmented Windows", proc_count)
with col3:
    st.metric("📊 Feature Datasets", feat_count)
with col4:
    st.metric("🤖 Saved Models", model_count)

if model_count > 0:
    st.success("✅ System ready! You can now test predictions using the **Combined Demo** page.")
else:
    st.warning("⚠️ No trained model found. System will auto-train when first needed.")

st.markdown("---")

# ------------------ PROJECT PURPOSE ------------------
st.subheader("💡 Purpose & Application")
st.markdown("""
This project demonstrates a practical implementation of **Brain–Computer Interfaces**  
where users can control a system using **motor imagery alone**, without physical movement.

Potential applications include:  
- Neuro-rehabilitation  
- Assistive technologies  
- Hands-free computer interaction  
- Research on motor cortex activation  

This system showcases the full pipeline from **raw EEG → features → prediction → cursor control**.
""")

st.markdown("---")

st.caption("BCI Motor Imagery Classification System — Developed for academic demonstration and research use.")
