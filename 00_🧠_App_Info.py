import streamlit as st
from PIL import Image
from pathlib import Path

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="BCI Motor Imagery Project",
    page_icon="🧠",
    layout="wide"
)

# -------------------------------------------------
# PROJECT DETAILS
# -------------------------------------------------
PROJECT_TITLE = "Characterizing the Brain Waves Associated with Controlled Hand Movement"
SUBTITLE = "Brain–Computer Interface (BCI) using Motor Imagery"

AUTHORS = (
    "Sanjay R , "
    "Mehaboob R, Pradeep V N, Radhika S Naik"
)

DEPARTMENT = "Department of Electrical and Electronics Engineering"
INSTITUTE = "BMS Institute of Technology and Management"
GUIDE = "Dr. Prashanth A. Athavale"
YEAR = "2025"

LOGO_PATH = "assets/logo.png"   # make sure logo exists

# -------------------------------------------------
# BIG LOGO / BANNER
# -------------------------------------------------
if Path(LOGO_PATH).exists():
    st.image(Image.open(LOGO_PATH), use_column_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------------------------
# TITLE SECTION (POSTER STYLE)
# -------------------------------------------------
st.markdown(
    f"""
    <div style="text-align:center;">
        <h1 style="font-size:42px;">{PROJECT_TITLE}</h1>
        <h3 style="font-weight:400;">{SUBTITLE}</h3>
        <br>
        <p style="font-size:20px;"><b>{AUTHORS}</b></p>
        <p style="font-size:22px;"><b>{DEPARTMENT}</b></p>
        <p style="font-size:22px;"><b>{INSTITUTE}</b></p>
        <p style="font-size:18px;">Guide: <b>{GUIDE}</b></p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# -------------------------------------------------
# INTRODUCTION
# -------------------------------------------------
st.subheader("🧠 Introduction")
st.markdown("""
The human brain generates electrical signals during movement, observation, and imagination.
Using EEG signals recorded from the motor cortex, this project characterizes brain-wave
patterns associated with **controlled hand movement** and **motor imagery**.

The objective is to translate these neural signals into meaningful **directional commands**
using signal processing and machine learning techniques.
""")

# -------------------------------------------------
# EXPERIMENTAL PROCEDURE
# -------------------------------------------------
st.subheader("🔄 Experimental Procedure")
st.markdown("""
**Baseline → Cue → Movement Task → Break → Observation → Break → Motor Imagery Task**  
*(Repeated for opposite direction)*
""")

# -------------------------------------------------
# METHODOLOGY / PIPELINE
# -------------------------------------------------
st.subheader("⚙️ Methodology (ML Pipeline)")
st.markdown("""
1. **Raw EEG Recording** using Mitsar SmartBCI x24  
2. **Auto-Segmentation** using ERD (optimal 3-second motor imagery window)  
3. **Feature Extraction** (time-domain, frequency-domain, ERD-based features)  
4. **Machine Learning Classification** (Left–Right vs Right–Left)  
5. **Cursor Movement Simulation**
""")

# -------------------------------------------------
# OBJECTIVES
# -------------------------------------------------
st.subheader("🎯 Objectives")
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
- Record EEG signals during controlled hand or cursor movement  
- Detect unique brain-wave signatures linked to movement direction  
""")

with col2:
    st.markdown("""
- Identify motor cortex activation using signal processing  
- Build a BCI system that predicts user intention from EEG signals  
""")

# -------------------------------------------------
# EQUIPMENT USED
# -------------------------------------------------
st.subheader("🧰 Equipment Used")
st.markdown("""
- **Mitsar SmartBCI x24** – 24-channel EEG headset (dry electrodes)  
- Wireless amplifier and Bluetooth adapter  
- EEGStudio acquisition software  
""")

# -------------------------------------------------
# RESULTS & OBSERVATIONS
# -------------------------------------------------
st.subheader("📊 Results & Observations")
st.markdown("""
- Clear differences in EEG rhythms, especially **mu and beta bands**, were observed  
- Machine learning models successfully predicted movement direction  
- Cursor demo responds dynamically based on model confidence  
""")

# -------------------------------------------------
# APPLICATIONS
# -------------------------------------------------
st.subheader("🚀 Applications")
st.markdown("""
- Neuro-rehabilitation systems  
- Assistive devices  
- Prosthetic control  
- Hands-free Human–Computer Interaction (HCI)  
- Neuroscience research  
""")

# -------------------------------------------------
# SDG IMPACT
# -------------------------------------------------
st.subheader("🌍 SDG Impact")
st.markdown("""
- **SDG 3:** Good Health and Well-Being  
- **SDG 9:** Industry, Innovation and Infrastructure  
- **SDG 4:** Quality Education  
""")

st.markdown("---")

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.caption(f"""
Work conducted at Computational Neuroscience & Engineering Research Laboratory, BMSIT&M  
© {YEAR} | Team Lead: Sanjay R | (sanjay.r.4110@gmail.com)
""")
