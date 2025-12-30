

# 🧠 Characterizing the Brain Waves Associated with Controlled Hand Movement

**EEG-Based Brain–Computer Interface (BCI) for Motor Imagery Direction Classification**

---

## 📌 Project Overview

This project presents an **end-to-end Brain–Computer Interface (BCI) system** that analyzes EEG signals recorded during controlled hand movement and motor imagery, and predicts the **intended movement direction (Left → Right or Right → Left)** using machine learning.

The system integrates **EEG signal acquisition, preprocessing, ERD-based auto-segmentation, feature extraction, machine learning classification, and a web-based prediction interface** into a single cohesive pipeline.

The project is developed as part of the **VTU BEEP705 Major Project**.

---

## 🎯 Key Objectives

* Record EEG signals during controlled hand movement and motor imagery
* Identify motor cortex activation using **Event-Related Desynchronization (ERD)**
* Extract discriminative EEG features (time, frequency, spatial)
* Train and evaluate ML models for direction classification
* Build an interactive **web application** for prediction and visualization

---

## 🧠 System Architecture

```
EEG Acquisition
      ↓
Preprocessing & Filtering
      ↓
ERD-Based Auto Segmentation
      ↓
Feature Extraction
      ↓
Machine Learning Training
      ↓
Direction Prediction (LR / RL)
      ↓
Web App Visualization & Report
```

---

## 🔬 Data Collection

* **EEG Device:** Mitsar SmartBCI x24
* **Channels:** 24 (10–20 system)
* **Recording Software:** Mitsar EEGStudio
* **Data Format:** EDF
* **Tasks:**

  * Left-to-right hand movement / imagery
  * Right-to-left hand movement / imagery
* **Dataset Size:** 98 EEG samples

Ethical consent was obtained from all participants, and no personal identity data is stored.

---

## ⚙️ Methodology

### 1️⃣ Preprocessing

* Band-pass filtering (1–45 Hz)
* Noise handling and channel selection
* Focus on motor cortex regions (C3, Cz, C4)

### 2️⃣ Auto-Segmentation

* Sliding 3-second window approach
* ERD computed in **Mu (8–12 Hz)** and **Beta (13–30 Hz)** bands
* Window with maximum ERD selected for analysis

### 3️⃣ Feature Extraction

* Time-domain features
* Frequency-domain features
* Spatial and inter-channel features

### 4️⃣ Machine Learning

* Models used:

  * Linear Discriminant Analysis (LDA)
  * Support Vector Machine (SVM)
  * Random Forest
  * Logistic Regression
* Train–test split with feature standardization
* Best model selected based on test accuracy

---

## 🌐 Web Application

An interactive **Streamlit-based web application** was developed to demonstrate the system.

### Features:

* Upload EEG sample (EDF)
* Automatic preprocessing & segmentation
* Direction prediction with confidence score
* Cursor movement simulation
* PDF report generation
* Training summary report

---

## 📊 Results

* Clear ERD patterns observed during motor imagery
* Machine learning models successfully classified movement direction
* Demonstrated practical EEG-based control through cursor visualization

*(Exact accuracy values depend on subject variability and signal quality.)*

---

## 💻 Software & Tools Used

* **EEGStudio (Mitsar)** – EEG acquisition
* **Python** – Core development
* **MNE, NumPy, SciPy** – EEG signal processing
* **Scikit-learn, XGBoost** – Machine learning
* **Streamlit** – Web application
* **Matplotlib** – Visualization
* **ReportLab** – PDF report generation
* **Git & GitHub** – Version control

---

## 📂 Project Structure

```
bci-motor-imagery-classification/
│
├── App_Info.py
├── project_meta.py
├── requirements.txt
├── README.md
│
├── pages/
│   ├── combined_demo.py
│   └── 7_📄_Training_Report.py
│
├── utils/
│   ├── eeg_processing.py
│   ├── segmentation.py
│   ├── feature_extraction.py
│   ├── ml_models.py
│   └── visualization.py
│
├── assets/
│   └── logo.png
│
├── data/
│   ├── raw/        # (ignored in GitHub)
│   ├── processed/ # (ignored in GitHub)
│   └── features/
│
└── models/         # (ignored in GitHub)
```

---

## ▶️ How to Run the Project Locally

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Sanjay4110/bci-motor-imagery-classification.git
cd bci-motor-imagery-classification
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the application

```bash
streamlit run App_Info.py
```

Open browser at:

```
http://localhost:8501
```

---

## ⚠️ Data Availability Note

Raw EEG data and trained model files are **not included** in the repository due to **ethical and privacy considerations**.
The system is designed to work with new EEG samples provided by the user.

---

## 🌍 Sustainable Development Goals (SDG)

* **SDG 3:** Good Health & Well-Being
* **SDG 4:** Quality Education
* **SDG 9:** Industry, Innovation & Infrastructure

---

## 👨‍🏫 Project Team

**Guide:**
Dr. Prashanth A. Athavale

**Team Members:**

* Sanjay R
* Mehaboob R
* Pradeep V N
* Radhika S Naik

**Institution:**
BMS Institute of Technology and Management
Department of Electronics & Communication / Computational Neuroscience Lab

---

## 🏁 Conclusion

This project demonstrates the feasibility of translating EEG-based motor imagery into meaningful directional commands using machine learning. It serves as a strong foundation for future work in real-time BCIs, assistive technologies, and neuro-rehabilitation systems.

---


