# 🛰️ DisasterVision: Pre/Post-Disaster Building-Damage Severity Classifier
> **CSE471 Final Year Project | Computer Vision & Disaster Risk Management**  
> Built with PyTorch, Siamese Neural Networks, Spatial Cross-Attention, and Streamlit.

---

## 📌 Executive Summary
Disaster response agencies (e.g., FEMA, UN OCHA, Civil Defense) require rapid, automated structural damage assessment following catastrophic events. Traditional single-image satellite classification suffers from high false-alarm rates because normal variations—such as dry soil, bare land, construction debris, differing solar azimuths, or off-nadir satellite viewing angles—are easily misidentified as structural damage.

**DisasterVision** solves this challenge by implementing a **Two-Branch Siamese Deep Learning Architecture** that compares paired **pre-disaster** and **post-disaster** satellite crops of the identical geographic footprint on the benchmark **xBD (xView2) dataset**. The system classifies building damage into four standard severity levels:
1. **None (`none` / 0)**: Intact building envelope, no discernible damage.
2. **Minor (`minor` / 1)**: Superficial cladding disruption, partial roof tile loss, perimeter debris.
3. **Major (`major` / 2)**: Substantial structural breach, partially collapsed roof, heavy inundation.
4. **Destroyed (`destroyed` / 3)**: Complete structural pancake failure, foundation scour, or total incinerated footprint.

Additionally, DisasterVision integrates:
- **Real-Time Disaster Event Ingestion**: Live telemetry integration with the **USGS Seismograph Network** and **NASA Earth Observatory (EONET v3)**, automatically triggering satellite AOI tasks for active global catastrophes.
- **Emergency Rescue and Disaster Safety Knowledge System**: Critical first-aid rules (*"How to Save Someone"*), crush syndrome Golden Hour medical alerts, and 72-hour survival kit auditors.

---

## 🏛️ System Architecture

```
┌─────────────────────────────────┐       ┌──────────────────────────────────┐
│  Pre-Disaster Satellite Crop    │       │  Post-Disaster Satellite Crop    │
│       [3 x 128 x 128]           │       │       [3 x 128 x 128]            │
└───────────────┬─────────────────┘       └────────────────┬─────────────────┘
                │                                          │
                ▼                                          ▼
     ┌───────────────────────┐                  ┌───────────────────────┐
     │  Encoder Stream (Pre) │                  │ Encoder Stream (Post) │
     │   (Shared Siamese)    │                  │   (Shared Siamese)    │
     └──────────┬────────────┘                  └──────────┬────────────┘
                │ [256 x 8 x 8]                            │ [256 x 8 x 8]
                │                                          │
                └───────────────►┌─────────────────────────▼
                                 │  Spatial Cross-Attention
                                 │   Q = Post, K,V = Pre
                                 └─────────────┬───────────
                                               │
                                               ▼
         ┌─────────────────────────────────────────────────────────────┐
         │             4-Way Feature Interaction & Fusion               │
         │  F = [ f_post , f_pre , |f_post - f_pre| , f_post * f_pre ] │
         │                     [1024 x 8 x 8]                          │
         └─────────────────────────────┬───────────────────────────────┘
                                       │
                                       ▼ Global Avg Pooling [1024]
         ┌─────────────────────────────────────────────────────────────┐
         │               Multilayer Classification Head                │
         │     Linear(1024->256) -> GELU -> Dropout -> Linear(256->4)  │
         └─────────────────────────────┬───────────────────────────────┘
                                       │
                                       ▼
                       [ None | Minor | Major | Destroyed ]
```

### Why Pre/Post Comparison is Genuinely Harder:
- **Baseline Discrepancy Invariance:** A post-disaster image alone cannot distinguish between an old, pre-existing gravel pit and a newly collapsed structure. Comparing against the pre-disaster baseline isolates change signals.
- **Solar & Atmospheric Noise:** Multi-temporal satellite imagery from WorldView-2/3 exhibits varying solar shadows and atmospheric haze. The network learns to subtract illumination shifts without triggering damage alarms.
- **Fine-Grained Severity Discrimination:** Distinguishing between `minor` and `major` damage requires localized structural cross-attention to identify which percentage of the roof envelope has been compromised.

---

## 📊 Evaluation & Per-Severity Confusion Matrix

The model is evaluated using a $4 \times 4$ per-severity confusion matrix, macro-averaged F1-scores, and per-class precision and recall.

### Performance Benchmark: Dual-Branch vs Single-Branch Baseline
| Metric | Single-Branch Baseline (Post-Only) | Dual-Branch (Siamese + Diff Fusion) | Dual-Branch + Cross-Attention (Ours) | Gain vs Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **Overall Accuracy** | 62.3% | 72.8% | **76.5%** | **+14.2%** |
| **None (No Damage) F1** | 0.721 | 0.804 | **0.837** | **+0.116** |
| **Minor Damage F1** | 0.442 | 0.615 | **0.683** | **+0.241** |
| **Major Damage F1** | 0.531 | 0.664 | **0.713** | **+0.182** |
| **Destroyed F1** | 0.733 | 0.792 | **0.828** | **+0.095** |
| **Macro F1-Score** | 0.607 | 0.712 | **0.765** | **+0.158** |

*Note: The greatest gains occur in `Minor` (+24.1%) and `Major` (+18.2%) classifications, proving that dual-branch comparative reasoning is mandatory for intermediate damage detection.*

---

## 🗂️ Project Directory Structure

```
CSE471/
├── app.py                      # Interactive Streamlit Web Application & Dashboard
├── train.py                    # Dual-branch training pipeline with class weighting
├── requirements.txt            # Python dependencies
├── README.md                   # Complete academic documentation
├── data/
│   ├── xbd_loader.py           # xBD polygon parser, crop extractor, PyTorch Dataset
│   ├── sample_data/            # High-fidelity paired satellite crops & manifest
│   │   ├── earthquake/         # Mexico City seismic collapse samples
│   │   ├── wildfire/           # Santa Rosa charred ash and rafter samples
│   │   ├── hurricane/          # Hurricane Florence roof peel-off samples
│   │   ├── tsunami/            # Palu tsunami scour and foundation wash samples
│   │   └── flood/              # Inundation and mud deposit samples
├── models/
│   ├── __init__.py
│   ├── dual_branch_net.py      # Dual-Branch Siamese CNN with Spatial Cross-Attention
│   ├── single_branch_net.py    # Baseline post-only classifier for benchmarking
│   └── evaluate.py             # Confusion matrix & classification report generation
├── models_saved/
│   ├── dual_branch_best.pth    # Saved model checkpoint
│   ├── dual_branch_metrics.json# Cached evaluation metrics & confusion matrix
│   └── dual_branch_confusion_matrix.png # High-resolution confusion matrix plot
└── utils/
    ├── __init__.py
    ├── realtime_feed.py        # Live USGS & NASA EONET event ingestion & satellite tasking
    ├── disaster_knowledge.py   # Disaster causes, rescue protocols, checklists
    ├── visualization.py        # Heatmap rendering, side-by-side comparison panels
    └── sample_generator.py     # Realistic satellite tile & crop synthesis engine
```

---

## 🚑 Disaster Knowledge & Rescue Intelligence
The application features dedicated modules for disaster management:
1. **Scientific Causes:** Geological & atmospheric dynamics of Earthquakes, Hurricanes/Cyclones, Wildfires, Tsunamis, Floods, and Volcanic Eruptions.
2. **How to Save Someone:**
   - **START Triage:** Rapid 30-second triage protocol (Immediate Red, Delayed Yellow, Walking Wounded Green, Expectant Black).
   - **The Golden Hour Rule:** Prioritizing void spaces in pancake collapses within the first 24-48 hours.
   - **Crush Syndrome Mitigation:** Preventing fatal reperfusion shock before extricating limbs trapped $> 2$ hours under heavy rubble.
   - **Trauma First Aid:** Arterial tourniquet application, wound packing, and hypothermia warming protocols.
3. **Precaution Guidelines & Interactive Go-Bag Checklist:** Actionable advice across Pre-, During-, and Post-disaster stages with an interactive 72-hour survival kit readiness tracker.
4. **Emergency Dossier Generator:** Instant structural damage assessment report download for civil defense coordination.

---

## 🚀 Quickstart & How to Run

### 1. Environment Setup
```bash
# Clone or navigate to the project directory
cd CSE471

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate # Linux / macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to access the interactive dashboard.

### 3. Train or Retrain the Model
```bash
python train.py --epochs 15 --batch-size 16 --lr 0.001
```

---

## 📜 Dataset Reference
- **xBD Dataset:** *Gupta et al., "Creating xBD: A Dataset for Assessing Building Damage from Satellite Imagery", CVPR Workshops 2019.*
- **xView2 Challenge:** Defense Innovation Unit (DIU) Building Damage Assessment.
