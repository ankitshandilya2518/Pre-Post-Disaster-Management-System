"""
DisasterVision: Pre/Post-Disaster Building Safety & Damage Assessment System
=============================================================================
CSE471 Final Year Project | Computer Vision & Civil Protection System
Dual-Branch Siamese Network for xBD Building Damage Severity Classification
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import streamlit as st

# Import project modules
from data.xbd_loader import (
    CLASS_COLORS,
    CLASS_HEX,
    DAMAGE_CLASSES,
    IDX_TO_CLASS,
)
from models.dual_branch_net import load_damage_classifier
from utils.disaster_knowledge import (
    DISASTER_DETAILS,
    DISASTER_TYPES,
    EMERGENCY_SUPPLIES_CHECKLIST,
)
from utils.visualization import (
    blend_heatmap_on_image,
)
from utils.sample_generator import generate_sample_crop_pair, populate_sample_dataset
from utils.realtime_feed import (
    get_satellite_crops_for_event,
)
from utils.india_disaster_feed import (
    INDIAN_DISASTER_EVENTS,
    get_india_disaster_catalog,
    get_ndrf_operational_directive,
)
from utils.report_generator import (
    generate_dossier_id,
    generate_verification_report_docx,
    generate_verification_report_html,
    generate_verification_report_json,
)

# Streamlit Page Config
st.set_page_config(
    page_title="DisasterVision • Building Safety & Damage Assessment",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom Styling: Humanized, Crisp White & Blue Emergency Operations Design System
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">

<style>
    /* Global Typography & Light Theme Reset */
    * {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    .mono {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Page Canvas: Soft Off-White Background */
    .stApp {
        background-color: #f8fafc !important;
        color: #0f172a !important;
    }

    /* Remove Streamlit default blank header gap */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    #MainMenu {
        visibility: hidden;
    }
    footer {
        visibility: hidden;
    }

    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1400px !important;
    }

    /* Streamlit Containers: Pure White Card Surfaces with Crisp Borders */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px !important;
        border: 1px solid #e2e8f0 !important;
        background: #ffffff !important;
        padding: 18px !important;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.04) !important;
    }

    /* Top Brand Bar: Pure White with Vibrant Royal Blue Accents */
    .brand-bar {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 18px 24px;
        margin-bottom: 18px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 14px;
        box-shadow: 0 4px 16px rgba(37, 99, 235, 0.06);
    }
    .brand-left {
        display: flex;
        align-items: center;
        gap: 16px;
    }
    .brand-emblem {
        width: 48px;
        height: 48px;
        background: linear-gradient(135deg, #1d4ed8, #2563eb);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.7rem;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
    }
    .brand-title {
        font-size: 1.55rem;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -0.3px;
        margin-bottom: 2px;
    }
    .brand-subtitle {
        font-size: 0.88rem;
        color: #475569;
        font-weight: 500;
    }
    .brand-status {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        color: #1d4ed8;
        font-size: 0.82rem;
        font-weight: 600;
        padding: 8px 18px;
        border-radius: 30px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .status-dot {
        width: 8px;
        height: 8px;
        background: #2563eb;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 6px #2563eb;
    }

    /* Human Cards (White with Soft Slate Border & Blue Hover Glow) */
    .human-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.04);
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .human-card:hover {
        border-color: #93c5fd;
        box-shadow: 0 6px 18px rgba(37, 99, 235, 0.08);
        transform: translateY(-2px);
    }

    /* Image Display Docks */
    .dock-wrapper {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        overflow: hidden;
        padding: 12px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .dock-wrapper:hover {
        transform: translateY(-2px);
        border-color: #2563eb;
        box-shadow: 0 6px 18px rgba(37, 99, 235, 0.12);
    }
    .dock-header {
        font-size: 0.82rem;
        font-weight: 700;
        color: #1e40af;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }

    /* Verdict Card */
    .verdict-box {
        background: #ffffff;
        border-radius: 12px;
        padding: 22px;
        margin-top: 16px;
        margin-bottom: 18px;
        border: 1px solid #e2e8f0;
        border-left: 6px solid #2563eb;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05);
    }

    /* Soft Safety Status Badges */
    .badge-pill {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 24px;
        font-size: 0.95rem;
        font-weight: 700;
        letter-spacing: 0.3px;
        text-transform: uppercase;
    }
    .badge-none { background: #dcfce7; color: #166534; border: 1px solid #86efac; }
    .badge-minor { background: #fef3c7; color: #92400e; border: 1px solid #fde68a; }
    .badge-major { background: #ffedd5; color: #9a3412; border: 1px solid #fed7aa; }
    .badge-destroyed { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }

    /* KPI Cards */
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 14px 12px;
        text-align: center;
        margin-bottom: 10px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
    }
    .kpi-lbl {
        font-size: 0.74rem;
        color: #64748b;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        margin-bottom: 4px;
    }
    .kpi-val {
        font-size: 1.45rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Action Banners */
    .action-banner {
        border-radius: 10px;
        padding: 16px 20px;
        margin-top: 14px;
        display: flex;
        gap: 14px;
        align-items: flex-start;
        font-size: 0.92rem;
        line-height: 1.6;
    }
    .action-safe {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        color: #14532d;
    }
    .action-warning {
        background: #fffbeb;
        border: 1px solid #fde68a;
        color: #78350f;
    }
    .action-danger {
        background: #fef2f2;
        border: 1px solid #fecaca;
        color: #7f1d1d;
    }

    /* Navigation Buttons: White & Blue Theme */
    div.stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 12px 18px !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button[kind="primary"] {
        background: #2563eb !important;
        color: #ffffff !important;
        border: 1px solid #1d4ed8 !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25) !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background: #1d4ed8 !important;
        color: #ffffff !important;
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.35) !important;
    }
    div.stButton > button[kind="secondary"] {
        background: #ffffff !important;
        color: #1e293b !important;
        border: 1px solid #cbd5e1 !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04) !important;
    }
    div.stButton > button[kind="secondary"]:hover {
        background: #eff6ff !important;
        color: #1d4ed8 !important;
        border-color: #93c5fd !important;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Model & Data
@st.cache_resource
def load_model():
    weights_path = Path("models_saved/dual_branch_best.pth")
    return load_damage_classifier(weights_path if weights_path.exists() else None)


@st.cache_data
def ensure_sample_dataset():
    sample_dir = Path("data/sample_data")
    if not sample_dir.exists() or len(list(sample_dir.rglob("*.png"))) < 20:
        sample_dir.mkdir(parents=True, exist_ok=True)
        populate_sample_dataset(sample_dir, samples_per_class=15)
    return True


ensure_sample_dataset()
model = load_model()

# Session State for Active Navigation
if "active_nav" not in st.session_state:
    st.session_state["active_nav"] = "classifier"

# Top Brand Header (White & Blue)
st.markdown("""
<div class="brand-bar">
    <div class="brand-left">
        <div class="brand-emblem">🏛️</div>
        <div>
            <div class="brand-title">DisasterVision • Building Safety & Damage Assessment</div>
            <div class="brand-subtitle">National Civil Defense & Emergency Response Infrastructure • Post-Disaster Structural Evaluation</div>
        </div>
    </div>
    <div class="brand-status">
        <span class="status-dot"></span>
        Operational • NDMA Structural Triage Standard Active
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# TOP NAVIGATION BAR (Accessible, Prominent, White & Blue Theme)
# ==============================================================================
top_n1, top_n2, top_n3, top_n4 = st.columns(4, gap="small")

with top_n1:
    is_t1 = (st.session_state["active_nav"] == "classifier")
    btn1_t = "🏢  Damage Assessment Studio"
    if st.button(btn1_t, key="top_nav_classifier", type="primary" if is_t1 else "secondary", use_container_width=True):
        st.session_state["active_nav"] = "classifier"
        st.rerun()

with top_n2:
    is_t2 = (st.session_state["active_nav"] == "realtime")
    btn2_t = "🛰️  India Disaster Live Monitor"
    if st.button(btn2_t, key="top_nav_realtime", type="primary" if is_t2 else "secondary", use_container_width=True):
        st.session_state["active_nav"] = "realtime"
        st.rerun()

with top_n3:
    is_t3 = (st.session_state["active_nav"] == "rescue")
    btn3_t = "🚑  Life-Saving Rescue Protocols"
    if st.button(btn3_t, key="top_nav_rescue", type="primary" if is_t3 else "secondary", use_container_width=True):
        st.session_state["active_nav"] = "rescue"
        st.rerun()

with top_n4:
    is_t4 = (st.session_state["active_nav"] == "precautions")
    btn4_t = "🛡️  Safety Guide & Survival Kit"
    if st.button(btn4_t, key="top_nav_precautions", type="primary" if is_t4 else "secondary", use_container_width=True):
        st.session_state["active_nav"] = "precautions"
        st.rerun()

st.markdown("<div style='margin-bottom: 14px;'></div>", unsafe_allow_html=True)


# ==============================================================================
# MAIN MODULE CONTENT
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. DAMAGE ASSESSMENT STUDIO MODULE
# ------------------------------------------------------------------------------
if st.session_state["active_nav"] == "classifier":
    st.markdown("### 🏢 Building Structural Safety & Damage Assessment")
    st.write(
        "Evaluate building stability by comparing high-resolution satellite imagery captured before and after a natural disaster. "
        "The analytical model measures structural disruption, produces impact heatmaps, and assigns an actionable safety rating to protect civilian lives and guide rescue teams."
    )

    c_opt, c_view = st.columns([1, 2.3], gap="medium")

    with c_opt:
        with st.container(border=True):
            st.markdown("##### 📋 Assessment Configuration")

            mode = st.radio(
                "Imagery Source:",
                ["Select a Real-World Disaster Case Study", "Upload Custom Satellite Imagery"],
            )

            pre_img = None
            post_img = None

            if mode == "Select a Real-World Disaster Case Study":
                example_options = [
                    "Intact Structure (Safe Baseline - No Damage)",
                    "🇮🇳 Wayanad Landslide - Total Structural Collapse (Kerala)",
                    "🇮🇳 Cyclone Dana - Major Roof & Cladding Breach (Odisha)",
                    "🇮🇳 Sikkim Teesta Flash Flood - Destroyed Foundation Scour",
                    "🇮🇳 Delhi Yamuna Flood - Submerged Perimeter (Civil Lines)",
                    "🇮🇳 Joshimath Subsidence - Major Structural Cracking (Uttarakhand)",
                    "🇮🇳 Himachal Cloudburst - Mountain Flood & Bridge Collapse (Mandi)",
                    "🇮🇳 Cyclone Michaung - Urban Deluge & Inundation (Chennai)",
                    "🇮🇳 Kutch Seismic Fault - Structural Masonry Fissures (Gujarat)",
                ]
                sample_pick = st.selectbox("Choose Case Study:", example_options)

                sample_map = {
                    "No Damage": "honshu",
                    "Wayanad": "wayanad",
                    "Cyclone Dana": "dana",
                    "Sikkim": "sikkim",
                    "Delhi": "delhi",
                    "Joshimath": "joshimath",
                    "Himachal": "himachal",
                    "Michaung": "michaung",
                    "Kutch": "kutch",
                }
                slug = "honshu"
                for key, val in sample_map.items():
                    if key in sample_pick:
                        slug = val
                        break

                slug_spec_map = {
                    "honshu": ("earthquake", 0),
                    "wayanad": ("flood", 3),
                    "dana": ("hurricane", 2),
                    "sikkim": ("flood", 3),
                    "delhi": ("flood", 1),
                    "joshimath": ("earthquake", 2),
                    "himachal": ("flood", 3),
                    "michaung": ("hurricane", 2),
                    "kutch": ("earthquake", 1),
                }
                d_type, sev = slug_spec_map.get(slug, ("earthquake", 0))

                actual_pre_path = Path("data/realtime_samples") / f"{slug}_pre.png"
                actual_post_path = Path("data/realtime_samples") / f"{slug}_post.png"

                if actual_pre_path.exists() and actual_post_path.exists():
                    pre_img = Image.open(actual_pre_path).convert("RGB")
                    post_img = Image.open(actual_post_path).convert("RGB")
                else:
                    pre_img, post_img, _ = generate_sample_crop_pair(disaster_type=d_type, severity=sev, crop_size=256)
                    pre_img.save(actual_pre_path)
                    post_img.save(actual_post_path)

            else:
                up_pre = st.file_uploader("Pre-Disaster Satellite Image (Before):", type=["png", "jpg", "jpeg"])
                up_post = st.file_uploader("Post-Disaster Satellite Image (After):", type=["png", "jpg", "jpeg"])

                if up_pre and up_post:
                    pre_img = Image.open(up_pre).convert("RGB")
                    post_img = Image.open(up_post).convert("RGB")
                else:
                    st.info("Upload both pre- and post-disaster satellite imagery to run structural assessment.")

            st.markdown("---")
            st.markdown("##### 🎨 Damage Heatmap Display")
            opacity = st.slider("Damage Overlay Intensity:", 0.0, 1.0, 0.55, 0.05)

    with c_view:
        if pre_img is not None and post_img is not None:
            # Predict with model
            pred_label, confidence, probs, heatmap = model.predict(pre_img, post_img)
            blended_heat = blend_heatmap_on_image(post_img, heatmap, alpha=opacity)

            # 3 Images Side-by-Side: Clean, Human-Friendly Cards
            i1, i2, i3 = st.columns(3)
            with i1:
                with st.container(border=True):
                    st.markdown("<div style='font-size: 0.82rem; font-weight: 700; color: #1e40af; text-transform: uppercase; margin-bottom: 4px;'>1. Before Disaster (Baseline)</div>", unsafe_allow_html=True)
                    st.image(pre_img, use_container_width=True)
                    st.caption("Intact building & roof footprint")

            with i2:
                with st.container(border=True):
                    st.markdown("<div style='font-size: 0.82rem; font-weight: 700; color: #1e40af; text-transform: uppercase; margin-bottom: 4px;'>2. After Disaster (Impact)</div>", unsafe_allow_html=True)
                    st.image(post_img, use_container_width=True)
                    st.caption("Post-event structural state")

            with i3:
                with st.container(border=True):
                    st.markdown("<div style='font-size: 0.82rem; font-weight: 700; color: #1e40af; text-transform: uppercase; margin-bottom: 4px;'>3. Damage Heatmap Overlay</div>", unsafe_allow_html=True)
                    st.image(blended_heat, use_container_width=True)
                    st.caption("Identified structural disruption zones")

            # Humanized Verdict Card (White & Blue Styling)
            hex_color = CLASS_HEX.get({"none": 0, "minor": 1, "major": 2, "destroyed": 3}.get(pred_label, 0), "#2563eb")
            badge_class = f"badge-{pred_label}"

            human_verdicts = {
                "none": "SAFE • NO STRUCTURAL DAMAGE",
                "minor": "CAUTION • MINOR SURFACE DAMAGE",
                "major": "RESTRICTED ENTRY • MAJOR STRUCTURAL DAMAGE",
                "destroyed": "CRITICAL HAZARD • BUILDING COLLAPSED",
            }
            verdict_text = human_verdicts.get(pred_label, pred_label.upper())

            st.markdown(f"""
            <div class="verdict-box" style="border-left-color: {hex_color};">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                    <div>
                        <div style="font-size: 0.8rem; color: #64748b; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px;">Building Safety Evaluation:</div>
                        <div style="margin-top: 6px;">
                            <span class="badge-pill {badge_class}">{verdict_text}</span>
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 0.8rem; color: #64748b; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px;">Assessment Confidence:</div>
                        <div style="font-size: 1.8rem; font-weight: 800; color: {hex_color}; font-family: 'JetBrains Mono', monospace; margin-top: 2px;">
                            {confidence * 100:.1f}%
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Direct Human Safety & Occupancy Guidance Banner
            if pred_label == "none":
                st.markdown("""
                <div class="action-banner action-safe">
                    <span style="font-size: 1.4rem;">✅</span>
                    <div>
                        <strong>Building Clear for Civilian Occupancy & Shelter:</strong><br>
                        The structural envelope, load-bearing walls, and roof ridgeline remain intact. Normal multi-temporal satellite lighting differences have been filtered out. The facility is certified safe for civilian shelter, staging, and relief distribution.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            elif pred_label == "minor":
                st.markdown("""
                <div class="action-banner action-warning">
                    <span style="font-size: 1.4rem;">⚠️</span>
                    <div>
                        <strong>Caution: Non-Structural Surface Disruption Detected:</strong><br>
                        Local roof tile loss, cladding displacement, or perimeter debris identified. The primary concrete/steel framing remains stable. Keep a 10-meter clearance from eaves to avoid falling tiles. Cleared for secondary relief staging with standard hardhat precautions.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            elif pred_label == "major":
                st.markdown("""
                <div class="action-banner action-danger">
                    <span style="font-size: 1.4rem;">🚨</span>
                    <div>
                        <strong>Restricted Entry: Structural Integrity Severely Compromised:</strong><br>
                        Significant roof cavity breach, wall displacement, or foundation inundation observed. Unsafe for human habitation. Enforce an immediate 50-meter perimeter cordon. Shoring and stabilization required before any rescue personnel enter.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="action-banner action-danger" style="background: #fef2f2; border-color: #f87171;">
                    <span style="font-size: 1.4rem;">💥</span>
                    <div>
                        <strong>Critical Emergency: Complete Structural Failure / Collapse:</strong><br>
                        Total building collapse into rubble mound with structural displacement. High danger of secondary collapse. Task specialized Heavy Urban Search & Rescue (USAR) teams with acoustic listening probes, hydraulic cutters, and search dogs immediately.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Severity Probabilities (White & Blue Cards)
            st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
            st.markdown("##### 📊 Damage Severity Assessment Breakdown:")
            p_cols = st.columns(4)
            labels = ["Intact (Safe)", "Minor Damage", "Major Damage", "Total Collapse"]
            for i, name in enumerate(labels):
                with p_cols[i]:
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-lbl">{name}</div>
                        <div class="kpi-val" style="color: {CLASS_HEX[i]};">{probs[i]*100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(float(probs[i]))

            # On-Site Civil Engineering Inspection Checklist
            with st.expander("📝 On-Site Field Engineering Checklist (For Structural Inspectors)", expanded=False):
                st.caption("Perform these ground checks before allowing civilian re-entry:")
                st.checkbox("1. Confirm no active natural gas leaks or severed high-voltage power lines nearby.", value=(pred_label == "none"))
                st.checkbox("2. Inspect exterior perimeter for 45-degree diagonal masonry shear cracks (X-cracks).", value=False)
                st.checkbox("3. Inspect foundation corners for differential settlement, sinkholes, or soil liquefaction.", value=(pred_label in ["none", "minor"]))
                st.checkbox("4. Verify interior ceiling framing and water pipes for overhead collapse risks.", value=False)
                st.checkbox("5. Check that emergency exits and evacuation paths are completely clear of fallen masonry.", value=(pred_label == "none"))

            # Verification Report & Dossier Export
            st.markdown("---")
            st.markdown("##### 📑 Official Inspection Dossier & Audit Report Generation")
            st.write(
                "Export an official technical verification dossier containing embedded pre/post satellite imagery, "
                "damage heatmaps, probability breakdowns, civilian safety ratings, and engineering sign-off fields."
            )

            is_india_case = ("🇮🇳" in sample_pick) if mode == "Select a Real-World Disaster Case Study" else False
            default_clf_agency = "National Disaster Response Force (NDRF) / NDMA Directorate" if is_india_case else "Civil Protection & Structural Safety Directorate"
            rep_col1, rep_col2, rep_col3 = st.columns([1.5, 1.5, 1.2])
            with rep_col1:
                assessor_name = st.text_input(
                    "Lead Technical Assessor:",
                    value="Eng. Ankit Kumar Shandilya",
                    key="classifier_assessor",
                )
            with rep_col2:
                agency_name = st.text_input(
                    "Directorate / Department:",
                    value=default_clf_agency,
                    key="classifier_agency",
                )
            with rep_col3:
                now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                loc_seed = sample_pick if mode == "Select a Real-World Disaster Case Study" else "Custom Upload"
                dossier_id = generate_dossier_id(loc_seed, now_str)
                st.markdown(f"<div style='padding-top: 28px;'><span class='badge-pill' style='background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; font-family: monospace; font-size: 0.85rem;'>DOSSIER #{dossier_id}</span></div>", unsafe_allow_html=True)

            loc_label = sample_pick if mode == "Select a Real-World Disaster Case Study" else "Custom Orthorectified AOI"
            prob_dict = {
                "None (No Damage)": float(probs[0]),
                "Minor Damage": float(probs[1]),
                "Major Damage": float(probs[2]),
                "Destroyed": float(probs[3]),
            }

            if pred_label == "none":
                directive_text = "Structural envelope intact. Certified structurally sound for civilian sheltering and staging operations. Regular vigilance advised."
            elif pred_label == "minor":
                directive_text = "Superficial cladding/roofing displacement. Maintain 10m perimeter clearance from roof edges. Primary frame structurally sound."
            elif pred_label == "major":
                directive_text = "Severe structural breach or partial roof collapse. Evacuate 50m perimeter immediately. Structural shoring required prior to interior entry."
            else:
                directive_text = "Total structural collapse detected. Level IV Heavy USAR deployment required. Deploy victim-locating cameras and hydraulic cutters."

            report_data = {
                "dossier_id": dossier_id,
                "timestamp": now_str,
                "location_name": loc_label,
                "coordinates": "Global Satellite Coordinate AOI",
                "sensor": "DigitalGlobe WorldView-3 / Sentinel-2 MSI (Sub-Meter Orthorectified)",
                "severity_label": pred_label.upper(),
                "confidence": float(confidence),
                "probabilities": prob_dict,
                "directive": directive_text,
                "assessor_name": assessor_name,
                "agency": agency_name,
                "triage_level": "RED" if pred_label in ["major", "destroyed"] else "YELLOW" if pred_label == "minor" else "GREEN",
            }

            docx_bytes = generate_verification_report_docx(report_data, pre_img, post_img, blended_heat)
            html_str = generate_verification_report_html(report_data, pre_img, post_img, blended_heat)
            json_str = generate_verification_report_json(report_data)

            btn_col1, btn_col2, btn_col3 = st.columns(3)
            with btn_col1:
                st.download_button(
                    label="📄 Download Inspection Dossier (.docx)",
                    data=docx_bytes,
                    file_name=f"Damage_Dossier_{dossier_id}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    key="dl_classifier_docx",
                    use_container_width=True,
                )
            with btn_col2:
                st.download_button(
                    label="🖨️ Download Print-Ready Report (.html)",
                    data=html_str,
                    file_name=f"Damage_Report_{dossier_id}.html",
                    mime="text/html",
                    key="dl_classifier_html",
                    use_container_width=True,
                )
            with btn_col3:
                st.download_button(
                    label="💾 Download Audit Verification Data (.json)",
                    data=json_str,
                    file_name=f"Audit_Telemetry_{dossier_id}.json",
                    mime="application/json",
                    key="dl_classifier_json",
                    use_container_width=True,
                )

            with st.expander("🔍 Preview Assessment Dossier (Interactive View)", expanded=False):
                st.components.v1.html(html_str, height=650, scrolling=True)


# ------------------------------------------------------------------------------
# 2. INDIAN NATIONAL DISASTER WATCH & SATELLITE INGESTION MODULE
# ------------------------------------------------------------------------------
elif st.session_state["active_nav"] == "realtime":
    st.markdown("### 🛰️ Indian National Disaster Live Incident Monitor")
    st.write(
        "Monitors active natural disaster alerts across India in real time from the **National Disaster Management Authority (NDMA)**, "
        "**National Center for Seismology (NCS / MoES)**, and the **India Meteorological Department (IMD)**. "
        "Select an active incident to inspect satellite passes, analyze structural damage, and review NDRF operational directives."
    )

    # Ingest Indian National Disasters (NCS / NDMA / IMD)
    events, source_label = get_india_disaster_catalog(use_live_network=True)

    # Status Bar & Refresh
    s_col1, s_col2 = st.columns([3, 1])
    with s_col1:
        st.markdown(f"<div style='color: #1d4ed8; font-size: 0.92rem; font-weight: 600; padding: 6px 0;'>{source_label} | Tracking {len(events)} Indian Disaster Zones</div>", unsafe_allow_html=True)
    with s_col2:
        if st.button("🔄 Refresh Live Feeds", use_container_width=True):
            st.rerun()

    c_event, c_sat = st.columns([1, 2.3], gap="medium")

    with c_event:
        with st.container(border=True):
            st.markdown("##### 🔴 Active Indian Incident Monitor")

            # Selection dropdown
            event_titles = [f"[{e['alert_level']}] {e['title']}" for e in events]
            chosen_idx = st.selectbox(
                "Choose Disaster Event to Inspect:",
                range(len(events)),
                format_func=lambda i: event_titles[i],
            )
            ev = events[chosen_idx]

            alert_color = "#dc2626" if ev["alert_level"] == "RED" else "#d97706" if ev["alert_level"] == "ORANGE" else "#059669"

            state_badge = f"<div style='font-size: 0.84rem; color: #334155; margin-bottom: 4px;'><strong>State / Territory:</strong> {ev['state']}</div>" if ev.get("state") else ""
            ndrf_badge = f"<div style='font-size: 0.84rem; color: #1d4ed8; margin-top: 4px; margin-bottom: 4px;'><strong>🇮🇳 Assigned Unit:</strong> {ev['ndrf_battalion']}</div>" if ev.get("ndrf_battalion") else ""

            st.markdown(f"""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px; margin-top: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.04);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-size: 0.75rem; color: #64748b; text-transform: uppercase; font-weight: 700;">Category:</span>
                    <span style="background: {alert_color}; color: white; padding: 3px 12px; border-radius: 14px; font-size: 0.75rem; font-weight: 700;">{ev['alert_level']} ALERT</span>
                </div>
                <div style="font-size: 1.05rem; font-weight: 700; color: #0f172a; margin-bottom: 6px;">{ev['category']}</div>
                <div style="font-size: 0.84rem; color: #334155; margin-bottom: 4px;"><strong>Intensity:</strong> {ev['magnitude']}</div>
                <div style="font-size: 0.84rem; color: #334155; margin-bottom: 4px;"><strong>Region:</strong> {ev['place']}</div>
                {state_badge}
                {ndrf_badge}
                <div style="font-size: 0.84rem; color: #334155; margin-bottom: 4px;"><strong>Coordinates:</strong> {ev['coordinates'][0]}°N, {ev['coordinates'][1]}°E</div>
                <div style="font-size: 0.84rem; color: #334155; margin-bottom: 4px;"><strong>Recorded:</strong> {ev['time']}</div>
                <div style="font-size: 0.84rem; color: #2563eb; margin-top: 6px;"><strong>Satellite AOI:</strong> {ev['satellite_aoi']}</div>
                <div style="font-size: 0.8rem; color: #64748b; margin-top: 8px; border-top: 1px solid #e2e8f0; padding-top: 6px;">
                    {ev['description']}
                </div>
            </div>
            <a href='https://www.openstreetmap.org/?mlat={ev['coordinates'][0]}&mlon={ev['coordinates'][1]}#map=16/{ev['coordinates'][0]}/{ev['coordinates'][1]}' target='_blank' style='display:block; text-align:center; margin-top:10px; font-size:0.82rem; color:#2563eb; font-weight:600; text-decoration:none;'>🗺️ View Epicenter on OpenStreetMap ↗</a>
            """, unsafe_allow_html=True)

            st.markdown("---")
            feed_mode = st.radio(
                "Satellite View Mode:",
                ["Standard Orthorectified Satellite Imagery", "Live Orbital Tile Stream (ESRI / Sentinel-2)"],
                index=0,
                key="rt_stream_choice"
            )
            attempt_live_stream = (feed_mode == "Live Orbital Tile Stream (ESRI / Sentinel-2)")

            with st.expander("📍 Custom Coordinates Explorer"):
                st.caption("Fetch satellite tiles for any custom coordinate or landmark:")
                loc_presets = {
                    "Use Current Disaster Event": None,
                    "LPU Campus, Phagwara, Punjab (31.25°N, 75.70°E)": (31.253, 75.703),
                    "Ishikawa Noto, Japan (37.50°N, 137.24°E)": (37.498, 137.242),
                    "Kahramanmaraş, Turkey (37.17°N, 37.03°E)": (37.174, 37.032),
                    "Hualien, Taiwan (23.82°N, 121.56°E)": (23.819, 121.562),
                    "Big Bend, Florida (29.85°N, -83.55°E)": (29.850, -83.550),
                    "San Francisco Bay, USA (37.77°N, -122.42°E)": (37.774, -122.419),
                    "Dubai Marina, UAE (25.08°N, 55.14°E)": (25.081, 55.141),
                    "Manhattan, New York (40.76°N, -73.98°E)": (40.758, -73.985),
                }
                preset_name = st.selectbox("Landmark Presets:", list(loc_presets.keys()), key="preset_loc_picker")
                if preset_name and loc_presets[preset_name]:
                    p_lat, p_lon = loc_presets[preset_name]
                else:
                    p_lat, p_lon = float(ev["coordinates"][0]), float(ev["coordinates"][1])

                c_lat = st.number_input("Latitude (°N):", -90.0, 90.0, p_lat, format="%.4f", key="inp_custom_lat")
                c_lon = st.number_input("Longitude (°E):", -180.0, 180.0, p_lon, format="%.4f", key="inp_custom_lon")
                c_sev = st.select_slider("Damage Severity Level:", options=[0, 1, 2, 3], value=int(ev.get("simulated_severity", 2)), format_func=lambda s: {0:"None", 1:"Minor", 2:"Major", 3:"Destroyed"}[s], key="inp_custom_sev")
                
                if st.button("🛰️ Inspect Coordinates", use_container_width=True, key="btn_run_custom"):
                    st.session_state["custom_override_event"] = {
                        "id": f"custom_{c_lat}_{c_lon}",
                        "title": f"Custom Orbital Satellite AOI [{c_lat:.3f}°N, {c_lon:.3f}°E]",
                        "category": ev["category"],
                        "disaster_type": ev["disaster_type"],
                        "source": "ESRI World Imagery Global High-Res Optical",
                        "magnitude": "Custom AOI Observation",
                        "place": preset_name if preset_name and loc_presets[preset_name] else f"Custom Lat {c_lat:.2f}, Lon {c_lon:.2f}",
                        "coordinates": [c_lat, c_lon],
                        "depth_km": None,
                        "time": "Real-Time Orbital Acquisition",
                        "alert_level": "RED" if c_sev >= 2 else "YELLOW" if c_sev == 1 else "GREEN",
                        "description": "Custom high-resolution optical satellite tile ingested from global Earth Observation server.",
                        "simulated_severity": c_sev,
                        "satellite_aoi": f"ESRI World Imagery [{c_lat:.2f}°N, {c_lon:.2f}°E]",
                        "sample_slug": "custom"
                    }
                    st.rerun()

            st.markdown("---")
            st.markdown("##### 🎨 Damage Heatmap Display")
            rt_opacity = st.slider("Damage Overlay Intensity:", 0.0, 1.0, 0.60, 0.05, key="rt_slider")

    with c_sat:
        # Check if custom override active
        if "custom_override_event" in st.session_state and st.session_state["custom_override_event"]:
            ev = st.session_state["custom_override_event"]
            if st.button("↩️ Reset to Live Incident Monitor", key="btn_reset_custom"):
                st.session_state["custom_override_event"] = None
                st.rerun()

        # Load satellite imagery for event
        pre_img, post_img, meta = get_satellite_crops_for_event(ev, attempt_live_tile=attempt_live_stream)

        # Run AI prediction
        pred_label, confidence, probs, heatmap = model.predict(pre_img, post_img)
        blended_heat = blend_heatmap_on_image(post_img, heatmap, alpha=rt_opacity)

        # Live Satellite Telemetry Dock (White & Blue)
        is_live = meta.get("is_live_tile", False)
        live_badge = "🟢 LIVE ORBITAL PASS" if is_live else "🛰️ VERIFIED HIGH-RES ARCHIVE"
        badge_bg = "#ecfdf5" if is_live else "#eff6ff"
        badge_color = "#047857" if is_live else "#1d4ed8"
        badge_border = "#a7f3d0" if is_live else "#bfdbfe"

        st.markdown(f"""
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 12px 18px; margin-bottom: 14px; box-shadow: 0 2px 8px rgba(0,0,0,0.04);">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="background: {badge_bg}; color: {badge_color}; border: 1px solid {badge_border}; border-radius: 14px; padding: 3px 12px; font-size: 0.74rem; font-weight: 700;">{live_badge}</span>
                    <span style="font-size: 0.85rem; color: #0f172a; font-weight: 700;">{meta.get('source', 'Orbital Satellite Pass')}</span>
                </div>
                <div style="font-size: 0.8rem; color: #64748b; font-family: 'JetBrains Mono', monospace;">
                    📍 {meta.get('coordinates', f"{ev['coordinates'][0]}°N, {ev['coordinates'][1]}°E")} | GSD: {meta.get('gsd', '0.3m/px')}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 3 Images Side-by-Side
        si1, si2, si3 = st.columns(3)
        with si1:
            with st.container(border=True):
                st.markdown("<div style='font-size: 0.82rem; font-weight: 700; color: #1e40af; text-transform: uppercase; margin-bottom: 4px;'>1. Baseline (Before Event)</div>", unsafe_allow_html=True)
                st.image(pre_img, use_container_width=True)
                st.caption(f"Baseline capture | {meta.get('tile_info', 'Sentinel-2 AOI')}")

        with si2:
            with st.container(border=True):
                st.markdown("<div style='font-size: 0.82rem; font-weight: 700; color: #1e40af; text-transform: uppercase; margin-bottom: 4px;'>2. Impact (After Event)</div>", unsafe_allow_html=True)
                st.image(post_img, use_container_width=True)
                st.caption(f"Post-event satellite pass | {ev['magnitude']}")

        with si3:
            with st.container(border=True):
                st.markdown("<div style='font-size: 0.82rem; font-weight: 700; color: #1e40af; text-transform: uppercase; margin-bottom: 4px;'>3. Impact Heatmap</div>", unsafe_allow_html=True)
                st.image(blended_heat, use_container_width=True)
                st.caption(f"Structural disruption focus | {confidence*100:.1f}%")

        # Severity Summary Card (White & Blue)
        hex_color = CLASS_HEX.get({"none": 0, "minor": 1, "major": 2, "destroyed": 3}.get(pred_label, 0), "#2563eb")
        badge_class = f"badge-{pred_label}"

        human_verdicts = {
            "none": "SAFE • NO STRUCTURAL DAMAGE",
            "minor": "CAUTION • MINOR SURFACE DAMAGE",
            "major": "RESTRICTED ENTRY • MAJOR STRUCTURAL DAMAGE",
            "destroyed": "CRITICAL HAZARD • BUILDING COLLAPSED",
        }
        rt_verdict = human_verdicts.get(pred_label, pred_label.upper())

        st.markdown(f"""
        <div class="verdict-box" style="border-left-color: {hex_color};">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>
                    <span style="font-size: 0.8rem; color: #64748b; text-transform: uppercase; font-weight: 700;">Structural Safety Verdict:</span>
                    <div style="margin-top: 6px;">
                        <span class="badge-pill {badge_class}">{rt_verdict}</span>
                    </div>
                </div>
                <div style="text-align: right;">
                    <span style="font-size: 0.8rem; color: #64748b; text-transform: uppercase; font-weight: 700;">Confidence:</span>
                    <div style="font-size: 1.8rem; font-weight: 800; color: {hex_color}; font-family: 'JetBrains Mono', monospace; margin-top: 2px;">
                        {confidence * 100:.1f}%
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 4 Probabilities (White & Blue Cards)
        st.markdown("##### 📊 Damage Severity Assessment Breakdown:")
        p_cols = st.columns(4)
        labels = ["Intact (Safe)", "Minor Damage", "Major Damage", "Total Collapse"]
        for i, name in enumerate(labels):
            with p_cols[i]:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-lbl">{name}</div>
                    <div class="kpi-val" style="color: {CLASS_HEX[i]};">{probs[i]*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(float(probs[i]))

        # First-Responder Directives
        st.markdown("##### 🚨 Official National Disaster Response Force (NDRF / SDRF) Mandate:")
        rt_directive = get_ndrf_operational_directive(pred_label, ev)
        if pred_label in ["destroyed", "major"]:
            st.error(rt_directive)
        elif pred_label == "minor":
            st.warning(rt_directive)
        else:
            st.success(rt_directive)

        # Pipeline Architecture Card (White & Blue)
        st.markdown("""
        <div style="background: #eff6ff; border: 1px dashed #93c5fd; border-radius: 12px; padding: 14px 18px; margin-top: 14px;">
            <span style="color: #1e40af; font-size: 0.82rem; font-weight: 700; text-transform: uppercase;">📡 National Disaster Telemetry Pipeline:</span>
            <div style="color: #334155; font-size: 0.84rem; margin-top: 4px; line-height: 1.6;">
                <code>NDMA Sachet / IMD Early Warning</code> &rarr; <code>ISRO Cartosat / Sentinel-2 Tasking</code> &rarr; <code>Sub-Pixel Coregistration</code> &rarr; <code>Dual-Branch Siamese CNN</code> &rarr; <code>NDRF Battalion Deployment</code>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Verification Report & Dossier Export for Real-Time Disaster Feed
        st.markdown("---")
        st.markdown("##### 📑 Indian National Disaster Verification Dossier & Audit Report")
        st.write(
            "Export an authenticated civil defense verification dossier containing live event telemetry, "
            "satellite imagery, damage heatmap evidence, and automated rescue directives."
        )

        r_col1, r_col2, r_col3 = st.columns([1.5, 1.5, 1.2])
        with r_col1:
            rt_assessor = st.text_input(
                "Command Assessor:",
                value="Eng. Ankit Kumar Shandilya",
                key="rt_assessor",
            )
        with r_col2:
            rt_agency = st.text_input(
                "Field Unit / Agency:",
                value="National Disaster Response Force (NDRF) / NDMA Directorate",
                key="rt_agency",
            )
        with r_col3:
            rt_now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            rt_dossier_id = generate_dossier_id(ev.get("title", "Disaster AOI"), rt_now_str)
            st.markdown(f"<div style='padding-top: 28px;'><span class='badge-pill' style='background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; font-family: monospace; font-size: 0.85rem;'>DOSSIER #{rt_dossier_id}</span></div>", unsafe_allow_html=True)

        rt_prob_dict = {
            "None (No Damage)": float(probs[0]),
            "Minor Damage": float(probs[1]),
            "Major Damage": float(probs[2]),
            "Destroyed": float(probs[3]),
        }

        rt_report_data = {
            "dossier_id": rt_dossier_id,
            "timestamp": rt_now_str,
            "location_name": f"{ev.get('title', 'Disaster AOI')} ({ev.get('category', 'Disaster')})",
            "coordinates": meta.get("coordinates", f"{ev.get('coordinates', [0, 0])[0]}°N, {ev.get('coordinates', [0, 0])[1]}°E"),
            "sensor": meta.get("source", "ISRO Cartosat-3 / RISAT-1A / Sentinel-2 MSI"),
            "severity_label": pred_label.upper(),
            "confidence": float(confidence),
            "probabilities": rt_prob_dict,
            "directive": rt_directive,
            "assessor_name": rt_assessor,
            "agency": rt_agency,
            "triage_level": ev.get("alert_level", "RED"),
        }

        rt_docx_bytes = generate_verification_report_docx(rt_report_data, pre_img, post_img, blended_heat)
        rt_html_str = generate_verification_report_html(rt_report_data, pre_img, post_img, blended_heat)
        rt_json_str = generate_verification_report_json(rt_report_data)

        b1, b2, b3 = st.columns(3)
        with b1:
            st.download_button(
                label="📄 Download Inspection Dossier (.docx)",
                data=rt_docx_bytes,
                file_name=f"India_Disaster_Dossier_{rt_dossier_id}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key="dl_rt_docx",
                use_container_width=True,
            )
        with b2:
            st.download_button(
                label="🖨️ Download Print-Ready Report (.html)",
                data=rt_html_str,
                file_name=f"India_Disaster_Report_{rt_dossier_id}.html",
                mime="text/html",
                key="dl_rt_html",
                use_container_width=True,
            )
        with b3:
            st.download_button(
                label="💾 Download Audit Verification Data (.json)",
                data=rt_json_str,
                file_name=f"India_Telemetry_{rt_dossier_id}.json",
                mime="application/json",
                key="dl_rt_json",
                use_container_width=True,
            )

        with st.expander("🔍 Preview Assessment Dossier (Interactive View)", expanded=False):
            st.components.v1.html(rt_html_str, height=650, scrolling=True)


# ------------------------------------------------------------------------------
# 3. HOW TO SAVE SOMEONE (LIFE-SAVING PROTOCOLS)
# ------------------------------------------------------------------------------
elif st.session_state["active_nav"] == "rescue":
    st.markdown("### 🚑 Life-Saving Rescue Protocols & Emergency Triage")
    st.write("Essential life-saving rules, first aid measures, and victim triage protocols when responding to structural collapse emergencies.")

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.markdown("""
        <div class="human-card" style="height: 100%;">
            <div style="font-size: 1.8rem; margin-bottom: 6px;">🛡️</div>
            <h5 style="color: #1e40af; margin-top: 0;">1. Ensure Safety First</h5>
            <p style="font-size: 0.88rem; color: #334155; line-height: 1.6;">
            Never enter an unstable building alone. Watch out for arcing electrical lines, leaking gas, and hanging concrete slabs.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with r2:
        st.markdown("""
        <div class="human-card" style="height: 100%;">
            <div style="font-size: 1.8rem; margin-bottom: 6px;">👂</div>
            <h5 style="color: #1e40af; margin-top: 0;">2. Acoustic Search</h5>
            <p style="font-size: 0.88rem; color: #334155; line-height: 1.6;">
            Call out loudly and listen in complete silence. Instruct trapped victims to tap three times on metallic pipes or solid concrete.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with r3:
        st.markdown("""
        <div class="human-card" style="height: 100%;">
            <div style="font-size: 1.8rem; margin-bottom: 6px;">🩹</div>
            <h5 style="color: #1e40af; margin-top: 0;">3. Stop Bleeding</h5>
            <p style="font-size: 0.88rem; color: #334155; line-height: 1.6;">
            Apply firm direct pressure with clean dressing. For severe limb arterial bleeding, apply a tourniquet 2-3 inches above the wound.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with r4:
        st.markdown("""
        <div class="human-card" style="height: 100%;">
            <div style="font-size: 1.8rem; margin-bottom: 6px;">🌡️</div>
            <h5 style="color: #1e40af; margin-top: 0;">4. Prevent Shock</h5>
            <p style="font-size: 0.88rem; color: #334155; line-height: 1.6;">
            Hypothermia and shock set in quickly. Wrap the victim in an emergency thermal blanket, reassure them, and elevate feet 12 inches.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Critical Medical Warning on Crush Syndrome
    st.markdown("""
    <div class="human-card" style="border-left: 6px solid #dc2626; background: #fef2f2; border-color: #fecaca;">
        <h5 style="color: #b91c1c; margin-top: 0;">⚠️ Mandatory Medical Warning: Crush Syndrome</h5>
        <p style="color: #7f1d1d; font-size: 0.92rem; line-height: 1.6; margin-bottom: 0;">
        If a survivor's limb has been trapped beneath a heavy concrete slab for <strong>more than 2 hours</strong>, 
        <strong>DO NOT suddenly lift the weight off</strong> without medical personnel prepared with IV hydration! 
        Sudden release flushes toxic potassium and myoglobin accumulated in the crushed muscle directly into circulation, causing 
        <strong>fatal cardiac arrest or acute kidney failure</strong> within minutes (reperfusion shock). Always administer IV saline before freeing the limb.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 4-Color Triage Guide
    st.markdown("##### 🚦 Emergency Triage Priority Guide (Who to Help First):")
    t1, t2, t3, t4 = st.columns(4)
    with t1:
        st.markdown("""
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-left: 5px solid #dc2626; padding: 14px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.03);">
            <strong style="color: #b91c1c;">🔴 Immediate (Priority 1)</strong><br>
            <span style="font-size: 0.85rem; color: #475569;">Airway obstruction, tension pneumothorax, or massive bleeding. Treat first!</span>
        </div>
        """, unsafe_allow_html=True)
    with t2:
        st.markdown("""
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-left: 5px solid #d97706; padding: 14px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.03);">
            <strong style="color: #b45309;">🟡 Delayed (Priority 2)</strong><br>
            <span style="font-size: 0.85rem; color: #475569;">Fractures, deep lacerations, stable vitals. Needs medical care after Priority 1.</span>
        </div>
        """, unsafe_allow_html=True)
    with t3:
        st.markdown("""
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-left: 5px solid #059669; padding: 14px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.03);">
            <strong style="color: #047857;">🟢 Minor (Walking Wounded)</strong><br>
            <span style="font-size: 0.85rem; color: #475569;">Minor cuts, abrasions, sprains. Able to walk and assist with minor tasks.</span>
        </div>
        """, unsafe_allow_html=True)
    with t4:
        st.markdown("""
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-left: 5px solid #64748b; padding: 14px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.03);">
            <strong style="color: #475569;">⚫ Expectant / Deceased</strong><br>
            <span style="font-size: 0.85rem; color: #475569;">No pulse or respiration after opening airway. Direct resources to viable victims.</span>
        </div>
        """, unsafe_allow_html=True)

    # Official Indian National Emergency Helplines (White & Blue)
    st.markdown("""
    <div style="background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 12px; padding: 16px 22px; margin-top: 22px;">
        <span style="color: #1e40af; font-size: 0.88rem; font-weight: 700; text-transform: uppercase;">🇮🇳 Official National Emergency Helplines (India):</span>
        <div style="display: flex; gap: 24px; flex-wrap: wrap; margin-top: 10px; font-size: 0.88rem; color: #1e293b;">
            <span>🚨 <strong>112</strong>: Pan-India Emergency All-in-One</span>
            <span>🏛️ <strong>1078</strong>: NDMA National Disaster Helpline</span>
            <span>🚑 <strong>108</strong>: Emergency Ambulance Service</span>
            <span>🌊 <strong>1070</strong>: State Relief Commissioner (SDRF)</span>
            <span>🚒 <strong>101</strong>: Fire & Heavy Rescue Service</span>
            <span>📞 <strong>011-24363260</strong>: NDRF HQ Control Room</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# 4. SAFETY PRECAUTIONS & 72-HOUR SURVIVAL KIT MODULE
# ------------------------------------------------------------------------------
elif st.session_state["active_nav"] == "precautions":
    st.markdown("### 🛡️ Safety Precautions & 72-Hour Survival Kit")
    st.write("Comprehensive safety protocols across all three disaster phases, and an interactive 72-hour survival kit readiness auditor.")

    stage = st.radio(
        "Disaster Phase:",
        ["Before Disaster (Preparation)", "During Disaster (Survival)", "After Disaster (Recovery)"],
        horizontal=True,
    )

    if stage == "Before Disaster (Preparation)":
        st.markdown("""
        <div class="human-card">
            <h5 style="color: #1e40af;">📦 Essential Precautions Before Disaster Strikes:</h5>
            <ul style="color: #334155; line-height: 1.9; font-size: 0.92rem;">
                <li><strong>Anchor Heavy Furniture:</strong> Secure tall bookcases, water heaters, and heavy appliances to wall studs to prevent them from toppling during seismic tremors.</li>
                <li><strong>Assemble 72-Hour Survival Go-Bag:</strong> Pack 4 liters of potable water per person per day, non-perishable canned food, medical supplies, and copies of important documents.</li>
                <li><strong>Establish Family Evacuation Plan:</strong> Designate two independent meeting spots and identify an out-of-state emergency family contact.</li>
                <li><strong>Structural Hurricane Strapping:</strong> In cyclonic and storm regions, secure roof trusses to load-bearing walls using galvanized steel hurricane ties.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    elif stage == "During Disaster (Survival)":
        st.markdown("""
        <div class="human-card">
            <h5 style="color: #1e40af;">🚨 Critical Survival Actions During Disaster:</h5>
            <ul style="color: #334155; line-height: 1.9; font-size: 0.92rem;">
                <li><strong>Earthquake:</strong> <strong>Drop, Cover, and Hold On!</strong> Drop to your hands and knees under a sturdy desk. Never run outside or stand in doorframes.</li>
                <li><strong>Flood:</strong> <strong>Turn Around, Don't Drown!</strong> Just 12 inches of rushing water can sweep away passenger cars. Never walk or drive through flooded roadways.</li>
                <li><strong>Cyclone / Hurricane:</strong> Shelter in an interior windowless room (hallway, closet, or bathroom) on the lowest floor. Stay away from windows.</li>
                <li><strong>Wildfire:</strong> Evacuate immediately when ordered. Wear 100% cotton clothing and an N95 respirator mask to protect lungs from hot embers and particulate smoke.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="human-card">
            <h5 style="color: #1e40af;">⚠️ Safety Precautions in the Disaster Aftermath:</h5>
            <ul style="color: #334155; line-height: 1.9; font-size: 0.92rem;">
                <li><strong>Never Flip Electrical Switches if Gas is Smelled:</strong> Sparks can ignite pocketed gas. Open windows and turn off the main gas valve from the outside immediately.</li>
                <li><strong>Maintain 35-Foot Distance from Downed Lines:</strong> Treat all fallen electrical lines as energized. Never touch standing water near downed poles.</li>
                <li><strong>Inspect for X-Cracks:</strong> Do not enter masonry structures displaying diagonal 45-degree shear cracks until certified safe by a civil engineer.</li>
                <li><strong>Boil Tap Water:</strong> Pipe breaks contaminate tap water with groundwater and sewage. Boil vigorously for at least 3 minutes before drinking.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### 🎒 Interactive 72-Hour Survival Kit Auditor:")
    st.write("Check off the emergency supplies you currently have equipped in your home or go-bag:")

    chk_cols = st.columns(3)
    checked_count = 0
    total_items = len(EMERGENCY_SUPPLIES_CHECKLIST)

    for i, item in enumerate(EMERGENCY_SUPPLIES_CHECKLIST):
        with chk_cols[i % 3]:
            if st.checkbox(f"{item['item']}", key=f"chk_audit_{i}"):
                checked_count += 1

    pct = int((checked_count / total_items) * 100)
    ready_color = "#059669" if pct >= 80 else "#d97706" if pct >= 40 else "#dc2626"
    st.markdown(f"<div style='font-size: 1rem; font-weight: 700; color: {ready_color}; margin-top: 10px;'>Go-Bag Preparedness Score: {pct}% ({checked_count}/{total_items} items equipped)</div>", unsafe_allow_html=True)
    st.progress(checked_count / total_items)


# ==============================================================================
# BOTTOM DOCK: QUICK MODULE SELECTOR (White & Blue Theme)
# ==============================================================================
st.markdown("<div style='margin-top: 36px; margin-bottom: 14px; text-align: center; font-size: 0.82rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; color: #64748b;'>🧭 QUICK MODULE SELECTOR:</div>", unsafe_allow_html=True)

col_b1, col_b2, col_b3, col_b4 = st.columns(4, gap="small")

with col_b1:
    is_active1 = (st.session_state["active_nav"] == "classifier")
    b1_text = "🏢  Damage Assessment Studio"
    if st.button(b1_text, key="bottom_btn_classifier", type="primary" if is_active1 else "secondary", use_container_width=True):
        st.session_state["active_nav"] = "classifier"
        st.rerun()

with col_b2:
    is_active2 = (st.session_state["active_nav"] == "realtime")
    b2_text = "🛰️  India Disaster Live Monitor"
    if st.button(b2_text, key="bottom_btn_realtime", type="primary" if is_active2 else "secondary", use_container_width=True):
        st.session_state["active_nav"] = "realtime"
        st.rerun()

with col_b3:
    is_active3 = (st.session_state["active_nav"] == "rescue")
    b3_text = "🚑  Life-Saving Rescue Guide"
    if st.button(b3_text, key="bottom_btn_rescue", type="primary" if is_active3 else "secondary", use_container_width=True):
        st.session_state["active_nav"] = "rescue"
        st.rerun()

with col_b4:
    is_active4 = (st.session_state["active_nav"] == "precautions")
    b4_text = "🛡️  Safety & Survival Kit"
    if st.button(b4_text, key="bottom_btn_precautions", type="primary" if is_active4 else "secondary", use_container_width=True):
        st.session_state["active_nav"] = "precautions"
        st.rerun()

# Professional Human-Centered Footer (White & Blue)
st.markdown("<br><hr style='border-color: #e2e8f0;'>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.86rem;'>"
    "DisasterVision • National Emergency Response Infrastructure | Final Year Computer Vision Project"
    "</div>",
    unsafe_allow_html=True,
)
