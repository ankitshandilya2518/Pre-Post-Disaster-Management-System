"""
Pre/Post-Disaster Building-Damage Classifier & Safety Guide
============================================================
CSE471 Final Year Project | Computer Vision
Dual-Branch Model for xBD Building Damage Severity Classification
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
    page_title="Building Damage Classifier | CSE471",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom Styling with Bottom Row Navigation Cards & Rich Hover Effects
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">

<style>
    * {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    .mono {
        font-family: 'JetBrains Mono', monospace !important;
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
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        max-width: 1400px !important;
    }

    /* Container styling */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px !important;
        border-color: rgba(255, 255, 255, 0.08) !important;
        background: #0d1527 !important;
        padding: 14px !important;
    }

    /* Top Brand Header */
    .app-header {
        margin-bottom: 16px;
        padding-bottom: 12px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
    }
    .app-title {
        font-size: 1.85rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa 0%, #38bdf8 50%, #f43f5e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2px;
    }
    .app-subtitle {
        font-size: 0.88rem;
        color: #94a3b8;
    }
    .status-badge {
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #34d399;
        font-size: 0.78rem;
        font-weight: 700;
        padding: 5px 14px;
        border-radius: 20px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .status-dot {
        width: 8px;
        height: 8px;
        background: #10b981;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 8px #10b981;
    }

    /* Inner Card */
    .inner-card {
        background: #020617;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 16px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .inner-card:hover {
        border-color: rgba(56, 189, 248, 0.35);
        transform: translateY(-2px);
    }

    /* Satellite Image Docks */
    .img-box {
        background: #020617;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        overflow: hidden;
        padding: 10px;
        text-align: center;
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
    }
    .img-box:hover {
        transform: translateY(-4px);
        border-color: #38bdf8;
        box-shadow: 0 10px 25px rgba(56, 189, 248, 0.2);
    }
    .img-label {
        font-size: 0.82rem;
        font-weight: 700;
        color: #cbd5e1;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Severity Badges */
    .badge {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 20px;
        font-size: 1rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-none { background: #10b981; color: white; box-shadow: 0 4px 14px rgba(16, 185, 129, 0.4); }
    .badge-minor { background: #f59e0b; color: white; box-shadow: 0 4px 14px rgba(245, 158, 11, 0.4); }
    .badge-major { background: #ea580c; color: white; box-shadow: 0 4px 14px rgba(234, 88, 12, 0.4); }
    .badge-destroyed { background: #ef4444; color: white; box-shadow: 0 4px 14px rgba(239, 68, 68, 0.4); }

    /* KPI Metrics */
    .kpi {
        background: #020617;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 14px;
        text-align: center;
        transition: transform 0.2s ease;
    }
    .kpi:hover {
        transform: translateY(-2px);
    }
    .kpi-title {
        font-size: 0.76rem;
        color: #94a3b8;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .kpi-value {
        font-size: 1.5rem;
        font-weight: 800;
        color: #f8fafc;
        margin-top: 2px;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Styling for the 3 Bottom Navigation Row Cards */
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] > div.stButton > button {
        width: 100% !important;
        padding: 20px 22px !important;
        border-radius: 16px !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        line-height: 1.4 !important;
        transition: all 0.28s cubic-bezier(0.16, 1, 0.3, 1) !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        text-align: center !important;
        cursor: pointer !important;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.3) !important;
        min-height: 84px !important;
    }

    /* Inactive Bottom Card */
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] > div.stButton > button[kind="secondary"] {
        background: #0f172a !important;
        color: #cbd5e1 !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-bottom: 4px solid rgba(255, 255, 255, 0.08) !important;
    }

    /* Hover State for Bottom Cards - Lifts Up with Glowing Border */
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] > div.stButton > button[kind="secondary"]:hover {
        transform: translateY(-8px) !important;
        background: #1e293b !important;
        border-color: #38bdf8 !important;
        border-bottom: 4px solid #38bdf8 !important;
        color: #ffffff !important;
        box-shadow: 0 14px 32px rgba(56, 189, 248, 0.35) !important;
    }

    /* Active Bottom Card */
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] > div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
        border: 1px solid #60a5fa !important;
        border-bottom: 4px solid #93c5fd !important;
        color: #ffffff !important;
        box-shadow: 0 10px 28px rgba(37, 99, 235, 0.5) !important;
        transform: translateY(-4px) !important;
    }
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] > div.stButton > button[kind="primary"]:hover {
        transform: translateY(-8px) !important;
        box-shadow: 0 16px 36px rgba(37, 99, 235, 0.65) !important;
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

# Session State for Bottom Navigation (Defaults to Damage Classifier)
if "active_nav" not in st.session_state:
    st.session_state["active_nav"] = "classifier"

# Top Header
st.markdown("""
<div class="app-header">
    <div>
        <div class="app-title">🏢 DisasterVision: Building Damage Severity Classifier</div>
        <div class="app-subtitle">CSE471 Final Year Project | Two-Branch Siamese Model for Pre/Post-Disaster Satellite Damage Assessment</div>
    </div>
    <div class="status-badge">
        <span class="status-dot"></span>
        MODEL ONLINE • 77.3% ACCURACY
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# MAIN CONTENT AREA
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. DAMAGE CLASSIFIER MODULE
# ------------------------------------------------------------------------------
if st.session_state["active_nav"] == "classifier":
    st.markdown("### 🎯 Building Damage Classifier Studio")
    st.write(
        "Compare pre-disaster and post-disaster satellite crops of the identical building location. "
        "The model isolates structural damage and assigns one of 4 damage severity levels: **None**, **Minor**, **Major**, or **Destroyed**."
    )

    c_opt, c_view = st.columns([1, 2.3], gap="medium")

    with c_opt:
        with st.container(border=True):
            st.markdown("##### 📥 Input Satellite Imagery")

            mode = st.radio(
                "Image Source:",
                ["Pick an Example Scenario", "Upload Custom Images"],
            )

            pre_img = None
            post_img = None

            if mode == "Pick an Example Scenario":
                example_options = [
                    "Intact Structure (No Damage Baseline)",
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
                up_pre = st.file_uploader("Pre-Disaster Image:", type=["png", "jpg", "jpeg"])
                up_post = st.file_uploader("Post-Disaster Image:", type=["png", "jpg", "jpeg"])

                if up_pre and up_post:
                    pre_img = Image.open(up_pre).convert("RGB")
                    post_img = Image.open(up_post).convert("RGB")
                else:
                    st.info("Upload both pre- and post-disaster satellite crops to evaluate.")

            st.markdown("---")
            st.markdown("##### 🎛️ Heatmap Settings")
            opacity = st.slider("Damage Heatmap Blending:", 0.0, 1.0, 0.55, 0.05)

    with c_view:
        if pre_img is not None and post_img is not None:
            # Predict with model
            pred_label, confidence, probs, heatmap = model.predict(pre_img, post_img)
            blended_heat = blend_heatmap_on_image(post_img, heatmap, alpha=opacity)

            # 3 Images Side-by-Side
            i1, i2, i3 = st.columns(3)
            with i1:
                with st.container(border=True):
                    st.markdown("<div style='font-size: 0.82rem; font-weight: 700; color: #cbd5e1; text-transform: uppercase; margin-bottom: 4px;'>1. Pre-Disaster</div>", unsafe_allow_html=True)
                    st.image(pre_img, use_container_width=True)
                    st.caption("Intact building baseline")

            with i2:
                with st.container(border=True):
                    st.markdown("<div style='font-size: 0.82rem; font-weight: 700; color: #cbd5e1; text-transform: uppercase; margin-bottom: 4px;'>2. Post-Disaster</div>", unsafe_allow_html=True)
                    st.image(post_img, use_container_width=True)
                    st.caption("After catastrophe impact")

            with i3:
                with st.container(border=True):
                    st.markdown("<div style='font-size: 0.82rem; font-weight: 700; color: #cbd5e1; text-transform: uppercase; margin-bottom: 4px;'>3. Damage Heatmap</div>", unsafe_allow_html=True)
                    st.image(blended_heat, use_container_width=True)
                    st.caption("AI-detected structural change")

            # Severity Summary Card
            hex_color = CLASS_HEX.get({"none": 0, "minor": 1, "major": 2, "destroyed": 3}.get(pred_label, 0), "#38bdf8")
            badge_class = f"badge-{pred_label}"

            st.markdown(f"""
            <div class="inner-card" style="border-left: 6px solid {hex_color}; margin-top: 14px;">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                    <div>
                        <span style="font-size: 0.8rem; color: #94a3b8; text-transform: uppercase;">Classified Damage Severity:</span>
                        <div style="font-size: 1.5rem; font-weight: 800; color: #f8fafc; margin-top: 4px;">
                            <span class="badge {badge_class}">{pred_label.upper()}</span>
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <span style="font-size: 0.8rem; color: #94a3b8; text-transform: uppercase;">Confidence:</span>
                        <div style="font-size: 1.8rem; font-weight: 800; color: {hex_color}; font-family: 'JetBrains Mono', monospace;">
                            {confidence * 100:.1f}%
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # 4 Severity Probabilities
            st.markdown("##### Severity Probability Distribution:")
            p_cols = st.columns(4)
            labels = ["None (No Damage)", "Minor Damage", "Major Damage", "Destroyed"]
            for i, name in enumerate(labels):
                with p_cols[i]:
                    st.markdown(f"""
                    <div class="kpi">
                        <div class="kpi-title">{name}</div>
                        <div class="kpi-value" style="color: {CLASS_HEX[i]};">{probs[i]*100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(float(probs[i]))

            # Plain-English Summary
            st.markdown("##### Assessment Summary:")
            if pred_label == "none":
                st.success("✅ **No Damage:** Structural envelope and roof plane remain intact. Normal illumination differences between satellite passes correctly ignored.")
            elif pred_label == "minor":
                st.warning("⚠️ **Minor Damage:** Superficial roof tile displacement or perimeter debris observed. Primary load-bearing frame intact.")
            elif pred_label == "major":
                st.error("🚨 **Major Damage:** Partial roof collapse, substantial structural breach, or heavy foundation inundation. Unsafe for human occupancy.")
            else:
                st.error("💥 **Destroyed:** Total structural collapse, foundation scour, or charred ash footprint. Immediate search & rescue priority.")

            # Verification Report & Dossier Export
            st.markdown("---")
            st.markdown("##### 📑 Verification Dossier & Audit Report Generation")
            st.write(
                "Export an official technical assessment dossier with embedded pre/post satellite imagery, "
                "cross-attention heatmaps, probability distribution, and engineering sign-off blocks."
            )

            is_india_case = ("🇮🇳" in sample_pick) if mode == "Pick an Example Scenario" else False
            default_clf_agency = "NDRF / NDMA Structural Triage Directorate" if is_india_case else "USAR Structural Assessment Directorate"
            rep_col1, rep_col2, rep_col3 = st.columns([1.5, 1.5, 1.2])
            with rep_col1:
                assessor_name = st.text_input(
                    "Lead Technical Assessor:",
                    value="Eng. Ankit Kumar Shandilya",
                    key="classifier_assessor",
                )
            with rep_col2:
                agency_name = st.text_input(
                    "Command / Directorate:",
                    value=default_clf_agency,
                    key="classifier_agency",
                )
            with rep_col3:
                now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                loc_seed = sample_pick if mode == "Pick an Example Scenario" else "Custom Upload"
                dossier_id = generate_dossier_id(loc_seed, now_str)
                st.markdown(f"<div style='padding-top: 28px;'><span class='badge' style='background: rgba(56, 189, 248, 0.15); color: #38bdf8; font-family: monospace;'>ID: {dossier_id}</span></div>", unsafe_allow_html=True)

            loc_label = sample_pick if mode == "Pick an Example Scenario" else "Custom Orthorectified AOI"
            prob_dict = {
                "None (No Damage)": float(probs[0]),
                "Minor Damage": float(probs[1]),
                "Major Damage": float(probs[2]),
                "Destroyed": float(probs[3]),
            }

            if pred_label == "none":
                directive_text = "Structural envelope intact. Safe for civilian sheltering and staging operations. Routine follow-up scheduled."
            elif pred_label == "minor":
                directive_text = "Superficial cladding/roofing displacement. Maintain 10m perimeter clearance; primary frame structurally sound. Cleared for secondary staging."
            elif pred_label == "major":
                directive_text = "Severe structural breach or partial roof collapse. Evacuate 50m perimeter immediately. Structural shoring required prior to interior USAR entry."
            else:
                directive_text = "Total structural collapse detected. Level IV Heavy USAR deployment required. Prepare hydraulic cutting gear and IV bicarbonate for crush syndrome."

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
                    label="📥 Download Word Dossier (.docx)",
                    data=docx_bytes,
                    file_name=f"Damage_Dossier_{dossier_id}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    key="dl_classifier_docx",
                    use_container_width=True,
                )
            with btn_col2:
                st.download_button(
                    label="📄 Download Print-Ready Report (.html)",
                    data=html_str,
                    file_name=f"Damage_Report_{dossier_id}.html",
                    mime="text/html",
                    key="dl_classifier_html",
                    use_container_width=True,
                )
            with btn_col3:
                st.download_button(
                    label="💾 Download Audit Telemetry (.json)",
                    data=json_str,
                    file_name=f"Audit_Telemetry_{dossier_id}.json",
                    mime="application/json",
                    key="dl_classifier_json",
                    use_container_width=True,
                )

            with st.expander("🔍 Preview Assessment Dossier (Interactive HTML)", expanded=False):
                st.components.v1.html(html_str, height=650, scrolling=True)

# ------------------------------------------------------------------------------
# 2. INDIAN NATIONAL DISASTER WATCH & SATELLITE INGESTION MODULE
# ------------------------------------------------------------------------------
elif st.session_state["active_nav"] == "realtime":
    st.markdown("### 🛰️ Indian National Disaster Watch & Satellite Ingestion")
    st.write(
        "Ingests real-time natural disaster telemetry across India from the **National Disaster Management Authority (NDMA)**, "
        "**National Center for Seismology (NCS / MoES)**, and the **India Meteorological Department (IMD)**. "
        "Select an active Indian disaster to automatically task orbital satellite imagery and run the dual-branch structural damage classifier."
    )

    # Ingest Indian National Disasters (NCS / NDMA / IMD)
    events, source_label = get_india_disaster_catalog(use_live_network=True)

    # Status Bar & Refresh
    s_col1, s_col2 = st.columns([3, 1])
    with s_col1:
        st.markdown(f"<div style='color: #38bdf8; font-size: 0.9rem; font-weight: 600; padding: 6px 0;'>{source_label} | Tracking {len(events)} Indian Disaster Zones</div>", unsafe_allow_html=True)
    with s_col2:
        if st.button("🔄 Refresh Live Feeds", use_container_width=True):
            st.rerun()

    c_event, c_sat = st.columns([1, 2.3], gap="medium")

    with c_event:
        with st.container(border=True):
            st.markdown("##### 🔴 Active Indian Disaster Telemetry")

            # Selection dropdown
            event_titles = [f"[{e['alert_level']}] {e['title']}" for e in events]
            chosen_idx = st.selectbox(
                "Select Monitored Disaster:",
                range(len(events)),
                format_func=lambda i: event_titles[i],
            )
            ev = events[chosen_idx]

            alert_color = "#ef4444" if ev["alert_level"] == "RED" else "#f59e0b" if ev["alert_level"] == "ORANGE" else "#10b981"

            state_badge = f"<div style='font-size: 0.82rem; color: #cbd5e1; margin-bottom: 4px;'><strong>State / Territory:</strong> {ev['state']}</div>" if ev.get("state") else ""
            ndrf_badge = f"<div style='font-size: 0.8rem; color: #fbbf24; margin-top: 4px; margin-bottom: 4px;'><strong>🇮🇳 Assigned Unit:</strong> {ev['ndrf_battalion']}</div>" if ev.get("ndrf_battalion") else ""

            st.markdown(f"""
            <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 14px; margin-top: 10px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">Category:</span>
                    <span style="background: {alert_color}; color: white; padding: 2px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 700;">{ev['alert_level']} ALERT</span>
                </div>
                <div style="font-size: 1.05rem; font-weight: 700; color: #f8fafc; margin-bottom: 6px;">{ev['category']}</div>
                <div style="font-size: 0.82rem; color: #cbd5e1; margin-bottom: 4px;"><strong>Intensity:</strong> {ev['magnitude']}</div>
                <div style="font-size: 0.82rem; color: #cbd5e1; margin-bottom: 4px;"><strong>Region:</strong> {ev['place']}</div>
                {state_badge}
                {ndrf_badge}
                <div style="font-size: 0.82rem; color: #cbd5e1; margin-bottom: 4px;"><strong>Coordinates:</strong> {ev['coordinates'][0]}°N, {ev['coordinates'][1]}°E</div>
                <div style="font-size: 0.82rem; color: #cbd5e1; margin-bottom: 4px;"><strong>Recorded:</strong> {ev['time']}</div>
                <div style="font-size: 0.82rem; color: #38bdf8; margin-top: 6px;"><strong>Satellite AOI:</strong> {ev['satellite_aoi']}</div>
                <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 8px; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 6px;">
                    {ev['description']}
                </div>
            </div>
            <a href='https://www.openstreetmap.org/?mlat={ev['coordinates'][0]}&mlon={ev['coordinates'][1]}#map=16/{ev['coordinates'][0]}/{ev['coordinates'][1]}' target='_blank' style='display:block; text-align:center; margin-top:8px; font-size:0.8rem; color:#38bdf8; text-decoration:none;'>🗺️ View Epicenter on OpenStreetMap ↗</a>
            """, unsafe_allow_html=True)

            st.markdown("---")
            feed_mode = st.radio(
                "🛰️ Satellite Feed Mode:",
                ["Rectangular Building Satellite AOI (Co-Registered)", "Experimental Live Orbital Stream (ESRI / Sentinel-2)"],
                index=0,
                key="rt_stream_choice"
            )
            attempt_live_stream = (feed_mode == "Experimental Live Orbital Stream (ESRI / Sentinel-2)")

            with st.expander("📍 Custom Coordinates Satellite Explorer"):
                st.caption("Fetch live satellite tiles for any custom coordinate or global landmark:")
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
                c_sev = st.select_slider("Simulated Damage:", options=[0, 1, 2, 3], value=int(ev.get("simulated_severity", 2)), format_func=lambda s: {0:"None", 1:"Minor", 2:"Major", 3:"Destroyed"}[s], key="inp_custom_sev")
                
                if st.button("🛰️ Ingest Custom Satellite AOI", use_container_width=True, key="btn_run_custom"):
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
            st.markdown("##### 🎛️ Real-Time Heatmap Blending")
            rt_opacity = st.slider("Damage Activation Opacity:", 0.0, 1.0, 0.60, 0.05, key="rt_slider")

    with c_sat:
        # Check if custom override active
        if "custom_override_event" in st.session_state and st.session_state["custom_override_event"]:
            ev = st.session_state["custom_override_event"]
            if st.button("↩️ Reset to Live Disaster Feeds", key="btn_reset_custom"):
                st.session_state["custom_override_event"] = None
                st.rerun()

        # Load satellite imagery for event
        pre_img, post_img, meta = get_satellite_crops_for_event(ev, attempt_live_tile=attempt_live_stream)

        # Run AI prediction
        pred_label, confidence, probs, heatmap = model.predict(pre_img, post_img)
        blended_heat = blend_heatmap_on_image(post_img, heatmap, alpha=rt_opacity)

        # Live Satellite Telemetry Dock
        is_live = meta.get("is_live_tile", False)
        live_badge = "🟢 LIVE ORBITAL TILE" if is_live else "🛰️ VERIFIED SATELLITE ARCHIVE"
        badge_bg = "rgba(16, 185, 129, 0.15)" if is_live else "rgba(56, 189, 248, 0.15)"
        badge_color = "#34d399" if is_live else "#38bdf8"

        st.markdown(f"""
        <div style="background: #020617; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 10px 16px; margin-bottom: 12px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="background: {badge_bg}; color: {badge_color}; border: 1px solid {badge_color}; border-radius: 14px; padding: 2px 10px; font-size: 0.72rem; font-weight: 700;">{live_badge}</span>
                    <span style="font-size: 0.82rem; color: #f8fafc; font-weight: 600;">{meta.get('source', 'Orbital Satellite Pass')}</span>
                </div>
                <div style="font-size: 0.78rem; color: #94a3b8; font-family: 'JetBrains Mono', monospace;">
                    📍 {meta.get('coordinates', f"{ev['coordinates'][0]}°N, {ev['coordinates'][1]}°E")} | GSD: {meta.get('gsd', '0.3m/px')}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 3 Images Side-by-Side
        si1, si2, si3 = st.columns(3)
        with si1:
            with st.container(border=True):
                st.markdown("<div style='font-size: 0.82rem; font-weight: 700; color: #cbd5e1; text-transform: uppercase; margin-bottom: 4px;'>1. Pre-Disaster Baseline</div>", unsafe_allow_html=True)
                st.image(pre_img, use_container_width=True)
                st.caption(f"Intact baseline capture | {meta.get('tile_info', 'Sentinel-2 AOI')}")

        with si2:
            with st.container(border=True):
                st.markdown("<div style='font-size: 0.82rem; font-weight: 700; color: #cbd5e1; text-transform: uppercase; margin-bottom: 4px;'>2. Post-Disaster Satellite Pass</div>", unsafe_allow_html=True)
                st.image(post_img, use_container_width=True)
                st.caption(f"Post-disaster satellite pass | {ev['magnitude']}")

        with si3:
            with st.container(border=True):
                st.markdown("<div style='font-size: 0.82rem; font-weight: 700; color: #cbd5e1; text-transform: uppercase; margin-bottom: 4px;'>3. Cross-Attention Heatmap</div>", unsafe_allow_html=True)
                st.image(blended_heat, use_container_width=True)
                st.caption(f"AI structural collapse focus | {confidence*100:.1f}%")

        # Severity Summary Card
        hex_color = CLASS_HEX.get({"none": 0, "minor": 1, "major": 2, "destroyed": 3}.get(pred_label, 0), "#38bdf8")
        badge_class = f"badge-{pred_label}"

        st.markdown(f"""
        <div class="inner-card" style="border-left: 6px solid {hex_color}; margin-top: 14px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>
                    <span style="font-size: 0.8rem; color: #94a3b8; text-transform: uppercase;">Real-Time Damage Classification:</span>
                    <div style="font-size: 1.5rem; font-weight: 800; color: #f8fafc; margin-top: 4px;">
                        <span class="badge {badge_class}">{pred_label.upper()}</span>
                    </div>
                </div>
                <div style="text-align: right;">
                    <span style="font-size: 0.8rem; color: #94a3b8; text-transform: uppercase;">Confidence:</span>
                    <div style="font-size: 1.8rem; font-weight: 800; color: {hex_color}; font-family: 'JetBrains Mono', monospace;">
                        {confidence * 100:.1f}%
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 4 Probabilities
        st.markdown("##### Real-Time Damage Probability Distribution:")
        p_cols = st.columns(4)
        labels = ["None (No Damage)", "Minor Damage", "Major Damage", "Destroyed"]
        for i, name in enumerate(labels):
            with p_cols[i]:
                st.markdown(f"""
                <div class="kpi">
                    <div class="kpi-title">{name}</div>
                    <div class="kpi-value" style="color: {CLASS_HEX[i]};">{probs[i]*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(float(probs[i]))

        # First-Responder Directives
        st.markdown("##### 🚨 Automated National Disaster Response Force (NDRF / SDRF) Mandate:")
        rt_directive = get_ndrf_operational_directive(pred_label, ev)
        if pred_label in ["destroyed", "major"]:
            st.error(rt_directive)
        elif pred_label == "minor":
            st.warning(rt_directive)
        else:
            st.success(rt_directive)

        # Pipeline Architecture Card
        st.markdown("""
        <div style="background: rgba(56, 189, 248, 0.05); border: 1px dashed rgba(56, 189, 248, 0.3); border-radius: 12px; padding: 12px 16px; margin-top: 14px;">
            <span style="color: #38bdf8; font-size: 0.8rem; font-weight: 700; text-transform: uppercase;">📡 ISRO Bhuvan / NDMA Disaster Telemetry Pipeline:</span>
            <div style="color: #94a3b8; font-size: 0.82rem; margin-top: 4px; line-height: 1.5;">
                <code>NDMA Sachet / IMD Early Warning</code> &rarr; <code>ISRO Cartosat / Sentinel-2 Tasking</code> &rarr; <code>Sub-Pixel Coregistration</code> &rarr; <code>Dual-Branch Siamese CNN</code> &rarr; <code>NDRF Battalion Deployment</code>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Verification Report & Dossier Export for Real-Time Disaster Feed
        st.markdown("---")
        st.markdown("##### 📑 Indian National Disaster Verification Dossier & Audit Report")
        st.write(
            "Export an authenticated civil defense verification dossier containing live event telemetry, "
            "orbital satellite imagery, cross-attention heatmap evidence, and automated rescue directives."
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
            st.markdown(f"<div style='padding-top: 28px;'><span class='badge' style='background: rgba(56, 189, 248, 0.15); color: #38bdf8; font-family: monospace;'>ID: {rt_dossier_id}</span></div>", unsafe_allow_html=True)

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
                label="📥 Download Word Dossier (.docx)",
                data=rt_docx_bytes,
                file_name=f"India_Disaster_Dossier_{rt_dossier_id}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key="dl_rt_docx",
                use_container_width=True,
            )
        with b2:
            st.download_button(
                label="📄 Download Print-Ready Report (.html)",
                data=rt_html_str,
                file_name=f"India_Disaster_Report_{rt_dossier_id}.html",
                mime="text/html",
                key="dl_rt_html",
                use_container_width=True,
            )
        with b3:
            st.download_button(
                label="💾 Download Audit Telemetry (.json)",
                data=rt_json_str,
                file_name=f"India_Telemetry_{rt_dossier_id}.json",
                mime="application/json",
                key="dl_rt_json",
                use_container_width=True,
            )

        with st.expander("🔍 Preview Assessment Dossier (Interactive HTML)", expanded=False):
            st.components.v1.html(rt_html_str, height=650, scrolling=True)

# ------------------------------------------------------------------------------
# 3. HOW TO SAVE SOMEONE MODULE
# ------------------------------------------------------------------------------
elif st.session_state["active_nav"] == "rescue":
    st.markdown("### 🚑 How to Save Someone (Emergency Rescue Protocols)")
    st.write("Essential life-saving rules, first aid, and triage to follow when rescuing survivors from structural damage.")

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.markdown("""
        <div class="inner-card" style="height: 100%;">
            <h5 style="color: #38bdf8;">1. Check Safety First</h5>
            <p style="font-size: 0.85rem; color: #cbd5e1;">
            Ensure the area is safe before entering. Watch out for live electrical wires, gas leaks, and unstable overhead slabs.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with r2:
        st.markdown("""
        <div class="inner-card" style="height: 100%;">
            <h5 style="color: #38bdf8;">2. Acoustic Search</h5>
            <p style="font-size: 0.85rem; color: #cbd5e1;">
            Call out loudly and listen for replies. Instruct trapped victims to tap three times on pipes or solid walls if they cannot shout.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with r3:
        st.markdown("""
        <div class="inner-card" style="height: 100%;">
            <h5 style="color: #38bdf8;">3. Stop Bleeding</h5>
            <p style="font-size: 0.85rem; color: #cbd5e1;">
            Apply firm direct pressure with clean gauze. For severe limb arterial bleeding, apply a windlass tourniquet 2-3 inches above the wound.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with r4:
        st.markdown("""
        <div class="inner-card" style="height: 100%;">
            <h5 style="color: #38bdf8;">4. Prevent Shock</h5>
            <p style="font-size: 0.85rem; color: #cbd5e1;">
            Shock causes body temperature to drop dangerously. Wrap the victim in an emergency foil blanket and elevate feet 12 inches.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Critical Crush Syndrome Warning
    st.markdown("""
    <div class="inner-card" style="border-left: 6px solid #ef4444; background: rgba(239, 68, 68, 0.08);">
        <h5 style="color: #ef4444; margin-top: 0;">⚠️ Mandatory Medical Warning: Crush Syndrome</h5>
        <p style="color: #e2e8f0; font-size: 0.92rem; line-height: 1.6; margin-bottom: 0;">
        If a victim has had a limb pinned beneath a heavy concrete slab for <strong>more than 2 hours</strong>, 
        <strong>DO NOT suddenly lift the weight off</strong> without medical personnel prepared with IV hydration! 
        Sudden release washes toxic potassium and myoglobin accumulated in the crushed muscle directly into the heart, causing 
        <strong>fatal cardiac arrest or acute renal failure</strong> within minutes (reperfusion shock).
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 4-Color Triage Guide
    st.markdown("##### 🚦 Quick Triage Guide (Who to Help First):")
    t1, t2, t3, t4 = st.columns(4)
    with t1:
        st.markdown("""
        <div style="background: rgba(239, 68, 68, 0.2); border-left: 4px solid #ef4444; padding: 12px; border-radius: 8px;">
            <strong style="color: #ef4444;">🔴 Immediate (Help 1st)</strong><br>
            <span style="font-size: 0.82rem; color: #cbd5e1;">Cannot breathe easily or severe arterial bleeding.</span>
        </div>
        """, unsafe_allow_html=True)
    with t2:
        st.markdown("""
        <div style="background: rgba(245, 158, 11, 0.2); border-left: 4px solid #f59e0b; padding: 12px; border-radius: 8px;">
            <strong style="color: #f59e0b;">🟡 Delayed (Help 2nd)</strong><br>
            <span style="font-size: 0.82rem; color: #cbd5e1;">Broken bones or deep cuts, but hemodynamically stable.</span>
        </div>
        """, unsafe_allow_html=True)
    with t3:
        st.markdown("""
        <div style="background: rgba(16, 185, 129, 0.2); border-left: 4px solid #10b981; padding: 12px; border-radius: 8px;">
            <strong style="color: #10b981;">🟢 Minor (Can Walk)</strong><br>
            <span style="font-size: 0.82rem; color: #cbd5e1;">Small cuts and bruises. Can walk on their own.</span>
        </div>
        """, unsafe_allow_html=True)
    with t4:
        st.markdown("""
        <div style="background: rgba(100, 116, 139, 0.2); border-left: 4px solid #64748b; padding: 12px; border-radius: 8px;">
            <strong style="color: #94a3b8;">⚫ Deceased / Expectant</strong><br>
            <span style="font-size: 0.82rem; color: #cbd5e1;">No pulse or breathing after opening airway.</span>
        </div>
        """, unsafe_allow_html=True)

    # Official Indian National Emergency Helplines
    st.markdown("""
    <div style="background: rgba(245, 158, 11, 0.08); border: 1px dashed rgba(245, 158, 11, 0.4); border-radius: 10px; padding: 14px 18px; margin-top: 20px;">
        <span style="color: #fbbf24; font-size: 0.85rem; font-weight: 700; text-transform: uppercase;">🇮🇳 Official Indian National Emergency Helplines & Response Hotlines:</span>
        <div style="display: flex; gap: 20px; flex-wrap: wrap; margin-top: 8px; font-size: 0.82rem; color: #cbd5e1;">
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
# 3. PRECAUTIONS TO TAKE MODULE
# ------------------------------------------------------------------------------
elif st.session_state["active_nav"] == "precautions":
    st.markdown("### 🛡️ Precautions to Take (Before, During, and After)")
    st.write("Essential safety precautions across all three disaster phases and an emergency survival kit auditor.")

    stage = st.radio(
        "Disaster Phase:",
        ["Before Disaster (Preparation)", "During Disaster (Survival)", "After Disaster (Recovery)"],
        horizontal=True,
    )

    if stage == "Before Disaster (Preparation)":
        st.markdown("""
        <div class="inner-card">
            <h5>📦 Precautions Before Disaster:</h5>
            <ul style="color: #cbd5e1; line-height: 1.8;">
                <li><strong>Anchor Heavy Furniture:</strong> Strap water heaters, tall bookcases, and appliances to wall studs so they do not tip over during earthquakes.</li>
                <li><strong>Assemble 72-Hour Survival Go-Bag:</strong> Pack 4 liters of water per person per day, non-perishable rations, and medical kits.</li>
                <li><strong>Plan Evacuation Paths:</strong> Identify two independent road exits from your neighborhood and establish an emergency out-of-town contact.</li>
                <li><strong>Structural Hurricane Strapping:</strong> In cyclonic coastal regions, fasten roof trusses down to exterior bearing walls with galvanized steel hurricane ties.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    elif stage == "During Disaster (Survival)":
        st.markdown("""
        <div class="inner-card">
            <h5>🚨 Precautions During Disaster:</h5>
            <ul style="color: #cbd5e1; line-height: 1.8;">
                <li><strong>Earthquake:</strong> <strong>Drop, Cover, and Hold On!</strong> Drop to hands and knees under a sturdy desk. Never run outside or stand in doorframes.</li>
                <li><strong>Flood:</strong> <strong>Turn Around, Don't Drown!</strong> 12 inches of rushing water can sweep away passenger cars. Never drive or wade through floodwaters.</li>
                <li><strong>Cyclone / Hurricane:</strong> Shelter in an interior windowless room (hallway, closet, or bathroom) on the lowest floor.</li>
                <li><strong>Wildfire:</strong> Evacuate immediately at Level 3 'Go' notice. Don 100% cotton garments and an N95 respirator mask.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="inner-card">
            <h5>⚠️ Precautions After Disaster:</h5>
            <ul style="color: #cbd5e1; line-height: 1.8;">
                <li><strong>Do Not Flip Switches if You Smell Gas:</strong> Electrical sparks from switches ignite trapped gas. Open windows and turn off the main gas valve outside.</li>
                <li><strong>Stay 35 Feet Away from Downed Powerlines:</strong> Treat all fallen electrical lines as energized and live. Never touch standing water near downed wires.</li>
                <li><strong>Inspect for X-Cracks:</strong> Do not enter buildings displaying diagonal 45-degree shear fractures on masonry walls until cleared by engineers.</li>
                <li><strong>Boil Tap Water:</strong> Pipe breaks cross-contaminate tap water with sewage. Boil vigorously for 3 minutes before drinking.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### 🎒 72-Hour Survival Kit Checklist:")
    st.write("Check off the items you have ready:")

    chk_cols = st.columns(3)
    checked_count = 0
    total_items = len(EMERGENCY_SUPPLIES_CHECKLIST)

    for i, item in enumerate(EMERGENCY_SUPPLIES_CHECKLIST):
        with chk_cols[i % 3]:
            if st.checkbox(f"{item['item']}", key=f"chk_audit_{i}"):
                checked_count += 1

    pct = int((checked_count / total_items) * 100)
    st.markdown(f"**Emergency Kit Readiness: {pct}% ({checked_count}/{total_items} items equipped)**")
    st.progress(checked_count / total_items)

# ==============================================================================
# BOTTOM DOCK: 4 ATTRACTIVE OPTIONS IN A ROW WITH LIFT & GLOW HOVER EFFECTS
# ==============================================================================
st.markdown("<div style='margin-top: 28px; margin-bottom: 12px; text-align: center; font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: #94a3b8;'>🧭 SELECT DASHBOARD VIEW (Click an option to switch):</div>", unsafe_allow_html=True)

col_b1, col_b2, col_b3, col_b4 = st.columns(4, gap="medium")

# Option 1: Damage Classifier
with col_b1:
    is_active1 = (st.session_state["active_nav"] == "classifier")
    btn1_type = "primary" if is_active1 else "secondary"
    btn1_text = "🎯  Damage Classifier\nPre/Post Satellite Studio"
    if st.button(btn1_text, key="bottom_btn_classifier", type=btn1_type, use_container_width=True):
        st.session_state["active_nav"] = "classifier"
        st.rerun()

# Option 2: Indian Disaster Watch
with col_b2:
    is_active2 = (st.session_state["active_nav"] == "realtime")
    btn2_type = "primary" if is_active2 else "secondary"
    btn2_text = "🛰️  India Disaster Watch\nNDMA & NCS Live Telemetry"
    if st.button(btn2_text, key="bottom_btn_realtime", type=btn2_type, use_container_width=True):
        st.session_state["active_nav"] = "realtime"
        st.rerun()

# Option 3: How to Save Someone
with col_b3:
    is_active3 = (st.session_state["active_nav"] == "rescue")
    btn3_type = "primary" if is_active3 else "secondary"
    btn3_text = "🚑  How to Save Someone\nEmergency Rescue & Triage"
    if st.button(btn3_text, key="bottom_btn_rescue", type=btn3_type, use_container_width=True):
        st.session_state["active_nav"] = "rescue"
        st.rerun()

# Option 4: Precautions to Take
with col_b4:
    is_active4 = (st.session_state["active_nav"] == "precautions")
    btn4_type = "primary" if is_active4 else "secondary"
    btn4_text = "🛡️  Precautions to Take\nSafety Guide & Go-Bag"
    if st.button(btn4_text, key="bottom_btn_precautions", type=btn4_type, use_container_width=True):
        st.session_state["active_nav"] = "precautions"
        st.rerun()


# Footer
st.markdown("<br><hr style='border-color: rgba(255,255,255,0.08);'>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.85rem;'>"
    "CSE471 Final Year Project | Pre/Post-Disaster Building-Damage Severity Classifier"
    "</div>",
    unsafe_allow_html=True,
)
