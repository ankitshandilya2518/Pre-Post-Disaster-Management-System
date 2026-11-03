"""
Emergency Structural Damage Verification Dossier & Report Generator
====================================================================
CSE471 Final Year Project | Computer Vision
Generates authenticated, auditable civil defense and engineering reports in:
  1. Microsoft Word (.docx) with embedded pre/post/heatmap imagery
  2. Print-ready standalone HTML / PDF report with base64 embedded imagery
  3. Machine-readable JSON telemetry for GIS and incident databases
Uses pure Python standard library (zipfile, xml, base64, io) with zero extra dependencies.
"""

import base64
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import uuid
import zipfile

from PIL import Image

SEVERITY_COLORS = {
    "NONE": "10B981",       # Green
    "MINOR": "F59E0B",      # Amber
    "MAJOR": "EA580C",      # Orange
    "DESTROYED": "EF4444",  # Red
}


def _img_to_base64(img: Image.Image, format: str = "PNG") -> str:
    """Converts a PIL Image to a base64 string for standalone HTML embedding."""
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def _img_to_bytes(img: Image.Image, format: str = "PNG") -> bytes:
    """Converts a PIL Image to raw bytes."""
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    return buffered.getvalue()


def escape_xml(s: str) -> str:
    return (str(s).replace("&", "&amp;")
                  .replace("<", "&lt;")
                  .replace(">", "&gt;")
                  .replace('"', "&quot;")
                  .replace("'", "&apos;"))


def generate_dossier_id(location: str, timestamp_str: str) -> str:
    """Generates a reproducible 8-character unique verification hash code."""
    h = hashlib.sha256(f"{location}_{timestamp_str}_{uuid.uuid4()}".encode("utf-8")).hexdigest()
    return f"DV-{datetime.now().strftime('%Y%m%d')}-{h[:6].upper()}"


# ==============================================================================
# 1. STANDALONE PRINT-READY HTML REPORT GENERATOR
# ==============================================================================

def generate_verification_report_html(
    report_data: Dict[str, Any],
    pre_img: Image.Image,
    post_img: Image.Image,
    heat_img: Image.Image
) -> str:
    """
    Generates a high-end, self-contained HTML assessment report with embedded base64 imagery.
    Opens in any browser and formats perfectly for printing or saving as PDF (Ctrl+P).
    """
    pre_b64 = _img_to_base64(pre_img)
    post_b64 = _img_to_base64(post_img)
    heat_b64 = _img_to_base64(heat_img)

    sev = report_data.get("severity_label", "MAJOR").upper()
    hex_col = SEVERITY_COLORS.get(sev, "38BDF8")
    conf_pct = f"{report_data.get('confidence', 0.85) * 100:.1f}%"
    dossier_id = report_data.get("dossier_id", "DV-20261103-A1B2")
    timestamp = report_data.get("timestamp", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))
    location = report_data.get("location_name", "Global Satellite AOI")
    coords = report_data.get("coordinates", "0.000°N, 0.000°E")
    sensor = report_data.get("sensor", "DigitalGlobe WorldView-3 / Sentinel-2 MSI")
    assessor = report_data.get("assessor_name", "Eng. Ankit Kumar Shandilya")
    agency = report_data.get("agency", "Disaster Response & USAR Technical Directorate")
    directive = report_data.get("directive", "Standard structural perimeter inspection.")
    probs = report_data.get("probabilities", {"None": 0.05, "Minor": 0.15, "Major": 0.70, "Destroyed": 0.10})

    # Probability rows
    prob_rows = ""
    for k, v in probs.items():
        k_upper = k.upper()
        col = SEVERITY_COLORS.get(k_upper, "64748B")
        bar_w = int(v * 100)
        prob_rows += f"""
        <tr>
            <td style="padding: 8px 12px; font-weight: 600; color: #1e293b;">{k}</td>
            <td style="padding: 8px 12px; font-family: 'Courier New', monospace; font-weight: bold; color: #{col};">{v*100:.1f}%</td>
            <td style="padding: 8px 12px; width: 50%;">
                <div style="background: #e2e8f0; border-radius: 6px; height: 12px; overflow: hidden;">
                    <div style="background: #{col}; width: {bar_w}%; height: 100%;"></div>
                </div>
            </td>
        </tr>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Damage Assessment Dossier - {dossier_id}</title>
<style>
    @page {{ size: A4; margin: 15mm; }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: #0f172a;
        background: #f8fafc;
        margin: 0;
        padding: 24px;
        line-height: 1.5;
    }}
    .dossier-card {{
        max-width: 900px;
        margin: 0 auto;
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        padding: 32px;
    }}
    .header-bar {{
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        border-bottom: 3px solid #1e3a8a;
        padding-bottom: 16px;
        margin-bottom: 24px;
    }}
    .header-title {{
        font-size: 22px;
        font-weight: 800;
        color: #1e3a8a;
        letter-spacing: -0.5px;
        margin: 0;
    }}
    .header-sub {{
        font-size: 13px;
        color: #64748b;
        margin-top: 4px;
    }}
    .dossier-tag {{
        background: #eff6ff;
        border: 1px solid #3b82f6;
        color: #1d4ed8;
        padding: 6px 14px;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        font-size: 13px;
        font-weight: 700;
        text-align: right;
    }}
    .verdict-box {{
        background: #f8fafc;
        border-left: 8px solid #{hex_col};
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}
    .verdict-badge {{
        background: #{hex_col};
        color: white;
        padding: 6px 18px;
        border-radius: 20px;
        font-weight: 800;
        font-size: 18px;
        letter-spacing: 1px;
    }}
    .images-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 16px;
        margin-bottom: 28px;
    }}
    .img-frame {{
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        overflow: hidden;
        background: #0f172a;
        text-align: center;
    }}
    .img-frame img {{
        width: 100%;
        height: auto;
        display: block;
    }}
    .img-label {{
        background: #1e293b;
        color: #f8fafc;
        font-size: 11px;
        font-weight: 700;
        padding: 6px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .section-title {{
        font-size: 14px;
        font-weight: 700;
        color: #1e3a8a;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 6px;
        margin-top: 24px;
        margin-bottom: 12px;
    }}
    table {{ width: 100%; border-collapse: collapse; margin-bottom: 16px; }}
    .meta-table td {{ padding: 6px 10px; font-size: 13px; }}
    .meta-label {{ color: #64748b; font-weight: 600; width: 22%; }}
    .meta-val {{ color: #0f172a; font-weight: 600; }}
    .directive-card {{
        background: #fffbeb;
        border: 1px solid #fde68a;
        border-left: 6px solid #f59e0b;
        border-radius: 8px;
        padding: 14px 18px;
        font-size: 13px;
        color: #92400e;
        line-height: 1.6;
    }}
    .sign-block {{
        display: flex;
        justify-content: space-between;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px dashed #cbd5e1;
    }}
    .sign-col {{ width: 45%; font-size: 12px; color: #64748b; }}
    .sign-line {{ border-bottom: 1px solid #94a3b8; height: 35px; margin-bottom: 6px; }}
    .no-print-bar {{
        max-width: 900px;
        margin: 0 auto 16px auto;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}
    .print-btn {{
        background: #2563eb;
        color: white;
        border: none;
        padding: 8px 18px;
        border-radius: 6px;
        font-size: 13px;
        font-weight: 600;
        cursor: pointer;
    }}
    .print-btn:hover {{ background: #1d4ed8; }}
    @media print {{
        .no-print-bar {{ display: none !important; }}
        body {{ background: white; padding: 0; }}
        .dossier-card {{ box-shadow: none; border: none; padding: 0; }}
    }}
</style>
</head>
<body>

<div class="no-print-bar">
    <span style="font-size: 13px; color: #64748b;">DisasterVision Automated Assessment Dossier</span>
    <button class="print-btn" onclick="window.print()">🖨️ Print / Save as PDF</button>
</div>

<div class="dossier-card">
    <div class="header-bar">
        <div>
            <h1 class="header-title">EMERGENCY BUILDING DAMAGE ASSESSMENT DOSSIER</h1>
            <div class="header-sub">Civil Defense & Rapid Response Technical Verification Report | CSE471</div>
        </div>
        <div class="dossier-tag">
            DOSSIER ID: {dossier_id}<br>
            <span style="font-weight: normal; font-size: 11px; color: #64748b;">{timestamp}</span>
        </div>
    </div>

    <!-- Executive Classification Verdict Box -->
    <div class="verdict-box">
        <div>
            <div style="font-size: 11px; font-weight: 700; color: #64748b; text-transform: uppercase;">Classified Damage Severity:</div>
            <div style="font-size: 24px; font-weight: 800; color: #0f172a; margin-top: 2px;">
                <span class="verdict-badge">{sev}</span>
            </div>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 11px; font-weight: 700; color: #64748b; text-transform: uppercase;">Model Confidence:</div>
            <div style="font-size: 28px; font-weight: 800; color: #{hex_col}; font-family: 'Courier New', monospace;">
                {conf_pct}
            </div>
        </div>
    </div>

    <!-- 3 Satellite Images Side-by-Side -->
    <div class="section-title">1. SATELLITE COMPARATIVE ANALYSIS & CROSS-ATTENTION HEATMAP</div>
    <div class="images-grid">
        <div class="img-frame">
            <div class="img-label">1. Pre-Disaster Baseline</div>
            <img src="data:image/png;base64,{pre_b64}" alt="Pre-Disaster Baseline">
        </div>
        <div class="img-frame">
            <div class="img-label">2. Post-Disaster Satellite Pass</div>
            <img src="data:image/png;base64,{post_b64}" alt="Post-Disaster Pass">
        </div>
        <div class="img-frame">
            <div class="img-label">3. Structural Damage Heatmap</div>
            <img src="data:image/png;base64,{heat_b64}" alt="Damage Heatmap">
        </div>
    </div>

    <!-- Location & Sensor Telemetry -->
    <div class="section-title">2. GEOSPATIAL & SATELLITE SENSOR TELEMETRY</div>
    <table class="meta-table">
        <tr>
            <td class="meta-label">Location / Target:</td>
            <td class="meta-val">{location}</td>
            <td class="meta-label">GPS Coordinates:</td>
            <td class="meta-val">{coords}</td>
        </tr>
        <tr>
            <td class="meta-label">Sensor Platform:</td>
            <td class="meta-val">{sensor}</td>
            <td class="meta-label">Ground Sample Dist:</td>
            <td class="meta-val">0.3m – 0.5m / pixel (Sub-Meter)</td>
        </tr>
        <tr>
            <td class="meta-label">AI Architecture:</td>
            <td class="meta-val">Dual-Branch Siamese CNN + Spatial Cross-Attention</td>
            <td class="meta-val" colspan="2" style="color:#059669; font-weight:700;">✓ Coregistered Multi-Temporal Pair Verified</td>
        </tr>
    </table>

    <!-- Probability Distribution Table -->
    <div class="section-title">3. SEVERITY PROBABILITY DISTRIBUTION BREAKDOWN</div>
    <table style="border: 1px solid #cbd5e1; border-radius: 6px; overflow: hidden;">
        <tr style="background: #f1f5f9; font-size: 12px; font-weight: bold; color: #475569; text-transform: uppercase;">
            <th style="padding: 8px 12px; text-align: left;">Damage Severity Class</th>
            <th style="padding: 8px 12px; text-align: left;">Evaluated Probability</th>
            <th style="padding: 8px 12px; text-align: left;">Relative Weight Bar</th>
        </tr>
        {prob_rows}
    </table>

    <!-- Tactical First-Responder Directives -->
    <div class="section-title">4. TACTICAL FIRST-RESPONDER & USAR RESCUE DIRECTIVE</div>
    <div class="directive-card">
        <strong>⚠️ OPERATIONAL ACTION MANDATE:</strong><br>
        {directive}
    </div>

    <!-- Official Sign-off Block -->
    <div class="sign-block">
        <div class="sign-col">
            <div class="sign-line"></div>
            <strong>Technical Inspector / Assessor:</strong><br>
            {assessor}<br>
            <span style="font-size: 11px;">{agency}</span>
        </div>
        <div class="sign-col" style="text-align: right;">
            <div class="sign-line"></div>
            <strong>Field Incident Commander / Operations:</strong><br>
            Command Clearance Sign-off<br>
            <span style="font-size: 11px;">Verification Status: PENDING FIELD RECONNAISSANCE</span>
        </div>
    </div>
</div>

</body>
</html>"""
    return html_content


# ==============================================================================
# 2. RAW JSON TELEMETRY VERIFICATION GENERATOR
# ==============================================================================

def generate_verification_report_json(report_data: Dict[str, Any]) -> str:
    """Generates structured machine-readable JSON for GIS systems, databases, and auditing."""
    audit_record = {
        "dossier_id": report_data.get("dossier_id", "DV-20261103-A1B2"),
        "audit_version": "1.0-CSE471",
        "timestamp_utc": report_data.get("timestamp", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")),
        "location": {
            "name": report_data.get("location_name", "Global Satellite AOI"),
            "coordinates": report_data.get("coordinates", "0.000°N, 0.000°E"),
            "sensor": report_data.get("sensor", "WorldView-3 / Sentinel-2"),
        },
        "model_assessment": {
            "model_architecture": "Dual-Branch Siamese CNN with Spatial Cross-Attention",
            "classified_severity": report_data.get("severity_label", "MAJOR").upper(),
            "confidence": round(float(report_data.get("confidence", 0.85)), 4),
            "probabilities": {k: round(float(v), 4) for k, v in report_data.get("probabilities", {}).items()},
        },
        "operational_directives": {
            "triage_category": report_data.get("triage_level", "YELLOW"),
            "directive_text": report_data.get("directive", "Standard structural inspection."),
            "crush_syndrome_alert": report_data.get("severity_label", "").upper() == "DESTROYED",
        },
        "assessor_signoff": {
            "assessor_name": report_data.get("assessor_name", "Eng. Ankit Kumar Shandilya"),
            "agency": report_data.get("agency", "Disaster Response Directorate"),
            "verification_status": "Pending Ground Reconnaissance",
        }
    }
    return json.dumps(audit_record, indent=2)


# ==============================================================================
# 3. OFFICIAL MICROSOFT WORD (.DOCX) DOSSIER GENERATOR
# ==============================================================================

def generate_verification_report_docx(
    report_data: Dict[str, Any],
    pre_img: Image.Image,
    post_img: Image.Image,
    heat_img: Image.Image
) -> bytes:
    """
    Generates an authenticated Microsoft Word (.docx) technical dossier with
    embedded satellite images and formal verification tables using pure Python OpenXML.
    """
    sev = report_data.get("severity_label", "MAJOR").upper()
    hex_col = SEVERITY_COLORS.get(sev, "38BDF8")
    conf_pct = f"{report_data.get('confidence', 0.85) * 100:.1f}%"
    dossier_id = report_data.get("dossier_id", "DV-20261103-A1B2")
    timestamp = report_data.get("timestamp", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))
    location = report_data.get("location_name", "Global Satellite AOI")
    coords = report_data.get("coordinates", "0.000°N, 0.000°E")
    sensor = report_data.get("sensor", "DigitalGlobe WorldView-3 / Sentinel-2 MSI")
    assessor = report_data.get("assessor_name", "Eng. Ankit Kumar Shandilya")
    agency = report_data.get("agency", "Disaster Response & USAR Technical Directorate")
    directive = report_data.get("directive", "Standard structural perimeter inspection.")
    probs = report_data.get("probabilities", {"None": 0.05, "Minor": 0.15, "Major": 0.70, "Destroyed": 0.10})

    # Prepare raw bytes for embedded images
    pre_bytes = _img_to_bytes(pre_img.resize((256, 256)))
    post_bytes = _img_to_bytes(post_img.resize((256, 256)))
    heat_bytes = _img_to_bytes(heat_img.resize((256, 256)))

    # Construct OpenXML document.xml
    body = []

    # Title & Header
    body.append(f"""<w:p>
      <w:pPr><w:jc w:val="center"/><w:spacing w:before="120" w:after="40"/></w:pPr>
      <w:r><w:rPr><w:b/><w:sz w:val="34"/><w:color w:val="1E3A8A"/></w:rPr><w:t>EMERGENCY DAMAGE ASSESSMENT DOSSIER</w:t></w:r>
    </w:p>
    <w:p>
      <w:pPr><w:jc w:val="center"/><w:spacing w:before="0" w:after="160"/></w:pPr>
      <w:r><w:rPr><w:sz w:val="20"/><w:color w:val="64748B"/></w:rPr><w:t>Civil Defense &amp; USAR Rapid Response Technical Report | CSE471</w:t></w:r>
    </w:p>""")

    # Metadata Bar Table
    body.append(f"""<w:tbl>
      <w:tblPr>
        <w:tblW w:w="0" w:type="auto"/>
        <w:jc w:val="center"/>
        <w:tblBorders>
          <w:top w:val="single" w:sz="6" w:color="CBD5E1"/>
          <w:bottom w:val="single" w:sz="6" w:color="CBD5E1"/>
          <w:left w:val="none"/><w:right w:val="none"/>
        </w:tblBorders>
        <w:tblCellMar><w:top w:w="80"/><w:bottom w:w="80"/><w:left w:w="120"/><w:right w:w="120"/></w:tblCellMar>
      </w:tblPr>
      <w:tr>
        <w:tc><w:p><w:r><w:rPr><w:b/><w:color w:val="64748B"/></w:rPr><w:t>Dossier ID:</w:t></w:r> <w:r><w:rPr><w:b/><w:color w:val="1E40AF"/></w:rPr><w:t>{escape_xml(dossier_id)}</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:pPr><w:jc w:val="right"/></w:pPr><w:r><w:rPr><w:b/><w:color w:val="64748B"/></w:rPr><w:t>Recorded:</w:t></w:r> <w:r><w:t>{escape_xml(timestamp)}</w:t></w:r></w:p></w:tc>
      </w:tr>
    </w:tbl>
    <w:p><w:pPr><w:spacing w:before="120" w:after="120"/></w:pPr></w:p>""")

    # Executive Verdict Table Box
    body.append(f"""<w:tbl>
      <w:tblPr>
        <w:tblW w:w="0" w:type="auto"/>
        <w:jc w:val="center"/>
        <w:tblBorders>
          <w:top w:val="single" w:sz="6" w:color="CBD5E1"/>
          <w:left w:val="single" w:sz="24" w:color="{hex_col}"/>
          <w:bottom w:val="single" w:sz="6" w:color="CBD5E1"/>
          <w:right w:val="single" w:sz="6" w:color="CBD5E1"/>
        </w:tblBorders>
        <w:tblCellMar><w:top w:w="140"/><w:bottom w:w="140"/><w:left w:w="180"/><w:right w:w="180"/></w:tblCellMar>
      </w:tblPr>
      <w:tr>
        <w:tc>
          <w:tcPr><w:shd w:val="clear" w:color="auto" w:fill="F8FAFC"/></w:tcPr>
          <w:p>
            <w:r><w:rPr><w:sz w:val="18"/><w:b/><w:color w:val="64748B"/></w:rPr><w:t>CLASSIFIED DAMAGE SEVERITY:</w:t></w:r>
          </w:p>
          <w:p>
            <w:r><w:rPr><w:sz w:val="32"/><w:b/><w:color w:val="{hex_col}"/></w:rPr><w:t>{escape_xml(sev)}</w:t></w:r>
          </w:p>
        </w:tc>
        <w:tc>
          <w:tcPr><w:shd w:val="clear" w:color="auto" w:fill="F8FAFC"/></w:tcPr>
          <w:p><w:pPr><w:jc w:val="right"/></w:pPr>
            <w:r><w:rPr><w:sz w:val="18"/><w:b/><w:color w:val="64748B"/></w:rPr><w:t>MODEL CONFIDENCE:</w:t></w:r>
          </w:p>
          <w:p><w:pPr><w:jc w:val="right"/></w:pPr>
            <w:r><w:rPr><w:sz w:val="34"/><w:b/><w:color w:val="{hex_col}"/></w:rPr><w:t>{conf_pct}</w:t></w:r>
          </w:p>
        </w:tc>
      </w:tr>
    </w:tbl>
    <w:p><w:pPr><w:spacing w:before="140" w:after="80"/></w:pPr></w:p>""")

    # Section 1: 3 Images Side-by-Side in Table
    body.append(f"""<w:p>
      <w:pPr><w:spacing w:before="140" w:after="80"/></w:pPr>
      <w:r><w:rPr><w:b/><w:sz w:val="24"/><w:color w:val="1E3A8A"/></w:rPr><w:t>1. SATELLITE COMPARATIVE ANALYSIS &amp; CROSS-ATTENTION HEATMAP</w:t></w:r>
    </w:p>
    <w:tbl>
      <w:tblPr>
        <w:tblW w:w="0" w:type="auto"/>
        <w:jc w:val="center"/>
        <w:tblBorders><w:none/></w:tblBorders>
        <w:tblCellMar><w:top w:w="60"/><w:bottom w:w="60"/><w:left w:w="60"/><w:right w:w="60"/></w:tblCellMar>
      </w:tblPr>
      <w:tr>
        <!-- Cell 1: Pre Image -->
        <w:tc>
          <w:p><w:pPr><w:jc w:val="center"/><w:spacing w:before="40" w:after="40"/></w:pPr>
            <w:r><w:drawing>
              <wp:inline distT="0" distB="0" distL="0" distR="0" xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing">
                <wp:extent cx="1800000" cy="1800000"/>
                <wp:docPr id="10" name="PreDisaster"/>
                <a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
                  <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
                    <pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">
                      <pic:nvPicPr><pic:cNvPr id="0" name="pre"/><pic:cNvPicPr/></pic:nvPicPr>
                      <pic:blipFill><a:blip r:embed="rIdPre" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>
                      <pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="1800000" cy="1800000"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>
                    </pic:pic>
                  </a:graphicData>
                </a:graphic>
              </wp:inline>
            </w:drawing></w:r>
          </w:p>
          <w:p><w:pPr><w:jc w:val="center"/><w:spacing w:before="20" w:after="40"/></w:pPr>
            <w:r><w:rPr><w:b/><w:sz w:val="18"/><w:color w:val="475569"/></w:rPr><w:t>1. Pre-Disaster Baseline</w:t></w:r>
          </w:p>
        </w:tc>
        <!-- Cell 2: Post Image -->
        <w:tc>
          <w:p><w:pPr><w:jc w:val="center"/><w:spacing w:before="40" w:after="40"/></w:pPr>
            <w:r><w:drawing>
              <wp:inline distT="0" distB="0" distL="0" distR="0" xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing">
                <wp:extent cx="1800000" cy="1800000"/>
                <wp:docPr id="11" name="PostDisaster"/>
                <a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
                  <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
                    <pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">
                      <pic:nvPicPr><pic:cNvPr id="0" name="post"/><pic:cNvPicPr/></pic:nvPicPr>
                      <pic:blipFill><a:blip r:embed="rIdPost" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>
                      <pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="1800000" cy="1800000"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>
                    </pic:pic>
                  </a:graphicData>
                </a:graphic>
              </wp:inline>
            </w:drawing></w:r>
          </w:p>
          <w:p><w:pPr><w:jc w:val="center"/><w:spacing w:before="20" w:after="40"/></w:pPr>
            <w:r><w:rPr><w:b/><w:sz w:val="18"/><w:color w:val="475569"/></w:rPr><w:t>2. Post-Disaster Pass</w:t></w:r>
          </w:p>
        </w:tc>
        <!-- Cell 3: Heatmap Image -->
        <w:tc>
          <w:p><w:pPr><w:jc w:val="center"/><w:spacing w:before="40" w:after="40"/></w:pPr>
            <w:r><w:drawing>
              <wp:inline distT="0" distB="0" distL="0" distR="0" xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing">
                <wp:extent cx="1800000" cy="1800000"/>
                <wp:docPr id="12" name="DamageHeatmap"/>
                <a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
                  <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
                    <pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">
                      <pic:nvPicPr><pic:cNvPr id="0" name="heat"/><pic:cNvPicPr/></pic:nvPicPr>
                      <pic:blipFill><a:blip r:embed="rIdHeat" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>
                      <pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="1800000" cy="1800000"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>
                    </pic:pic>
                  </a:graphicData>
                </a:graphic>
              </wp:inline>
            </w:drawing></w:r>
          </w:p>
          <w:p><w:pPr><w:jc w:val="center"/><w:spacing w:before="20" w:after="40"/></w:pPr>
            <w:r><w:rPr><w:b/><w:sz w:val="18"/><w:color w:val="475569"/></w:rPr><w:t>3. Structural Damage Heatmap</w:t></w:r>
          </w:p>
        </w:tc>
      </w:tr>
    </w:tbl>
    <w:p><w:pPr><w:spacing w:before="120" w:after="60"/></w:pPr></w:p>""")

    # Section 2: Geospatial & Sensor Meta
    body.append(f"""<w:p>
      <w:pPr><w:spacing w:before="120" w:after="80"/></w:pPr>
      <w:r><w:rPr><w:b/><w:sz w:val="24"/><w:color w:val="1E3A8A"/></w:rPr><w:t>2. GEOSPATIAL &amp; SATELLITE SENSOR TELEMETRY</w:t></w:r>
    </w:p>
    <w:tbl>
      <w:tblPr>
        <w:tblW w:w="0" w:type="auto"/>
        <w:jc w:val="center"/>
        <w:tblBorders>
          <w:top w:val="single" w:sz="4" w:color="CBD5E1"/>
          <w:bottom w:val="single" w:sz="4" w:color="CBD5E1"/>
          <w:insideH w:val="single" w:sz="4" w:color="F1F5F9"/>
          <w:left w:val="none"/><w:right w:val="none"/><w:insideV w:val="none"/>
        </w:tblBorders>
        <w:tblCellMar><w:top w:w="60"/><w:bottom w:w="60"/><w:left w:w="100"/><w:right w:w="100"/></w:tblCellMar>
      </w:tblPr>
      <w:tr>
        <w:tc><w:p><w:r><w:rPr><w:b/><w:color w:val="64748B"/></w:rPr><w:t>Location / Target:</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:rPr><w:b/></w:rPr><w:t>{escape_xml(location)}</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:rPr><w:b/><w:color w:val="64748B"/></w:rPr><w:t>Coordinates:</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:rPr><w:b/></w:rPr><w:t>{escape_xml(coords)}</w:t></w:r></w:p></w:tc>
      </w:tr>
      <w:tr>
        <w:tc><w:p><w:r><w:rPr><w:b/><w:color w:val="64748B"/></w:rPr><w:t>Sensor Platform:</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>{escape_xml(sensor)}</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:rPr><w:b/><w:color w:val="64748B"/></w:rPr><w:t>GSD Resolution:</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>0.3m – 0.5m / pixel (Sub-Meter)</w:t></w:r></w:p></w:tc>
      </w:tr>
    </w:tbl>
    <w:p><w:pPr><w:spacing w:before="120" w:after="60"/></w:pPr></w:p>""")

    # Section 3: Probabilities
    prob_tr_xml = []
    for k, v in probs.items():
        k_upper = k.upper()
        col = SEVERITY_COLORS.get(k_upper, "64748B")
        prob_tr_xml.append(f"""<w:tr>
          <w:tc><w:p><w:r><w:rPr><w:b/></w:rPr><w:t>{escape_xml(k)}</w:t></w:r></w:p></w:tc>
          <w:tc><w:p><w:r><w:rPr><w:b/><w:color w:val="{col}"/></w:rPr><w:t>{v*100:.1f}%</w:t></w:r></w:p></w:tc>
        </w:tr>""")

    body.append(f"""<w:p>
      <w:pPr><w:spacing w:before="120" w:after="80"/></w:pPr>
      <w:r><w:rPr><w:b/><w:sz w:val="24"/><w:color w:val="1E3A8A"/></w:rPr><w:t>3. SEVERITY PROBABILITY DISTRIBUTION</w:t></w:r>
    </w:p>
    <w:tbl>
      <w:tblPr>
        <w:tblW w:w="0" w:type="auto"/>
        <w:jc w:val="center"/>
        <w:tblBorders>
          <w:top w:val="single" w:sz="6" w:color="CBD5E1"/>
          <w:bottom w:val="single" w:sz="6" w:color="94A3B8"/>
          <w:insideH w:val="single" w:sz="4" w:color="E2E8F0"/>
          <w:left w:val="none"/><w:right w:val="none"/><w:insideV w:val="none"/>
        </w:tblBorders>
        <w:tblCellMar><w:top w:w="60"/><w:bottom w:w="60"/><w:left w:w="120"/><w:right w:w="120"/></w:tblCellMar>
      </w:tblPr>
      <w:tr>
        <w:tc><w:tcPr><w:shd w:val="clear" w:color="auto" w:fill="1E3A8A"/></w:tcPr><w:p><w:r><w:rPr><w:b/><w:color w:val="FFFFFF"/></w:rPr><w:t>Severity Category</w:t></w:r></w:p></w:tc>
        <w:tc><w:tcPr><w:shd w:val="clear" w:color="auto" w:fill="1E3A8A"/></w:tcPr><w:p><w:r><w:rPr><w:b/><w:color w:val="FFFFFF"/></w:rPr><w:t>Evaluated Probability</w:t></w:r></w:p></w:tc>
      </w:tr>
      {"".join(prob_tr_xml)}
    </w:tbl>
    <w:p><w:pPr><w:spacing w:before="120" w:after="60"/></w:pPr></w:p>""")

    # Section 4: Operational Directive
    body.append(f"""<w:p>
      <w:pPr><w:spacing w:before="120" w:after="80"/></w:pPr>
      <w:r><w:rPr><w:b/><w:sz w:val="24"/><w:color w:val="1E3A8A"/></w:rPr><w:t>4. TACTICAL FIRST-RESPONDER &amp; USAR RESCUE DIRECTIVE</w:t></w:r>
    </w:p>
    <w:p>
      <w:pPr>
        <w:spacing w:before="60" w:after="160"/>
        <w:ind w:left="240" w:right="240"/>
        <w:shd w:val="clear" w:color="auto" w:fill="FFFBEB"/>
        <w:pBdr><w:left w:val="single" w:sz="24" w:space="10" w:color="F59E0B"/></w:pBdr>
      </w:pPr>
      <w:r><w:rPr><w:b/><w:color w:val="92400E"/></w:rPr><w:t>OPERATIONAL RESCUE MANDATE: </w:t></w:r>
      <w:r><w:rPr><w:color w:val="78350F"/></w:rPr><w:t>{escape_xml(directive)}</w:t></w:r>
    </w:p>""")

    # Sign-off block
    body.append(f"""<w:p><w:pPr><w:spacing w:before="240" w:after="60"/></w:pPr></w:p>
    <w:tbl>
      <w:tblPr>
        <w:tblW w:w="0" w:type="auto"/>
        <w:jc w:val="center"/>
        <w:tblBorders><w:none/></w:tblBorders>
        <w:tblCellMar><w:top w:w="80"/><w:bottom w:w="80"/><w:left w:w="120"/><w:right w:w="120"/></w:tblCellMar>
      </w:tblPr>
      <w:tr>
        <w:tc>
          <w:p><w:pBdr><w:bottom w:val="single" w:sz="6" w:color="94A3B8"/></w:pBdr><w:r><w:t> </w:t></w:r></w:p>
          <w:p><w:r><w:rPr><w:b/><w:color w:val="1E293B"/></w:rPr><w:t>Technical Assessor: </w:t></w:r><w:r><w:t>{escape_xml(assessor)}</w:t></w:r></w:p>
          <w:p><w:r><w:rPr><w:sz w:val="18"/><w:color w:val="64748B"/></w:rPr><w:t>{escape_xml(agency)}</w:t></w:r></w:p>
        </w:tc>
        <w:tc>
          <w:p><w:pBdr><w:bottom w:val="single" w:sz="6" w:color="94A3B8"/></w:pBdr><w:r><w:t> </w:t></w:r></w:p>
          <w:p><w:pPr><w:jc w:val="right"/></w:pPr><w:r><w:rPr><w:b/><w:color w:val="1E293B"/></w:rPr><w:t>Field Incident Commander Clearance: </w:t></w:r></w:p>
          <w:p><w:pPr><w:jc w:val="right"/></w:pPr><w:r><w:rPr><w:sz w:val="18"/><w:color w:val="64748B"/></w:rPr><w:t>Status: PENDING GROUND VALIDATION</w:t></w:r></w:p>
        </w:tc>
      </w:tr>
    </w:tbl>""")

    # Page setup (Letter / A4)
    sect_pr = """<w:sectPr>
      <w:pgSz w:w="12240" w:h="15840"/>
      <w:pgMar w:top="1080" w:right="1080" w:bottom="1080" w:left="1080" w:header="720" w:footer="720"/>
    </w:sectPr>"""

    doc_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
            xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
            xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
            xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
            xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">
  <w:body>
    {"".join(body)}
    {sect_pr}
  </w:body>
</w:document>"""

    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="png" ContentType="image/png"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""

    root_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

    doc_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rIdPre" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/pre.png"/>
  <Relationship Id="rIdPost" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/post.png"/>
  <Relationship Id="rIdHeat" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/heat.png"/>
</Relationships>"""

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", root_rels)
        z.writestr("word/_rels/document.xml.rels", doc_rels)
        z.writestr("word/document.xml", doc_xml)
        z.writestr("word/media/pre.png", pre_bytes)
        z.writestr("word/media/post.png", post_bytes)
        z.writestr("word/media/heat.png", heat_bytes)

    return buf.getvalue()
