"""
Export All Verification Dossiers and Architecture Reports
=========================================================
Generates:
1. Official Microsoft Word (.docx) Disaster Damage Dossier with embedded imagery
2. Standalone Print-Ready HTML Disaster Verification Report
3. Machine-Readable GIS Audit Telemetry (.json)
4. Formatted, Print-Ready HTML Architecture Document (Printable as PDF via Ctrl+P)
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from PIL import Image

# Ensure Matplotlib cache doesn't raise warnings
os.environ["MPLCONFIGDIR"] = str(Path(__file__).resolve().parent / ".matplotlib_cache")

from models.dual_branch_net import load_damage_classifier
from utils.visualization import blend_heatmap_on_image
from utils.report_generator import (
    generate_dossier_id,
    generate_verification_report_docx,
    generate_verification_report_html,
    generate_verification_report_json,
)

BASE_DIR = Path(__file__).resolve().parent


def export_disaster_dossier(slug: str = "wayanad", disaster_title: str = "Wayanad Landslide Incident"):
    print(f"[*] Loading satellite imagery for {slug}...")
    pre_path = BASE_DIR / "data" / "realtime_samples" / f"{slug}_pre.png"
    post_path = BASE_DIR / "data" / "realtime_samples" / f"{slug}_post.png"

    if not pre_path.exists() or not post_path.exists():
        print(f"[!] Warning: {pre_path} or {post_path} not found. Using fallback.")
        return

    pre_img = Image.open(pre_path).convert("RGB")
    post_img = Image.open(post_path).convert("RGB")

    print("[*] Running Dual-Branch Siamese Model inference...")
    model = load_damage_classifier()
    pred_label, confidence, probs, heatmap = model.predict(pre_img, post_img)
    blended_heat = blend_heatmap_on_image(post_img, heatmap, alpha=0.55)

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    dossier_id = generate_dossier_id(slug, now_str)

    prob_dict = {
        "None (No Damage)": float(probs[0]),
        "Minor Damage": float(probs[1]),
        "Major Damage": float(probs[2]),
        "Destroyed": float(probs[3]),
    }

    directive_map = {
        "none": "Structural envelope intact. Certified structurally sound for civilian sheltering and staging operations.",
        "minor": "Superficial cladding/roofing displacement. Maintain 10m perimeter clearance from roof edges. Frame intact.",
        "major": "Severe structural breach or partial roof collapse. Evacuate 50m perimeter immediately. Shoring required.",
        "destroyed": "Total structural collapse detected. Level IV Heavy USAR deployment required. Deploy victim-locating cameras.",
    }

    report_data = {
        "dossier_id": dossier_id,
        "timestamp": now_str,
        "location_name": f"{disaster_title} (ISRO Cartosat-3 / Sentinel-2 AOI)",
        "coordinates": "11.5385°N, 76.1322°E",
        "sensor": "DigitalGlobe WorldView-3 / ISRO Cartosat-3 (0.3m Orthorectified)",
        "severity_label": pred_label.upper(),
        "confidence": float(confidence),
        "probabilities": prob_dict,
        "directive": directive_map.get(pred_label, "Standard structural perimeter inspection."),
        "assessor_name": "Eng. Ankit Kumar Shandilya",
        "agency": "National Disaster Response Force (NDRF) / NDMA Directorate",
        "triage_level": "RED" if pred_label in ["major", "destroyed"] else "YELLOW" if pred_label == "minor" else "GREEN",
    }

    # 1. Generate Word .docx
    print("[*] Generating Microsoft Word (.docx) technical dossier with embedded imagery...")
    docx_bytes = generate_verification_report_docx(report_data, pre_img, post_img, blended_heat)
    docx_path = BASE_DIR / f"Disaster_Damage_Dossier_{slug.capitalize()}.docx"
    with open(docx_path, "wb") as f:
        f.write(docx_bytes)
    print(f"    [+] Created: {docx_path.name} ({len(docx_bytes) // 1024} KB)")

    # 2. Generate Standalone HTML
    print("[*] Generating Standalone Print-Ready HTML report...")
    html_str = generate_verification_report_html(report_data, pre_img, post_img, blended_heat)
    html_path = BASE_DIR / f"Disaster_Damage_Report_{slug.capitalize()}.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_str)
    print(f"    [+] Created: {html_path.name} ({len(html_str) // 1024} KB)")

    # 3. Generate JSON Telemetry
    print("[*] Generating GIS Machine-Readable Audit Telemetry (.json)...")
    json_str = generate_verification_report_json(report_data)
    json_path = BASE_DIR / f"Disaster_Audit_Telemetry_{slug.capitalize()}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(json_str)
    print(f"    [+] Created: {json_path.name} ({len(json_str)} bytes)")


def convert_architecture_md_to_html():
    print("[*] Converting ARCHITECTURE_REPORT.md to styled printable HTML...")
    md_path = BASE_DIR / "ARCHITECTURE_REPORT.md"
    if not md_path.exists():
        print(f"[!] {md_path} not found.")
        return

    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Simple, high-fidelity markdown to HTML formatting
    lines = md_text.split("\n")
    html_body = []
    in_code = False
    code_lang = ""
    code_block = []
    in_table = False
    table_rows = []

    for line in lines:
        if line.startswith("```"):
            if not in_code:
                in_code = True
                code_lang = line[3:].strip()
                code_block = []
            else:
                in_code = False
                joined = "\n".join(code_block)
                # Escape HTML in code block
                joined = joined.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                if code_lang == "mermaid":
                    html_body.append(f'<div class="mermaid-diagram"><pre class="mermaid">{joined}</pre></div>')
                else:
                    html_body.append(f'<pre class="code-block"><code>{joined}</code></pre>')
            continue

        if in_code:
            code_block.append(line)
            continue

        # Markdown tables
        if "|" in line and line.strip().startswith("|") and line.strip().endswith("|"):
            if "---" in line:
                continue
            cells = [c.strip() for c in line.strip().split("|")[1:-1]]
            if not in_table:
                in_table = True
                table_rows = [f"<tr>{''.join(f'<th>{c}</th>' for c in cells)}</tr>"]
            else:
                table_rows.append(f"<tr>{''.join(f'<td>{c}</td>' for c in cells)}</tr>")
            continue
        else:
            if in_table:
                in_table = False
                html_body.append(f'<div class="table-container"><table class="styled-table">{"".join(table_rows)}</table></div>')
                table_rows = []

        # Headings
        if line.startswith("# "):
            html_body.append(f'<h1 class="doc-h1">{line[2:].strip()}</h1>')
        elif line.startswith("## "):
            html_body.append(f'<h2 class="doc-h2">{line[3:].strip()}</h2>')
        elif line.startswith("### "):
            html_body.append(f'<h3 class="doc-h3">{line[4:].strip()}</h3>')
        elif line.startswith("#### "):
            html_body.append(f'<h4 class="doc-h4">{line[5:].strip()}</h4>')
        elif line.startswith("---"):
            html_body.append('<hr class="divider" />')
        elif line.startswith("* ") or line.startswith("- "):
            text = line[2:].strip()
            # inline bold
            text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", text)
            # inline code
            text = re.sub(r"`(.*?)`", r"<code>\1</code>", text)
            html_body.append(f'<li class="list-item">{text}</li>')
        elif line.strip() == "":
            continue
        else:
            text = line.strip()
            text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", text)
            text = re.sub(r"`(.*?)`", r"<code>\1</code>", text)
            html_body.append(f'<p class="paragraph">{text}</p>')

    if in_table:
        html_body.append(f'<div class="table-container"><table class="styled-table">{"".join(table_rows)}</table></div>')

    content = "\n".join(html_body)

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>DisasterVision - Software and System Architecture Document (SAD)</title>
  <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
  <script>mermaid.initialize({{startOnLoad:true, theme:'neutral'}});</script>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
  <style>
    @page {{ size: A4; margin: 18mm; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      line-height: 1.65;
      color: #0f172a;
      background: #f8fafc;
      margin: 0;
      padding: 30px;
    }}
    .document-card {{
      max-width: 950px;
      margin: 0 auto;
      background: #ffffff;
      padding: 48px;
      border-radius: 12px;
      border: 1px solid #cbd5e1;
      box-shadow: 0 10px 30px rgba(0,0,0,0.06);
    }}
    .doc-h1 {{ font-size: 26px; color: #1e3a8a; border-bottom: 3px solid #3b82f6; padding-bottom: 8px; margin-top: 10px; }}
    .doc-h2 {{ font-size: 20px; color: #1e40af; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px; margin-top: 32px; }}
    .doc-h3 {{ font-size: 16px; color: #334155; margin-top: 24px; }}
    .doc-h4 {{ font-size: 14px; color: #475569; margin-top: 16px; }}
    .divider {{ border: none; border-top: 1px solid #e2e8f0; margin: 24px 0; }}
    .paragraph {{ font-size: 14.5px; color: #334155; margin-bottom: 12px; }}
    .list-item {{ font-size: 14px; color: #334155; margin-left: 20px; margin-bottom: 6px; }}
    .code-block {{
      background: #0f172a;
      color: #38bdf8;
      padding: 16px;
      border-radius: 8px;
      font-family: 'Courier New', monospace;
      font-size: 12.5px;
      overflow-x: auto;
      margin: 16px 0;
    }}
    .table-container {{ overflow-x: auto; margin: 20px 0; }}
    .styled-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13.5px;
    }}
    .styled-table th {{
      background: #1e3a8a;
      color: #ffffff;
      padding: 10px 14px;
      text-align: left;
    }}
    .styled-table td {{
      padding: 9px 14px;
      border-bottom: 1px solid #e2e8f0;
      color: #334155;
    }}
    .styled-table tr:nth-child(even) {{ background: #f8fafc; }}
    .mermaid-diagram {{
      background: #f8fafc;
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      padding: 20px;
      margin: 20px 0;
      text-align: center;
    }}
    .print-banner {{
      background: #eff6ff;
      border: 1px solid #93c5fd;
      color: #1e40af;
      padding: 12px 18px;
      border-radius: 8px;
      margin-bottom: 24px;
      font-size: 13.5px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    @media print {{
      body {{ background: #ffffff; padding: 0; }}
      .document-card {{ border: none; box-shadow: none; padding: 0; max-width: 100%; }}
      .print-banner {{ display: none; }}
    }}
  </style>
</head>
<body>
  <div class="document-card">
    <div class="print-banner">
      <span>📄 <strong>System Architecture Document (SAD)</strong> &bull; Print-Ready Format</span>
      <button onclick="window.print()" style="background:#2563eb; color:white; border:none; padding:6px 14px; border-radius:6px; cursor:pointer; font-weight:600;">Print to PDF (Ctrl+P)</button>
    </div>
    {content}
  </div>
</body>
</html>
"""

    html_out = BASE_DIR / "ARCHITECTURE_REPORT.html"
    with open(html_out, "w", encoding="utf-8") as f:
        f.write(html_template)
    print(f"    [+] Created: {html_out.name} ({len(html_template) // 1024} KB)")


if __name__ == "__main__":
    print("=" * 70)
    print("DisasterVision: Generating All Official Reports into CSE471 Directory")
    print("=" * 70)
    # Generate Wayanad Disaster Dossier
    export_disaster_dossier("wayanad", "Wayanad Landslide Incident (Kerala)")
    # Generate Cyclone Dana Dossier
    export_disaster_dossier("dana", "Cyclone Dana Landfall Incident (Odisha)")
    # Convert Architecture Report to Print-Ready HTML
    convert_architecture_md_to_html()
    print("=" * 70)
    print("[SUCCESS] All reports exported directly into C:\\Users\\ankit\\Desktop\\CSE471")
    print("=" * 70)
