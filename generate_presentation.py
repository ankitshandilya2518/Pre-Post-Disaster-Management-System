"""
DisasterVision 10-Slide Widescreen Presentation Generator (.pptx)
Generates an executive, highly polished 10-slide PowerPoint presentation
strictly following ECMA-376 OpenXML PresentationML standards without external dependencies.
"""

import io
import os
from pathlib import Path
import re
import zipfile

# 16:9 Widescreen dimensions in EMUs
SLIDE_W = 12192000
SLIDE_H = 6858000

def escape_xml(s: str) -> str:
    return (str(s).replace("&", "&amp;")
                  .replace("<", "&lt;")
                  .replace(">", "&gt;")
                  .replace('"', "&quot;")
                  .replace("'", "&apos;"))

def shape_card(sp_id: int, name: str, x: int, y: int, cx: int, cy: int, bg_hex: str = "FFFFFF", border_hex: str = "CBD5E1", border_w: int = 12700):
    """Generates an OpenXML rectangle shape for a card container."""
    return f"""<p:sp>
  <p:nvSpPr>
    <p:cNvPr id="{sp_id}" name="{escape_xml(name)}"/>
    <p:cNvSpPr/>
    <p:nvPr/>
  </p:nvSpPr>
  <p:spPr>
    <a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>
    <a:prstGeom prst="roundRect"><a:avLst><a:gd name="adj" fmla="val 1200"/></a:avLst></a:prstGeom>
    <a:solidFill><a:srgbClr val="{bg_hex}"/></a:solidFill>
    <a:ln w="{border_w}"><a:solidFill><a:srgbClr val="{border_hex}"/></a:solidFill></a:ln>
  </p:spPr>
</p:sp>"""

def text_box(sp_id: int, name: str, x: int, y: int, cx: int, cy: int, paragraphs: list, align: str = "l", margin: int = 100000):
    """
    Generates an OpenXML text box.
    paragraphs is a list of tuples: (runs_list, font_size_pt, is_bold, color_hex, is_bullet, space_after_pt)
    where runs_list is a list of (text, is_bold, color_hex)
    """
    p_xml = []
    for item in paragraphs:
        runs, sz, bold, col, is_bullet, sp_after = item
        bu_xml = """<a:buClr><a:srgbClr val="2563EB"/></a:buClr><a:buSzPct val="80000"/><a:buChar char="&#x25CF;"/>""" if is_bullet else "<a:buNone/>"
        mar_l = ' marL="288000" indent="-288000"' if is_bullet else ""
        sp_after_val = int(sp_after * 100) if sp_after else 1000
        
        runs_xml = []
        for r_text, r_bold, r_col in runs:
            r_sz_val = int(sz * 100)
            b_flag = ' b="1"' if (r_bold or bold) else ' b="0"'
            c_val = r_col if r_col else col
            runs_xml.append(f"""<a:r>
  <a:rPr lang="en-US" sz="{r_sz_val}"{b_flag}>
    <a:solidFill><a:srgbClr val="{c_val}"/></a:solidFill>
    <a:latin typeface="Calibri"/>
  </a:rPr>
  <a:t>{escape_xml(r_text)}</a:t>
</a:r>""")

        p_xml.append(f"""<a:p>
  <a:pPr algn="{align}"{mar_l}>
    {bu_xml}
    <a:spcAft><a:spcPts val="{sp_after_val}"/></a:spcAft>
  </a:pPr>
  {"".join(runs_xml)}
</a:p>""")

    return f"""<p:sp>
  <p:nvSpPr>
    <p:cNvPr id="{sp_id}" name="{escape_xml(name)}"/>
    <p:cNvSpPr txBox="1"/>
    <p:nvPr/>
  </p:nvSpPr>
  <p:spPr>
    <a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>
    <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
    <a:noFill/>
  </p:spPr>
  <p:txBody>
    <a:bodyPr lIns="{margin}" tIns="{margin}" rIns="{margin}" bIns="{margin}" wrap="square"/>
    <a:lstStyle/>
    {"".join(p_xml)}
  </p:txBody>
</p:sp>"""

def slide_header(sp_start: int, category: str, title: str, subtitle: str):
    """Standard professional slide header for content slides."""
    shapes = []
    # Badge / Category
    shapes.append(shape_card(sp_start, "HeaderBadge", 685800, 457200, 3200000, 320000, bg_hex="EFF6FF", border_hex="3B82F6", border_w=10000))
    badge_p = [([ (category.upper(), True, "1E40AF") ], 10, True, "1E40AF", False, 0)]
    shapes.append(text_box(sp_start+1, "BadgeText", 685800, 470000, 3200000, 300000, badge_p, align="ctr", margin=0))
    
    # Title & Subtitle
    title_p = [
        ([ (title, True, "0F172A") ], 24, True, "0F172A", False, 4),
        ([ (subtitle, False, "475569") ], 12, False, "475569", False, 0)
    ]
    shapes.append(text_box(sp_start+2, "SlideTitle", 685800, 850000, 10820400, 950000, title_p, align="l", margin=0))
    return "".join(shapes)

def image_shape(sp_id: int, r_id: str, x: int, y: int, cx: int, cy: int, desc: str):
    """Draws an embedded image."""
    return f"""<p:pic>
  <p:nvPicPr>
    <p:cNvPr id="{sp_id}" name="{escape_xml(desc)}"/>
    <p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr>
    <p:nvPr/>
  </p:nvPicPr>
  <p:blipFill>
    <a:blip r:embed="{r_id}"/>
    <a:stretch><a:fillRect/></a:stretch>
  </p:blipFill>
  <p:spPr>
    <a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>
    <a:prstGeom prst="roundRect"><a:avLst><a:gd name="adj" fmla="val 800"/></a:avLst></a:prstGeom>
    <a:ln w="12700"><a:solidFill><a:srgbClr val="CBD5E1"/></a:solidFill></a:ln>
  </p:spPr>
</p:pic>"""

# ==============================================================================
# 10 SLIDES CONTENT BUILDER
# ==============================================================================

def build_slide_1():
    """Slide 1: Title Slide (Dark Navy Executive Theme)"""
    shapes = []
    # Dark Background
    shapes.append(f"""<p:sp>
  <p:nvSpPr><p:cNvPr id="101" name="Bg"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
  <p:spPr>
    <a:xfrm><a:off x="0" y="0"/><a:ext cx="{SLIDE_W}" cy="{SLIDE_H}"/></a:xfrm>
    <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
    <a:solidFill><a:srgbClr val="0B1329"/></a:solidFill>
    <a:ln><a:noFill/></a:ln>
  </p:spPr>
</p:sp>""")

    # Top Pill Badge
    shapes.append(shape_card(102, "HeroPill", 1000000, 1000000, 4200000, 360000, bg_hex="1E293B", border_hex="38BDF8", border_w=12700))
    pill_p = [([ ("CSE471 FINAL YEAR CAPSTONE PROJECT", True, "38BDF8") ], 11, True, "38BDF8", False, 0)]
    shapes.append(text_box(103, "HeroPillText", 1000000, 1030000, 4200000, 300000, pill_p, align="ctr", margin=0))

    # Main Title & Subtitle
    title_p = [
        ([ ("DisasterVision: Pre/Post-Disaster", True, "FFFFFF") ], 34, True, "FFFFFF", False, 4),
        ([ ("Building-Damage Severity Classifier", True, "38BDF8") ], 34, True, "38BDF8", False, 16),
        ([ ("A Dual-Branch Siamese Network with Spatial Cross-Attention,", False, "94A3B8") ], 16, False, "94A3B8", False, 2),
        ([ ("Real-Time Global Event Telemetry, and Emergency Rescue Intelligence", False, "CBD5E1") ], 16, False, "CBD5E1", False, 0),
    ]
    shapes.append(text_box(104, "MainTitle", 1000000, 1500000, 10192000, 2400000, title_p, align="l", margin=0))

    # Author Card (Left Bottom)
    shapes.append(shape_card(105, "AuthorCard", 1000000, 4300000, 4800000, 1800000, bg_hex="111C38", border_hex="1E3A8A", border_w=12700))
    author_p = [
        ([ ("CANDIDATE / LEAD RESEARCHER", True, "38BDF8") ], 10, True, "38BDF8", False, 6),
        ([ ("Ankit Kumar Shandilya", True, "FFFFFF") ], 16, True, "FFFFFF", False, 4),
        ([ ("School of Computer Science & Engineering", False, "CBD5E1") ], 11, False, "CBD5E1", False, 2),
        ([ ("Lovely Professional University, Phagwara, Punjab", False, "94A3B8") ], 11, False, "94A3B8", False, 2),
        ([ ("Email: ankitkumar20218321@gmail.com", False, "64748B") ], 10, False, "64748B", False, 0),
    ]
    shapes.append(text_box(106, "AuthorText", 1000000, 4300000, 4800000, 1800000, author_p, align="l", margin=180000))

    # System Highlights Card (Right Bottom)
    shapes.append(shape_card(107, "StatsCard", 6100000, 4300000, 5092000, 1800000, bg_hex="111C38", border_hex="1E3A8A", border_w=12700))
    stats_p = [
        ([ ("SYSTEM HIGHLIGHTS & BENCHMARK RESULTS", True, "38BDF8") ], 10, True, "38BDF8", False, 6),
        ([ ("• 77.33% Accuracy", True, "10B981"), (" (+15.03% gain vs single-image baseline)", False, "CBD5E1") ], 12, False, "CBD5E1", False, 4),
        ([ ("• 100% Intact Recall", True, "10B981"), (" (0 false alarms on undamaged buildings)", False, "CBD5E1") ], 12, False, "CBD5E1", False, 4),
        ([ ("• Real-Time Feeds: ", True, "38BDF8"), ("Live USGS Earthquake & NASA EONET v3 ingestion", False, "CBD5E1") ], 12, False, "CBD5E1", False, 4),
        ([ ("• Life-Safety Action: ", True, "F59E0B"), ("START Triage & Golden Hour Crush Syndrome alerts", False, "CBD5E1") ], 12, False, "CBD5E1", False, 0),
    ]
    shapes.append(text_box(108, "StatsText", 6100000, 4300000, 5092000, 1800000, stats_p, align="l", margin=180000))

    return "".join(shapes)

def build_slide_2():
    """Slide 2: Introduction & Operational Motivation"""
    shapes = [slide_header(200, "1. Background & Context", "The Post-Disaster Information Vacuum", "Why rapid, automated structural damage assessment from orbit is critical for survival.")]
    
    # Card 1: The Problem
    shapes.append(shape_card(210, "Card1", 685800, 2000000, 3400000, 4300000, bg_hex="F8FAFC", border_hex="CBD5E1"))
    c1_p = [
        ([ ("🚨 The 'Golden Window'", True, "1E3A8A") ], 15, True, "1E3A8A", False, 10),
        ([ ("Critical 24–72 Hours", True, "0F172A") ], 12, True, "0F172A", False, 8),
        ([ ("Survival rates drop exponentially after 72 hours for trapped casualties.", False, "475569") ], 11, False, "475569", True, 8),
        ([ ("Ground reconnaissance blocked by severed bridges, landslides, and debris.", False, "475569") ], 11, False, "475569", True, 8),
        ([ ("Satellites offer the only ubiquitous view of metropolitan destruction.", False, "475569") ], 11, False, "475569", True, 0),
    ]
    shapes.append(text_box(211, "Card1Text", 685800, 2000000, 3400000, 4300000, c1_p, margin=200000))

    # Card 2: Single-Image Limitation
    shapes.append(shape_card(220, "Card2", 4395800, 2000000, 3400000, 4300000, bg_hex="FEF2F2", border_hex="FCA5A5"))
    c2_p = [
        ([ ("⚠️ Single-Image Failure", True, "991B1B") ], 15, True, "991B1B", False, 10),
        ([ ("Why Post-Only AI Fails", True, "7F1D1D") ], 12, True, "7F1D1D", False, 8),
        ([ ("Ambiguity: Cannot distinguish old gravel/soil from newly collapsed rubble.", False, "475569") ], 11, False, "475569", True, 8),
        ([ ("Solar Variations: Varying sun azimuth angles cast shadows mistaken for damage.", False, "475569") ], 11, False, "475569", True, 8),
        ([ ("Off-Nadir Skew: Angled passes show facades mistaken for structural breach.", False, "475569") ], 11, False, "475569", True, 0),
    ]
    shapes.append(text_box(221, "Card2Text", 4395800, 2000000, 3400000, 4300000, c2_p, margin=200000))

    # Card 3: Our Solution
    shapes.append(shape_card(230, "Card3", 8105800, 2000000, 3400000, 4300000, bg_hex="F0FDF4", border_hex="86EFAC"))
    c3_p = [
        ([ ("✅ The DisasterVision Solution", True, "166534") ], 15, True, "166534", False, 10),
        ([ ("Dual-Branch Reasoning", True, "14532D") ], 12, True, "14532D", False, 8),
        ([ ("Co-registered Pairs: Compares intact baseline with post-impact satellite crops.", False, "475569") ], 11, False, "475569", True, 8),
        ([ ("Differential Subtraction: Isolates true catastrophe damage from ambient noise.", False, "475569") ], 11, False, "475569", True, 8),
        ([ ("Actionable Triage: Automatically maps damage to life-safety rescue protocols.", False, "475569") ], 11, False, "475569", True, 0),
    ]
    shapes.append(text_box(231, "Card3Text", 8105800, 2000000, 3400000, 4300000, c3_p, margin=200000))

    return "".join(shapes)

def build_slide_3():
    """Slide 3: Problem Formulation & Research Objectives"""
    shapes = [slide_header(300, "2. Problem Definition", "Problem Formulation & Research Objectives", "Rigorous mathematical framing of building damage assessment and operational research goals.")]

    # Left: Joint Damage Tiers Card
    shapes.append(shape_card(310, "TiersCard", 685800, 2000000, 5200000, 4300000, bg_hex="F8FAFC", border_hex="CBD5E1"))
    left_p = [
        ([ ("🎯 The 4-Tier Joint Damage Scale", True, "1E3A8A") ], 15, True, "1E3A8A", False, 12),
        ([ ("Class 0: None (Undamaged)", True, "10B981") ], 12, True, "10B981", False, 2),
        ([ ("Roof envelope intact, no structural collapse. Retains baseline state.", False, "475569") ], 10.5, False, "475569", False, 8),
        ([ ("Class 1: Minor Damage", True, "F59E0B") ], 12, True, "F59E0B", False, 2),
        ([ ("Superficial roof tile loss, light siding peel, structure framing sound.", False, "475569") ], 10.5, False, "475569", False, 8),
        ([ ("Class 2: Major Damage", True, "EA580C") ], 12, True, "EA580C", False, 2),
        ([ ("Substantial roof cave-in, primary wall breaches, standing water inundation.", False, "475569") ], 10.5, False, "475569", False, 8),
        ([ ("Class 3: Destroyed", True, "EF4444") ], 12, True, "EF4444", False, 2),
        ([ ("Pancake structural collapse, complete foundation scour, charred ash.", False, "475569") ], 10.5, False, "475569", False, 0),
    ]
    shapes.append(text_box(311, "TiersText", 685800, 2000000, 5200000, 4300000, left_p, margin=220000))

    # Right: Research Questions Card
    shapes.append(shape_card(320, "RQCard", 6305800, 2000000, 5200000, 4300000, bg_hex="EFF6FF", border_hex="93C5FD"))
    right_p = [
        ([ ("🔬 Core Research Questions (RQs)", True, "1E40AF") ], 15, True, "1E40AF", False, 12),
        ([ ("RQ1: Pre/Post Siamese Gain", True, "1E3A8A") ], 12, True, "1E3A8A", False, 2),
        ([ ("Does weight-shared Siamese feature comparison eliminate false alarms caused by solar shadows and off-nadir viewing angles?", False, "334155") ], 10.5, False, "334155", False, 10),
        ([ ("RQ2: Spatial Cross-Attention", True, "1E3A8A") ], 12, True, "1E3A8A", False, 2),
        ([ ("Can cross-attention between post queries and pre keys accurately separate subtle Minor from Major damage breaches?", False, "334155") ], 10.5, False, "334155", False, 10),
        ([ ("RQ3: Live Telemetry Feasibility", True, "1E3A8A") ], 12, True, "1E3A8A", False, 2),
        ([ ("Can live USGS & NASA global feeds be ingested to autonomously task satellite AOI crops within field-deployable latencies (< 1 s)?", False, "334155") ], 10.5, False, "334155", False, 0),
    ]
    shapes.append(text_box(321, "RQText", 6305800, 2000000, 5200000, 4300000, right_p, margin=220000))

    return "".join(shapes)

def build_slide_4():
    """Slide 4: Dataset & Multi-Hazard Coverage"""
    shapes = [slide_header(400, "3. Benchmark Dataset", "xBD Dataset & Multi-Hazard Diversity", "850,000+ building polygons across 19 global catastrophes at 0.3m ground resolution.")]

    # 4 Cards for Disaster Coverage
    cards_data = [
        ("🌋 Earthquakes", "Mexico City & Noto Peninsula", "Pancake collapses, soft-story shearing, structural rubble, tilted foundations.", "F8FAFC", "CBD5E1", 685800, 2000000),
        ("🔥 Wildfires", "Santa Rosa & Texas Panhandle", "Charred foundations, collapsed rafters, total loss of combustible roofs.", "FFF7ED", "FDBA74", 6305800, 2000000),
        ("🌀 Hurricanes", "Hurricane Florence & Helene", "Wind-induced roof peel-off, siding stripping, coastal storm surge.", "F0FDF4", "86EFAC", 685800, 4250000),
        ("🌊 Tsunamis & Floods", "Palu Tsunami & Derna Breach", "Total hydrodynamic foundation scouring, standing water inundation.", "EFF6FF", "93C5FD", 6305800, 4250000)
    ]

    for idx, (title, sub, desc, bg, bdr, x, y) in enumerate(cards_data):
        shapes.append(shape_card(410 + idx*2, f"DisCard{idx}", x, y, 5200000, 2050000, bg_hex=bg, border_hex=bdr))
        p = [
            ([ (title, True, "0F172A") ], 14, True, "0F172A", False, 2),
            ([ (sub, True, "2563EB") ], 11, True, "2563EB", False, 4),
            ([ (desc, False, "475569") ], 10.5, False, "475569", False, 0),
        ]
        shapes.append(text_box(411 + idx*2, f"DisText{idx}", x, y, 5200000, 2050000, p, margin=160000))

    return "".join(shapes)

def build_slide_5():
    """Slide 5: Proposed Dual-Branch Architecture with Cross-Attention"""
    shapes = [slide_header(500, "4. Deep Learning Architecture", "Dual-Branch Siamese Network with Spatial Cross-Attention", "End-to-end comparative convolutional framework with 4-way interaction fusion.")]

    # Left: Siamese Architecture Flow
    shapes.append(shape_card(510, "ArchCard", 685800, 2000000, 5200000, 4300000, bg_hex="F8FAFC", border_hex="CBD5E1"))
    left_p = [
        ([ ("🏗️ Network Architectural Components", True, "1E3A8A") ], 14, True, "1E3A8A", False, 10),
        ([ ("1. Shared Siamese Convolutions", True, "0F172A") ], 12, True, "0F172A", False, 2),
        ([ ("Pre and post satellite crops pass through identical weight-sharing ResNet blocks. Projects inputs into aligned latent space [256 x 8 x 8].", False, "475569") ], 10.5, False, "475569", False, 8),
        ([ ("2. Spatial Cross-Attention", True, "0F172A") ], 12, True, "0F172A", False, 2),
        ([ ("Post-disaster features serve as Queries (Q); intact pre baseline serves as Keys (K) and Values (V). Accentuates structural changes.", False, "475569") ], 10.5, False, "475569", False, 8),
        ([ ("3. Multilayer Classification Head", True, "0F172A") ], 12, True, "0F172A", False, 2),
        ([ ("GAP collapses spatial tensor to 1024-D vector. Two-layer MLP with GELU and Dropout (p=0.3) produces 4-class logits.", False, "475569") ], 10.5, False, "475569", False, 0),
    ]
    shapes.append(text_box(511, "ArchText", 685800, 2000000, 5200000, 4300000, left_p, margin=200000))

    # Right: 4-Way Fusion & Mathematical Formulation
    shapes.append(shape_card(520, "MathCard", 6305800, 2000000, 5200000, 4300000, bg_hex="EFF6FF", border_hex="93C5FD"))
    right_p = [
        ([ ("📐 4-Way Difference & Fusion Layer", True, "1E40AF") ], 14, True, "1E40AF", False, 10),
        ([ ("F_fused = [ f_post , f_pre , |f_post - f_pre| , f_post ⊙ f_pre ]", True, "2563EB") ], 11, True, "2563EB", False, 8),
        ([ ("• f_post: ", True, "0F172A"), ("Preserves absolute post-disaster features.", False, "475569") ], 10.5, False, "475569", False, 4),
        ([ ("• f_pre: ", True, "0F172A"), ("Maintains intact architectural baseline context.", False, "475569") ], 10.5, False, "475569", False, 4),
        ([ ("• |f_post - f_pre|: ", True, "0F172A"), ("Isolates change magnitude (illumination invariant).", False, "475569") ], 10.5, False, "475569", False, 4),
        ([ ("• f_post ⊙ f_pre: ", True, "0F172A"), ("Hadamard product measures structural feature correlation.", False, "475569") ], 10.5, False, "475569", False, 10),
        ([ ("🔍 Explainable Heatmap Activation", True, "1E40AF") ], 13, True, "1E40AF", False, 2),
        ([ ("Difference activations are normalized and blended onto post-disaster crops to provide first responders with transparent AI explanations.", False, "475569") ], 10.5, False, "475569", False, 0),
    ]
    shapes.append(text_box(521, "MathText", 6305800, 2000000, 5200000, 4300000, right_p, margin=200000))

    return "".join(shapes)

def build_slide_6():
    """Slide 6: Real-Time Event Ingestion Engine (USGS & NASA EONET)"""
    shapes = [slide_header(600, "5. Real-Time Telemetry", "Live Disaster Watch & Automated Satellite Ingestion", "Bridging academic computer vision to operational global disaster monitoring networks.")]

    # Left: API Connectors
    shapes.append(shape_card(610, "ApiCard", 685800, 2000000, 5200000, 4300000, bg_hex="F8FAFC", border_hex="CBD5E1"))
    left_p = [
        ([ ("🛰️ Live Earth Science Ingestion APIs", True, "1E3A8A") ], 14, True, "1E3A8A", False, 10),
        ([ ("1. USGS Earthquake Hazards Program", True, "0F172A") ], 12, True, "0F172A", False, 2),
        ([ ("Polls real-time GeoJSON for global seismic events (M >= 4.5). Ingests epicenter coordinates [lat, lon], depth, and magnitude.", False, "475569") ], 10.5, False, "475569", False, 10),
        ([ ("2. NASA Earth Observatory (EONET v3)", True, "0F172A") ], 12, True, "0F172A", False, 2),
        ([ ("Tracks active tropical cyclones, severe flood inundations, and wildfire perimeters across the globe via NASA EOS satellites.", False, "475569") ], 10.5, False, "475569", False, 10),
        ([ ("3. Fail-Safe Offline Benchmark Cache", True, "0F172A") ], 12, True, "0F172A", False, 2),
        ([ ("Embedded verified real-world disaster catalog (Noto, Helene, Turkey) guarantees 100% demo uptime even without internet access.", False, "475569") ], 10.5, False, "475569", False, 0),
    ]
    shapes.append(text_box(611, "ApiText", 685800, 2000000, 5200000, 4300000, left_p, margin=200000))

    # Right: Pipeline & Latency
    shapes.append(shape_card(620, "PipeCard", 6305800, 2000000, 5200000, 4300000, bg_hex="EFF6FF", border_hex="93C5FD"))
    right_p = [
        ([ ("⚡ End-to-End Operational Pipeline", True, "1E40AF") ], 14, True, "1E40AF", False, 10),
        ([ ("Step 1: Event Broadcast Detection", True, "0F172A") ], 11.5, True, "0F172A", False, 2),
        ([ ("USGS / NASA event detected (lat, lon, origin time extracted).", False, "475569") ], 10.5, False, "475569", False, 6),
        ([ ("Step 2: Automated Satellite AOI Tasking", True, "0F172A") ], 11.5, True, "0F172A", False, 2),
        ([ ("Pulls matching pre-baseline and post-impact satellite tiles.", False, "475569") ], 10.5, False, "475569", False, 6),
        ([ ("Step 3: Two-Branch Siamese AI Inference", True, "0F172A") ], 11.5, True, "0F172A", False, 2),
        ([ ("Spatial cross-attention computes damage severity in 42 ms.", False, "475569") ], 10.5, False, "475569", False, 6),
        ([ ("Step 4: Automated USAR Triage Dispatch", True, "0F172A") ], 11.5, True, "0F172A", False, 2),
        ([ ("Instant directive issued to emergency operations command.", False, "475569") ], 10.5, False, "475569", False, 12),
        ([ ("⏱️ Measured End-to-End Latency: < 0.9 Seconds", True, "16A34A") ], 12, True, "16A34A", False, 0),
    ]
    shapes.append(text_box(621, "PipeText", 6305800, 2000000, 5200000, 4300000, right_p, margin=200000))

    return "".join(shapes)

def build_slide_7():
    """Slide 7: Emergency Rescue Intelligence & Clinical Triage"""
    shapes = [slide_header(700, "6. Operational Intelligence", "Life-Saving Protocols & Emergency Rescue Intelligence", "Transforming abstract machine learning outputs into frontline tactical directives.")]

    # 3 Cards: Triage, Crush Syndrome, Preparedness
    # Card 1: START Triage
    shapes.append(shape_card(710, "TriCard", 685800, 2000000, 3400000, 4300000, bg_hex="F8FAFC", border_hex="CBD5E1"))
    c1_p = [
        ([ ("🚦 START Rescue Triage", True, "1E3A8A") ], 14, True, "1E3A8A", False, 8),
        ([ ("Simple Triage & Rapid Treatment", False, "64748B") ], 10.5, False, "64748B", False, 10),
        ([ ("🔴 Red (Immediate): ", True, "EF4444"), ("Pancake collapse voids, severe arterial bleeding. Help 1st.", False, "475569") ], 10.5, False, "475569", False, 8),
        ([ ("🟡 Yellow (Delayed): ", True, "F59E0B"), ("Major fractures but breathing stable. Help 2nd.", False, "475569") ], 10.5, False, "475569", False, 8),
        ([ ("🟢 Green (Minor): ", True, "10B981"), ("Walking wounded; ambulatory survivors.", False, "475569") ], 10.5, False, "475569", False, 8),
        ([ ("⚫ Black (Expectant): ", True, "64748B"), ("No pulse/breathing after airway open.", False, "475569") ], 10.5, False, "475569", False, 0),
    ]
    shapes.append(text_box(711, "TriText", 685800, 2000000, 3400000, 4300000, c1_p, margin=180000))

    # Card 2: Crush Syndrome Warning
    shapes.append(shape_card(720, "CrushCard", 4395800, 2000000, 3400000, 4300000, bg_hex="FEF2F2", border_hex="FCA5A5"))
    c2_p = [
        ([ ("⚠️ Medical Warning", True, "991B1B") ], 14, True, "991B1B", False, 8),
        ([ ("Crush Injury Reperfusion Shock", True, "7F1D1D") ], 11, True, "7F1D1D", False, 10),
        ([ ("• Critical Rule: ", True, "991B1B"), ("If victim is pinned under heavy concrete for > 2 hours, DO NOT suddenly lift the slab!", False, "475569") ], 10.5, False, "475569", False, 8),
        ([ ("• Mechanism: ", True, "0F172A"), ("Sudden release flushes toxic potassium & myoglobin into circulation.", False, "475569") ], 10.5, False, "475569", False, 8),
        ([ ("• Protocol: ", True, "0F172A"), ("Establish IV saline/bicarbonate BEFORE extrication to prevent fatal cardiac arrest.", False, "475569") ], 10.5, False, "475569", False, 0),
    ]
    shapes.append(text_box(721, "CrushText", 4395800, 2000000, 3400000, 4300000, c2_p, margin=180000))

    # Card 3: 72-Hour Kit
    shapes.append(shape_card(730, "KitCard", 8105800, 2000000, 3400000, 4300000, bg_hex="F0FDF4", border_hex="86EFAC"))
    c3_p = [
        ([ ("🎒 Disaster Preparedness", True, "166534") ], 14, True, "166534", False, 8),
        ([ ("72-Hour Survival Go-Bag", True, "14532D") ], 11, True, "14532D", False, 10),
        ([ ("• 4L Water / person / day (minimum 3-day supply).", False, "475569") ], 10.5, False, "475569", True, 8),
        ([ ("• Non-perishable energy bars & MRE rations.", False, "475569") ], 10.5, False, "475569", True, 8),
        ([ ("• Windlass tourniquet, clean trauma gauze, emergency blanket.", False, "475569") ], 10.5, False, "475569", True, 8),
        ([ ("• Hand-crank emergency NOAA radio & whistle.", False, "475569") ], 10.5, False, "475569", True, 0),
    ]
    shapes.append(text_box(731, "KitText", 8105800, 2000000, 3400000, 4300000, c3_p, margin=180000))

    return "".join(shapes)

def build_slide_8():
    """Slide 8: Experimental Results & Benchmark Evaluation"""
    shapes = [slide_header(800, "7. Results & Evaluation", "Benchmark Performance & Empirical Validation", "Quantitative evaluation on 300 multi-hazard test pairs with per-severity analysis.")]

    # Top KPI Row (3 stat boxes)
    kpis = [
        ("77.33%", "OVERALL ACCURACY", "+15.03% gain over post-only baseline", "10B981", "F0FDF4", "86EFAC", 685800),
        ("0.770", "MACRO F1-SCORE", "+0.163 boost across all 4 severity tiers", "2563EB", "EFF6FF", "93C5FD", 4395800),
        ("100%", "INTACT CLASS RECALL", "0 false alarms on undamaged building pairs", "059669", "F0FDF4", "86EFAC", 8105800)
    ]
    for idx, (num, label, desc, col, bg, bdr, x) in enumerate(kpis):
        shapes.append(shape_card(810 + idx*2, f"KpiCard{idx}", x, 2000000, 3400000, 1100000, bg_hex=bg, border_hex=bdr))
        p = [
            ([ (num, True, col) ], 22, True, col, False, 2),
            ([ (label, True, "0F172A") ], 10, True, "0F172A", False, 1),
            ([ (desc, False, "64748B") ], 9, False, "64748B", False, 0),
        ]
        shapes.append(text_box(811 + idx*2, f"KpiText{idx}", x, 2000000, 3400000, 1100000, p, align="ctr", margin=100000))

    # Bottom Left: Confusion Matrix Summary
    shapes.append(shape_card(830, "CmCard", 685800, 3300000, 5200000, 3000000, bg_hex="F8FAFC", border_hex="CBD5E1"))
    cm_p = [
        ([ ("📊 4x4 Confusion Matrix Highlights", True, "1E3A8A") ], 13, True, "1E3A8A", False, 8),
        ([ ("• None (0): ", True, "10B981"), ("75/75 correct (100% recall) — absolute invariance to lighting shifts.", False, "475569") ], 10.5, False, "475569", False, 6),
        ([ ("• Minor (1): ", True, "F59E0B"), ("57/75 correct (0.740 F1) — dramatic +29.8% gain over baseline.", False, "475569") ], 10.5, False, "475569", False, 6),
        ([ ("• Major (2): ", True, "EA580C"), ("47/75 correct (0.648 F1) — resolves structural envelope breach.", False, "475569") ], 10.5, False, "475569", False, 6),
        ([ ("• Destroyed (3): ", True, "EF4444"), ("53/75 correct (96.4% precision) — zero false alarms on total collapse.", False, "475569") ], 10.5, False, "475569", False, 0),
    ]
    shapes.append(text_box(831, "CmText", 685800, 3300000, 5200000, 3000000, cm_p, margin=180000))

    # Bottom Right: Embedded Visual Figure (Confusion Matrix Plot)
    # Check if figure exists; if so, embed image
    shapes.append(image_shape(840, "rIdCmImg", 6305800, 3300000, 5200000, 3000000, "Confusion Matrix Figure"))

    return "".join(shapes)

def build_slide_9():
    """Slide 9: Conclusion & Future Scope"""
    shapes = [slide_header(900, "8. Conclusion", "Project Summary & Future Horizons", "Synthesizing research contributions and outlining next-generation multimodal capabilities.")]

    # Left: Key Conclusions
    shapes.append(shape_card(910, "ConclCard", 685800, 2000000, 5200000, 4300000, bg_hex="F8FAFC", border_hex="CBD5E1"))
    left_p = [
        ([ ("🏁 Key Research Conclusions", True, "1E3A8A") ], 14, True, "1E3A8A", False, 10),
        ([ ("1. Two-Branch Architecture is Mandatory", True, "0F172A") ], 12, True, "0F172A", False, 2),
        ([ ("Single-image models fail in real disaster conditions. Siamese difference subtraction eliminates ambient radiometric and shadow noise.", False, "475569") ], 10.5, False, "475569", False, 8),
        ([ ("2. Cross-Attention Solves Intermediate Tiers", True, "0F172A") ], 12, True, "0F172A", False, 2),
        ([ ("Spatial cross-attention between post queries and pre keys drives a +29.8% surge in Minor damage discrimination (0.442 -> 0.740).", False, "475569") ], 10.5, False, "475569", False, 8),
        ([ ("3. Bridging AI to Frontline Life-Safety", True, "0F172A") ], 12, True, "0F172A", False, 2),
        ([ ("Successfully connected live USGS/NASA telemetry and START triage directly to interactive zero-overhead Streamlit dashboard.", False, "475569") ], 10.5, False, "475569", False, 0),
    ]
    shapes.append(text_box(911, "ConclText", 685800, 2000000, 5200000, 4300000, left_p, margin=200000))

    # Right: Future Scope
    shapes.append(shape_card(920, "FutureCard", 6305800, 2000000, 5200000, 4300000, bg_hex="EFF6FF", border_hex="93C5FD"))
    right_p = [
        ([ ("🔮 Future Research Directions", True, "1E40AF") ], 14, True, "1E40AF", False, 10),
        ([ ("1. All-Weather SAR Fusion (Sentinel-1)", True, "0F172A") ], 12, True, "0F172A", False, 2),
        ([ ("Combine optical imagery with Synthetic Aperture Radar (SAR) to penetrate thick hurricane clouds and wildfire smoke plumes.", False, "475569") ], 10.5, False, "475569", False, 8),
        ([ ("2. Tactical Drone Edge Deployment", True, "0F172A") ], 12, True, "0F172A", False, 2),
        ([ ("Quantize model via TensorRT/ONNX for real-time video stream inference directly on search-and-rescue UAV payloads.", False, "475569") ], 10.5, False, "475569", False, 8),
        ([ ("3. Autonomous Evacuation Routing", True, "0F172A") ], 12, True, "0F172A", False, 2),
        ([ ("Couple structural damage heatmaps with road network graphs to calculate safest civilian evacuation corridors in real time.", False, "475569") ], 10.5, False, "475569", False, 0),
    ]
    shapes.append(text_box(921, "FutureText", 6305800, 2000000, 5200000, 4300000, right_p, margin=200000))

    return "".join(shapes)

def build_slide_10():
    """Slide 10: References (Mandatory Last Slide per User Instruction)"""
    shapes = [slide_header(1000, "9. Bibliography", "Key Academic & Technical References", "Formal peer-reviewed publications and standard operational disaster frameworks.")]

    # Left Column: Computer Vision & Remote Sensing Literature
    shapes.append(shape_card(1010, "RefCol1", 685800, 2000000, 5200000, 4300000, bg_hex="F8FAFC", border_hex="CBD5E1"))
    left_p = [
        ([ ("📚 Remote Sensing & Deep Learning Literature", True, "1E3A8A") ], 13, True, "1E3A8A", False, 10),
        ([ ("[1] R. Gupta et al., ", True, "0F172A"), ("'Creating xBD: A Dataset for Assessing Building Damage from Satellite Imagery,' in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. Workshops (CVPRW), 2019, pp. 10–17.", False, "475569") ], 10, False, "475569", False, 8),
        ([ ("[2] J. Bromley et al., ", True, "0F172A"), ("'Signature Verification using a Siamese Time Delay Neural Network,' in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 1993, pp. 737–744.", False, "475569") ], 10, False, "475569", False, 8),
        ([ ("[3] A. Vaswani et al., ", True, "0F172A"), ("'Attention Is All You Need,' in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2017, pp. 5998–6008.", False, "475569") ], 10, False, "475569", False, 8),
        ([ ("[4] M. S. Rahman et al., ", True, "0F172A"), ("'RescueNet: High-Resolution Post-Disaster Building Damage Assessment,' IEEE Trans. Geosci. Remote Sens., vol. 60, pp. 1–18, 2022.", False, "475569") ], 10, False, "475569", False, 0),
    ]
    shapes.append(text_box(1011, "RefText1", 685800, 2000000, 5200000, 4300000, left_p, margin=180000))

    # Right Column: Earth Science APIs & Clinical Disaster Medicine
    shapes.append(shape_card(1020, "RefCol2", 6305800, 2000000, 5200000, 4300000, bg_hex="EFF6FF", border_hex="93C5FD"))
    right_p = [
        ([ ("🌐 Earth Science APIs & Disaster Medicine", True, "1E40AF") ], 13, True, "1E40AF", False, 10),
        ([ ("[5] United States Geological Survey (USGS), ", True, "0F172A"), ("'Real-Time Earthquake Hazards Program GeoJSON Feed,' 2026. [Online]. Available: https://earthquake.usgs.gov/earthquakes/feed/.", False, "475569") ], 10, False, "475569", False, 8),
        ([ ("[6] NASA Earth Science Data, ", True, "0F172A"), ("'Earth Observatory Natural Event Tracker (EONET v3) API,' 2026. [Online]. Available: https://eonet.gsfc.nasa.gov/.", False, "475569") ], 10, False, "475569", False, 8),
        ([ ("[7] M. S. Sever & R. Vanholder, ", True, "0F172A"), ("'Management of Crush Casualties in Disasters,' New England Journal of Medicine, vol. 354, no. 10, pp. 1052–1063, 2006.", False, "475569") ], 10, False, "475569", False, 8),
        ([ ("[8] FEMA, ", True, "0F172A"), ("'Urban Search and Rescue (USAR) Field Operations Guide,' Federal Emergency Management Agency, Washington, D.C., USA, 2022.", False, "475569") ], 10, False, "475569", False, 0),
    ]
    shapes.append(text_box(1021, "RefText2", 6305800, 2000000, 5200000, 4300000, right_p, margin=180000))

    return "".join(shapes)

# ==============================================================================
# MAIN PPTX PACKAGE BUILDER
# ==============================================================================

def generate_presentation(dest_pptx: Path):
    print("Building 10-slide presentation package...")
    
    # 1. Content Types
    content_types = ['''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="png" ContentType="image/png"/>
  <Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
  <Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>
  <Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>
  <Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>''']
    for s_idx in range(1, 11):
        content_types.append(f'<Override PartName="/ppt/slides/slide{s_idx}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>')
    content_types.append('</Types>')

    # 2. Root _rels/.rels
    root_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
</Relationships>'''

    # 3. ppt/presentation.xml
    pres_sld_ids = []
    for s_idx in range(1, 11):
        pres_sld_ids.append(f'<p:sldId id="{255 + s_idx}" r:id="rId{s_idx + 1}"/>')
        
    pres_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
                xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
                xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:sldMasterIdLst>
    <p:sldMasterId id="2147483648" r:id="rId1"/>
  </p:sldMasterIdLst>
  <p:sldIdLst>
    {"".join(pres_sld_ids)}
  </p:sldIdLst>
  <p:sldSz cx="{SLIDE_W}" cy="{SLIDE_H}" type="screen16x9"/>
  <p:notesSz cx="6858000" cy="9144000"/>
</p:presentation>'''

    # 4. ppt/_rels/presentation.xml.rels
    pres_rels_items = ['<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>']
    for s_idx in range(1, 11):
        pres_rels_items.append(f'<Relationship Id="rId{s_idx + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{s_idx}.xml"/>')
    pres_rels_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  {"".join(pres_rels_items)}
</Relationships>'''

    # 5. Master & Layout XML
    master_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
             xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
             xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
    </p:spTree>
  </p:cSld>
  <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
  <p:sldLayoutIdLst>
    <p:sldLayoutId id="2147483649" r:id="rId1"/>
  </p:sldLayoutIdLst>
</p:sldMaster>'''

    master_rels_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>
</Relationships>'''

    layout_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
             xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
             xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
             type="blank" preserve="1">
  <p:cSld>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sldLayout>'''

    layout_rels_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>
</Relationships>'''

    theme_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="DisasterVision Theme">
  <a:themeElements>
    <a:clrScheme name="DisasterVisionColors">
      <a:dk1><a:srgbClr val="0F172A"/></a:dk1>
      <a:lt1><a:srgbClr val="FFFFFF"/></a:lt1>
      <a:dk2><a:srgbClr val="1E293B"/></a:dk2>
      <a:lt2><a:srgbClr val="F8FAFC"/></a:lt2>
      <a:accent1><a:srgbClr val="1E3A8A"/></a:accent1>
      <a:accent2><a:srgbClr val="2563EB"/></a:accent2>
      <a:accent3><a:srgbClr val="38BDF8"/></a:accent3>
      <a:accent4><a:srgbClr val="10B981"/></a:accent4>
      <a:accent5><a:srgbClr val="F59E0B"/></a:accent5>
      <a:accent6><a:srgbClr val="EF4444"/></a:accent6>
      <a:hlink><a:srgbClr val="2563EB"/></a:hlink>
      <a:folHlink><a:srgbClr val="1D4ED8"/></a:folHlink>
    </a:clrScheme>
    <a:fontScheme name="DisasterVisionFonts">
      <a:majorFont><a:latin typeface="Calibri"/></a:majorFont>
      <a:minorFont><a:latin typeface="Calibri"/></a:minorFont>
    </a:fontScheme>
    <a:fmtScheme name="DisasterVisionFmt">
      <a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst>
      <a:lnStyleLst><a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln><a:ln w="25400"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln><a:ln w="38100"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst>
      <a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst>
      <a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst>
    </a:fmtScheme>
  </a:themeElements>
</a:theme>'''

    # Build individual slide XMLs
    slide_builders = [
        build_slide_1,
        build_slide_2,
        build_slide_3,
        build_slide_4,
        build_slide_5,
        build_slide_6,
        build_slide_7,
        build_slide_8,
        build_slide_9,
        build_slide_10,
    ]

    slide_xml_list = []
    slide_rels_list = []
    
    cm_img_path = Path("models_saved/dual_branch_confusion_matrix.png")
    embed_cm = cm_img_path.exists()

    for idx, builder in enumerate(slide_builders, start=1):
        content_shapes = builder()
        sld_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
       xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
       xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
      {content_shapes}
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>'''
        slide_xml_list.append(sld_xml)

        # Slide relationships
        if idx == 8 and embed_cm:
            # Slide 8 has embedded confusion matrix figure
            s_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
  <Relationship Id="rIdCmImg" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/cm_image.png"/>
</Relationships>'''
        else:
            s_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
</Relationships>'''
        slide_rels_list.append(s_rels)

    # Package into .pptx ZIP archive
    print(f"Packaging {dest_pptx}...")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", "".join(content_types))
        z.writestr("_rels/.rels", root_rels)
        z.writestr("ppt/presentation.xml", pres_xml)
        z.writestr("ppt/_rels/presentation.xml.rels", pres_rels_xml)
        z.writestr("ppt/slideMasters/slideMaster1.xml", master_xml)
        z.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", master_rels_xml)
        z.writestr("ppt/slideLayouts/slideLayout1.xml", layout_xml)
        z.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", layout_rels_xml)
        z.writestr("ppt/theme/theme1.xml", theme_xml)
        
        for s_idx in range(1, 11):
            z.writestr(f"ppt/slides/slide{s_idx}.xml", slide_xml_list[s_idx - 1])
            z.writestr(f"ppt/slides/_rels/slide{s_idx}.xml.rels", slide_rels_list[s_idx - 1])
            
        # Write media image if present
        if embed_cm:
            with open(cm_img_path, "rb") as im_f:
                z.writestr("ppt/media/cm_image.png", im_f.read())
            print(f"  Embedded confusion matrix into ppt/media/cm_image.png")

    with open(dest_pptx, "wb") as f:
        f.write(buf.getvalue())
        
    print(f"SUCCESS: Generated {dest_pptx} ({dest_pptx.stat().st_size / 1024:.1f} KB, exactly 10 slides)")

if __name__ == "__main__":
    out_file = Path("DisasterVision_Presentation.pptx")
    generate_presentation(out_file)
