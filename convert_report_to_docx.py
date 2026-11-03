"""
Markdown to OpenXML DOCX Converter
Converts PROJECT_REPORT.md into a fully formatted Microsoft Word (.docx) document
without requiring any external packages (uses python's built-in zipfile & xml).
"""

import html
import io
import os
from pathlib import Path
import re
import zipfile

def escape_xml(s: str) -> str:
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace('"', "&quot;")
             .replace("'", "&apos;"))

def parse_inline_runs(text: str) -> str:
    """
    Parses **bold**, *italic*, and `code` into OpenXML <w:r> elements.
    """
    # Regex to tokenize bold, italic, inline code, and normal text
    pattern = re.compile(r'(\*\*[^*]+?\*\*|\*[^*]+?\*|`[^`]+?`|\$[^$]+?\$|[^*`$]+)')
    tokens = pattern.findall(text)
    
    xml_runs = []
    for token in tokens:
        if token.startswith("**") and token.endswith("**") and len(token) >= 4:
            clean = token[2:-2]
            xml_runs.append(f'<w:r><w:rPr><w:b/><w:bCs/></w:rPr><w:t xml:space="preserve">{escape_xml(clean)}</w:t></w:r>')
        elif token.startswith("*") and token.endswith("*") and len(token) >= 2:
            clean = token[1:-1]
            xml_runs.append(f'<w:r><w:rPr><w:i/><w:iCs/></w:rPr><w:t xml:space="preserve">{escape_xml(clean)}</w:t></w:r>')
        elif token.startswith("`") and token.endswith("`") and len(token) >= 2:
            clean = token[1:-1]
            xml_runs.append(f'<w:r><w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/><w:sz w:val="19"/><w:color w:val="0F172A"/><w:shd w:val="clear" w:color="auto" w:fill="F1F5F9"/></w:rPr><w:t xml:space="preserve">{escape_xml(clean)}</w:t></w:r>')
        elif token.startswith("$") and token.endswith("$") and len(token) >= 2:
            clean = token[1:-1]
            xml_runs.append(f'<w:r><w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math"/><w:i/><w:color w:val="1E293B"/></w:rPr><w:t xml:space="preserve">{escape_xml(clean)}</w:t></w:r>')
        else:
            if token:
                xml_runs.append(f'<w:r><w:t xml:space="preserve">{escape_xml(token)}</w:t></w:r>')
                
    return "".join(xml_runs)

def build_paragraph(text: str, style: str = "Normal", jc: str = "left", space_before: int = 0, space_after: int = 120) -> str:
    runs = parse_inline_runs(text)
    p_pr = f'<w:pPr><w:pStyle w:val="{style}"/><w:jc w:val="{jc}"/><w:spacing w:before="{space_before}" w:after="{space_after}"/></w:pPr>'
    return f'<w:p>{p_pr}{runs}</w:p>'

def build_table_xml(table_rows: list) -> str:
    """
    Generates OpenXML table with header shading and borders.
    """
    if not table_rows:
        return ""
        
    xml = ['<w:tbl>']
    xml.append('''<w:tblPr>
        <w:tblStyle w:val="TableGrid"/>
        <w:tblW w:w="0" w:type="auto"/>
        <w:jc w:val="center"/>
        <w:tblBorders>
            <w:top w:val="single" w:sz="6" w:space="0" w:color="CBD5E1"/>
            <w:left w:val="none"/>
            <w:bottom w:val="single" w:sz="8" w:space="0" w:color="94A3B8"/>
            <w:right w:val="none"/>
            <w:insideH w:val="single" w:sz="4" w:space="0" w:color="E2E8F0"/>
            <w:insideV w:val="none"/>
        </w:tblBorders>
        <w:tblCellMar>
            <w:top w:w="120" w:type="dxa"/>
            <w:bottom w:w="120" w:type="dxa"/>
            <w:left w:w="160" w:type="dxa"/>
            <w:right w:w="160" w:type="dxa"/>
        </w:tblCellMar>
    </w:tblPr>''')

    is_header = True
    for r_idx, row in enumerate(table_rows):
        xml.append('<w:tr>')
        if is_header:
            xml.append('<w:trPr><w:tblHeader/></w:trPr>')
            
        bg_color = "1E3A8A" if is_header else ("F8FAFC" if r_idx % 2 == 1 else "FFFFFF")
        
        for cell_text in row:
            clean_cell = cell_text.strip()
            # If header, bold and white
            if is_header:
                runs = parse_inline_runs(f"**{clean_cell}**")
                # replace color inside runs with white
                runs = runs.replace('<w:t xml:space="preserve">', '<w:color w:val="FFFFFF"/><w:t xml:space="preserve">')
            else:
                runs = parse_inline_runs(clean_cell)
                
            xml.append(f'''<w:tc>
                <w:tcPr>
                    <w:shd w:val="clear" w:color="auto" w:fill="{bg_color}"/>
                </w:tcPr>
                <w:p>
                    <w:pPr><w:spacing w:before="60" w:after="60"/></w:pPr>
                    {runs}
                </w:p>
            </w:tc>''')
        xml.append('</w:tr>')
        is_header = False
        
    xml.append('</w:tbl>')
    return "".join(xml)

def build_image_drawing(r_id: str, cx: int, cy: int, desc: str) -> str:
    """
    Builds OpenXML drawing element for embedding an image.
    cx, cy in EMUs (1 inch = 914400 EMUs)
    e.g. 5.5 inches wide = 5029200
    """
    return f'''<w:p>
        <w:pPr><w:jc w:val="center"/><w:spacing w:before="180" w:after="80"/></w:pPr>
        <w:r>
            <w:drawing>
                <wp:inline distT="0" distB="0" distL="0" distR="0" xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing">
                    <wp:extent cx="{cx}" cy="{cy}"/>
                    <wp:effectExtent l="0" t="0" r="0" b="0"/>
                    <wp:docPr id="1" name="{desc}"/>
                    <wp:cNvGraphicFramePr>
                        <a:graphicFrameLocks xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" noChangeAspect="1"/>
                    </wp:cNvGraphicFramePr>
                    <a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
                        <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
                            <pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">
                                <pic:nvPicPr>
                                    <pic:cNvPr id="0" name="{desc}"/>
                                    <pic:cNvPicPr/>
                                </pic:nvPicPr>
                                <pic:blipFill>
                                    <a:blip r:embed="{r_id}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"/>
                                    <a:stretch><a:fillRect/></a:stretch>
                                </pic:blipFill>
                                <pic:spPr>
                                    <a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>
                                    <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
                                </pic:spPr>
                            </pic:pic>
                        </a:graphicData>
                    </a:graphic>
                </wp:inline>
            </w:drawing>
        </w:r>
    </w:p>'''

def convert_markdown_to_docx(md_path: Path, docx_path: Path):
    print(f"Reading markdown from {md_path}...")
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.split("\n")
    body_elements = []
    
    in_table = False
    table_rows = []
    in_code_block = False
    code_lines = []

    # Check for images to embed
    img1_path = Path("models_saved/dual_branch_confusion_matrix.png")
    img2_path = Path("models_saved/dual_branch_metrics_bar.png")
    
    images_to_embed = {}
    if img1_path.exists():
        images_to_embed["rIdImg1"] = (img1_path, "media/image1.png")
    if img2_path.exists():
        images_to_embed["rIdImg2"] = (img2_path, "media/image2.png")

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        
        # Check code block fences
        if line.startswith("```"):
            if in_code_block:
                # Close code block
                in_code_block = False
                code_text = "\n".join(code_lines)
                body_elements.append(f'''<w:p>
                    <w:pPr>
                        <w:pStyle w:val="CodeBlock"/>
                        <w:spacing w:before="120" w:after="160"/>
                        <w:shd w:val="clear" w:color="auto" w:fill="F1F5F9"/>
                        <w:pBdr>
                            <w:left w:val="single" w:sz="18" w:space="15" w:color="3B82F6"/>
                        </w:pBdr>
                    </w:pPr>
                    <w:r>
                        <w:rPr>
                            <w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/>
                            <w:sz w:val="18"/>
                            <w:color w:val="0F172A"/>
                        </w:rPr>
                        <w:t xml:space="preserve">{escape_xml(code_text)}</w:t>
                    </w:r>
                </w:p>''')
                code_lines = []
            else:
                in_code_block = True
                code_lines = []
            i += 1
            continue

        if in_code_block:
            code_lines.append(line)
            i += 1
            continue

        # Check tables
        if line.startswith("|") and line.endswith("|"):
            # Check if separator line e.g. |---|---|
            if set(line.replace("|", "").replace(" ", "").replace("-", "").replace(":", "")) == set():
                i += 1
                continue
                
            # Parse row
            cells = [c.strip() for c in line.split("|")[1:-1]]
            if cells:
                table_rows.append(cells)
            in_table = True
            i += 1
            continue
        else:
            if in_table:
                # Flush table
                body_elements.append(build_table_xml(table_rows))
                body_elements.append('<w:p><w:pPr><w:spacing w:before="60" w:after="120"/></w:pPr></w:p>')
                table_rows = []
                in_table = False

        # Empty lines
        if not line:
            i += 1
            continue

        # Horizontal rule
        if line.startswith("---") or line.startswith("***"):
            body_elements.append('<w:p><w:pPr><w:pBdr><w:bottom w:val="single" w:sz="12" w:space="8" w:color="E2E8F0"/></w:pBdr><w:spacing w:before="120" w:after="180"/></w:pPr></w:p>')
            i += 1
            continue

        # Headings
        if line.startswith("# "):
            title_text = line[2:].strip()
            body_elements.append(build_paragraph(title_text, style="Title", jc="center", space_before=240, space_after=180))
            i += 1
            continue
        elif line.startswith("## "):
            h1_text = line[3:].strip()
            body_elements.append(build_paragraph(h1_text, style="Heading1", jc="left", space_before=320, space_after=120))
            i += 1
            continue
        elif line.startswith("### "):
            h2_text = line[4:].strip()
            body_elements.append(build_paragraph(h2_text, style="Heading2", jc="left", space_before=240, space_after=80))
            i += 1
            continue
        elif line.startswith("#### "):
            h3_text = line[5:].strip()
            body_elements.append(build_paragraph(h3_text, style="Heading3", jc="left", space_before=180, space_after=60))
            i += 1
            continue

        # Blockquotes
        if line.startswith("> "):
            quote_text = line[2:].strip()
            body_elements.append(f'''<w:p>
                <w:pPr>
                    <w:spacing w:before="100" w:after="100"/>
                    <w:ind w:left="360" w:right="360"/>
                    <w:shd w:val="clear" w:color="auto" w:fill="EFF6FF"/>
                    <w:pBdr>
                        <w:left w:val="single" w:sz="24" w:space="15" w:color="2563EB"/>
                    </w:pBdr>
                </w:pPr>
                {parse_inline_runs(quote_text)}
            </w:p>''')
            i += 1
            continue

        # Bullet lists
        if line.startswith("* ") or line.startswith("- "):
            bullet_text = line[2:].strip()
            body_elements.append(f'''<w:p>
                <w:pPr>
                    <w:pStyle w:val="ListBullet"/>
                    <w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr>
                    <w:spacing w:before="40" w:after="60"/>
                    <w:ind w:left="480" w:hanging="240"/>
                </w:pPr>
                <w:r><w:rPr><w:rFonts w:ascii="Symbol" w:hAnsi="Symbol"/><w:color w:val="2563EB"/></w:rPr><w:t>&#x2022; </w:t></w:r>
                {parse_inline_runs(bullet_text)}
            </w:p>''')
            i += 1
            continue

        # Numbered lists
        m_num = re.match(r'^(\d+)\.\s+(.*)$', line)
        if m_num:
            num_str = m_num.group(1)
            num_text = m_num.group(2).strip()
            body_elements.append(f'''<w:p>
                <w:pPr>
                    <w:spacing w:before="40" w:after="60"/>
                    <w:ind w:left="480" w:hanging="300"/>
                </w:pPr>
                <w:r><w:rPr><w:b/><w:color w:val="1E3A8A"/></w:rPr><w:t>{num_str}. </w:t></w:r>
                {parse_inline_runs(num_text)}
            </w:p>''')
            i += 1
            continue

        # Regular paragraph
        body_elements.append(build_paragraph(line, style="Normal", jc="both", space_before=0, space_after=120))

        # Check if we should insert figures after certain sections
        if "TABLE VI. $4 \\times 4$ Per-Severity Confusion Matrix" in line and "rIdImg1" in images_to_embed:
            # 5 inches width = 4572000 EMUs, 4 inches height = 3657600 EMUs
            body_elements.append(build_image_drawing("rIdImg1", 4572000, 3657600, "Figure 2: Confusion Matrix"))
            body_elements.append(build_paragraph("*Fig. 2. Per-Severity Confusion Matrix on xBD Testbed.*", style="Normal", jc="center", space_before=40, space_after=180))

        if "TABLE V. Benchmark Performance Comparison" in line and "rIdImg2" in images_to_embed:
            # 5.2 inches width = 4754880 EMUs, 3.2 inches height = 2926080 EMUs
            body_elements.append(build_image_drawing("rIdImg2", 4754880, 2926080, "Figure 3: Benchmark Comparison"))
            body_elements.append(build_paragraph("*Fig. 3. Precision, Recall, and F1 Metrics across Damage Severity Classes.*", style="Normal", jc="center", space_before=40, space_after=180))

        i += 1

    # End-of-file table flush if needed
    if in_table and table_rows:
        body_elements.append(build_table_xml(table_rows))

    # Page setup: Standard Letter/A4 with 1 inch (1440 dxa) margins
    sect_pr = '''<w:sectPr>
        <w:pgSz w:w="12240" w:h="15840"/>
        <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="720" w:footer="720" w:gutter="0"/>
        <w:cols w:space="720"/>
        <w:docGrid w:linePitch="360"/>
    </w:sectPr>'''

    doc_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
            xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
            xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"
            xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
            xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
            xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">
  <w:body>
    {"".join(body_elements)}
    {sect_pr}
  </w:body>
</w:document>'''

    # Build Content_Types.xml
    content_types = ['''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="png" ContentType="image/png"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>''']

    # Build _rels/.rels
    rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>'''

    # Build word/_rels/document.xml.rels
    doc_rels = ['''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>''']
    for r_id, (img_path, target_rel) in images_to_embed.items():
        doc_rels.append(f'<Relationship Id="{r_id}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="{target_rel}"/>')
    doc_rels.append('</Relationships>')

    # Professional styles
    styles_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:docDefaults>
    <w:rPrDefault>
      <w:rPr>
        <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:cs="Calibri"/>
        <w:sz w:val="22"/>
        <w:szCs w:val="22"/>
        <w:color w:val="1E293B"/>
        <w:lang w:val="en-US"/>
      </w:rPr>
    </w:rPrDefault>
    <w:pPrDefault>
      <w:pPr>
        <w:spacing w:line="276" w:lineRule="auto"/>
      </w:pPr>
    </w:pPrDefault>
  </w:docDefaults>

  <!-- Title Style -->
  <w:style w:type="paragraph" w:styleId="Title">
    <w:name w:val="Title"/>
    <w:rPr>
      <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
      <w:b/><w:bCs/>
      <w:sz w:val="38"/>
      <w:color w:val="1E3A8A"/>
    </w:rPr>
  </w:style>

  <!-- Heading 1 -->
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:pPr>
      <w:pBdr><w:bottom w:val="single" w:sz="8" w:space="6" w:color="3B82F6"/></w:pBdr>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
      <w:b/><w:bCs/>
      <w:sz w:val="28"/>
      <w:color w:val="1E3A8A"/>
    </w:rPr>
  </w:style>

  <!-- Heading 2 -->
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="heading 2"/>
    <w:rPr>
      <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
      <w:b/><w:bCs/>
      <w:sz w:val="25"/>
      <w:color w:val="2563EB"/>
    </w:rPr>
  </w:style>

  <!-- Heading 3 -->
  <w:style w:type="paragraph" w:styleId="Heading3">
    <w:name w:val="heading 3"/>
    <w:rPr>
      <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
      <w:b/><w:bCs/>
      <w:sz w:val="23"/>
      <w:color w:val="334155"/>
    </w:rPr>
  </w:style>

  <!-- Normal Body -->
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:rPr>
      <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
      <w:sz w:val="22"/>
      <w:color w:val="1E293B"/>
    </w:rPr>
  </w:style>
</w:styles>'''

    # Package into .docx zip
    print(f"Creating OpenXML DOCX package at {docx_path}...")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", "".join(content_types))
        z.writestr("_rels/.rels", rels)
        z.writestr("word/_rels/document.xml.rels", "".join(doc_rels))
        z.writestr("word/styles.xml", styles_xml)
        z.writestr("word/document.xml", doc_xml)
        
        # Write embedded images
        for r_id, (img_file, target_rel) in images_to_embed.items():
            if img_file.exists():
                with open(img_file, "rb") as im_f:
                    z.writestr(f"word/{target_rel}", im_f.read())
                print(f"  Embedded {img_file} as word/{target_rel}")

    with open(docx_path, "wb") as f:
        f.write(buf.getvalue())
        
    print(f"SUCCESS: Successfully generated Word document: {docx_path} ({docx_path.stat().st_size / 1024:.1f} KB)")

if __name__ == "__main__":
    src_md = Path("PROJECT_REPORT.md")
    dest_docx = Path("PROJECT_REPORT.docx")
    convert_markdown_to_docx(src_md, dest_docx)
