import os
import re
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def clean_markdown_for_pdf(text: str) -> str:
    """Basic cleaner to convert common markdown tags to HTML tags accepted by ReportLab."""
    # Convert bold **text** to <b>text</b>
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    # Convert italic *text* to <i>text</i>
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
    # Convert markdown links [text](url) to text (url)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<u>\1</u> (\2)', text)
    return text

def export_report_to_pdf(report_markdown: str, output_path: str = "research_report.pdf") -> str:
    """Generates a styled, publication-ready PDF from markdown report text."""
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=14
    )
    
    h2_style = ParagraphStyle(
        'DocH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#2563EB'),
        spaceBefore=12,
        spaceAfter=6
    )
    
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor('#334155'),
        spaceAfter=6
    )

    story = []
    lines = report_markdown.split("\n")
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            story.append(Spacer(1, 4))
            continue
            
        if stripped.startswith("# "):
            clean_text = clean_markdown_for_pdf(stripped[2:])
            story.append(Paragraph(clean_text, title_style))
        elif stripped.startswith("## ") or stripped.startswith("### "):
            clean_text = clean_markdown_for_pdf(stripped.lstrip("#").strip())
            story.append(Paragraph(clean_text, h2_style))
        else:
            clean_text = clean_markdown_for_pdf(stripped)
            try:
                story.append(Paragraph(clean_text, body_style))
            except Exception:
                # Fallback if any unescaped XML/HTML entities exist
                story.append(Paragraph(stripped.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'), body_style))

    doc.build(story)
    return output_path
