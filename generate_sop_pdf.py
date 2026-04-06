"""
Generate SOP/Procedure PDF for FDA MAUDE Data Processing & Analysis Platform.
"""
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, ListFlowable, ListItem, HRFlowable
)

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "SOP_MAUDE_Platform_Operating_Procedure.pdf")

# ── Styles ────────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

style_cover_title = ParagraphStyle(
    'CoverTitle', parent=styles['Title'],
    fontSize=26, leading=32, textColor=HexColor('#1E40AF'),
    spaceAfter=12, alignment=TA_CENTER,
)
style_cover_sub = ParagraphStyle(
    'CoverSub', parent=styles['Normal'],
    fontSize=14, leading=18, textColor=HexColor('#475569'),
    spaceAfter=6, alignment=TA_CENTER,
)
style_h1 = ParagraphStyle(
    'H1', parent=styles['Heading1'],
    fontSize=18, leading=22, textColor=HexColor('#1E3A5F'),
    spaceBefore=20, spaceAfter=10,
    borderWidth=0, borderPadding=0,
)
style_h2 = ParagraphStyle(
    'H2', parent=styles['Heading2'],
    fontSize=14, leading=18, textColor=HexColor('#2563EB'),
    spaceBefore=14, spaceAfter=6,
)
style_h3 = ParagraphStyle(
    'H3', parent=styles['Heading3'],
    fontSize=12, leading=15, textColor=HexColor('#1E40AF'),
    spaceBefore=10, spaceAfter=4,
)
style_body = ParagraphStyle(
    'Body', parent=styles['Normal'],
    fontSize=10, leading=14, textColor=HexColor('#1F2937'),
    spaceAfter=6, alignment=TA_JUSTIFY,
)
style_body_bold = ParagraphStyle(
    'BodyBold', parent=style_body,
    fontName='Helvetica-Bold',
)
style_bullet = ParagraphStyle(
    'Bullet', parent=style_body,
    leftIndent=24, bulletIndent=12,
    spaceAfter=3,
)
style_sub_bullet = ParagraphStyle(
    'SubBullet', parent=style_body,
    leftIndent=44, bulletIndent=32,
    spaceAfter=2, fontSize=9.5,
)
style_note = ParagraphStyle(
    'Note', parent=style_body,
    fontSize=9.5, leading=13,
    textColor=HexColor('#6B7280'),
    leftIndent=12, borderColor=HexColor('#CBD5E1'),
    borderWidth=1, borderPadding=6,
    backColor=HexColor('#F8FAFC'),
)
style_table_header = ParagraphStyle(
    'TableHeader', parent=styles['Normal'],
    fontSize=9.5, leading=12, fontName='Helvetica-Bold',
    textColor=HexColor('#FFFFFF'),
)
style_table_cell = ParagraphStyle(
    'TableCell', parent=styles['Normal'],
    fontSize=9, leading=12, textColor=HexColor('#1F2937'),
)

# ── Helpers ───────────────────────────────────────────────────────────────────
def hr():
    return HRFlowable(width="100%", thickness=1, color=HexColor('#CBD5E1'),
                       spaceBefore=6, spaceAfter=6)

def bullet(text):
    return Paragraph(f"\u2022  {text}", style_bullet)

def sub_bullet(text):
    return Paragraph(f"\u2013  {text}", style_sub_bullet)

def note(text):
    return Paragraph(f"<b>Note:</b> {text}", style_note)

def para(text):
    return Paragraph(text, style_body)

def bold_para(text):
    return Paragraph(text, style_body_bold)

def heading1(text):
    return Paragraph(text, style_h1)

def heading2(text):
    return Paragraph(text, style_h2)

def heading3(text):
    return Paragraph(text, style_h3)

def spacer(h=0.15):
    return Spacer(1, h * inch)

def make_table(headers, rows, col_widths=None):
    """Create a styled table with header row."""
    hdr = [Paragraph(h, style_table_header) for h in headers]
    data = [hdr]
    for row in rows:
        data.append([Paragraph(str(c), style_table_cell) for c in row])
    if col_widths is None:
        col_widths = [None] * len(headers)
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1E40AF')),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#FFFFFF')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9.5),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('TOPPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), HexColor('#FFFFFF')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#FFFFFF'), HexColor('#F1F5F9')]),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#CBD5E1')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 1), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 5),
    ]))
    return t


# ── Build Document ────────────────────────────────────────────────────────────
def build():
    doc = SimpleDocTemplate(
        OUTPUT_PATH, pagesize=letter,
        leftMargin=0.75*inch, rightMargin=0.75*inch,
        topMargin=0.75*inch, bottomMargin=0.75*inch,
        title="SOP - FDA MAUDE Data Processing & Analysis Platform",
        author="FDA MAUDE Platform Team",
    )
    story = []
    W = doc.width  # usable width

    # ══════════════════════════════════════════════════════════════════════════
    # COVER PAGE
    # ══════════════════════════════════════════════════════════════════════════
    story.append(Spacer(1, 1.5*inch))
    story.append(Paragraph("STANDARD OPERATING PROCEDURE", style_cover_sub))
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph("FDA MAUDE Data Processing<br/>&amp; Analysis Platform", style_cover_title))
    story.append(Spacer(1, 0.3*inch))
    story.append(HRFlowable(width="60%", thickness=2, color=HexColor('#2563EB'),
                             spaceBefore=6, spaceAfter=6))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("Operating Procedure &amp; User Guide", style_cover_sub))
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph("Document Version: 1.0", style_cover_sub))
    story.append(Spacer(1, 0.8*inch))

    cover_table_data = [
        ["Prepared By:", "FDA MAUDE Platform Team"],
        ["Department:", "Regulatory / Postmarket Surveillance"],
        ["Application:", "Web-Based (Flask / Firebase)"],
        ["Data Source:", "FDA openFDA MAUDE Database &amp; IMDRF Annexes A\u2013G"],
    ]
    ct = Table(cover_table_data, colWidths=[2*inch, 4*inch])
    ct.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('TEXTCOLOR', (0, 0), (0, -1), HexColor('#475569')),
        ('TEXTCOLOR', (1, 0), (1, -1), HexColor('#1F2937')),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ]))
    story.append(ct)
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # TABLE OF CONTENTS
    # ══════════════════════════════════════════════════════════════════════════
    story.append(heading1("Table of Contents"))
    story.append(hr())
    toc_items = [
        ("1.", "Purpose &amp; Scope"),
        ("2.", "System Overview"),
        ("3.", "User Authentication (Login / Register / Password Reset)"),
        ("4.", "Dashboard &amp; Navigation"),
        ("5.", "Module 1 \u2014 MAUDE Data Pipeline (Fetch, Clean, Analyze)"),
        ("6.", "Module 2 \u2014 File Upload &amp; Processing"),
        ("7.", "Module 3 \u2014 IMDRF Insights &amp; Analysis"),
        ("8.", "Module 4 \u2014 Device Recall Search"),
        ("9.", "Module 5 \u2014 TXT to CSV Converter"),
        ("10.", "Module 6 \u2014 CSV Viewer"),
        ("11.", "Backend Processing Pipeline (Technical Details)"),
        ("12.", "Data Flow Diagram"),
        ("13.", "Glossary"),
    ]
    for num, title in toc_items:
        story.append(Paragraph(f"<b>{num}</b>&nbsp;&nbsp;{title}", style_body))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # 1. PURPOSE & SCOPE
    # ══════════════════════════════════════════════════════════════════════════
    story.append(heading1("1. Purpose &amp; Scope"))
    story.append(hr())
    story.append(para(
        "This Standard Operating Procedure (SOP) describes the step-by-step operation of the "
        "<b>FDA MAUDE Data Processing &amp; Analysis Platform</b>. The platform is a web-based tool "
        "designed for regulatory professionals, postmarket surveillance teams, and quality engineers "
        "to fetch, clean, analyze, and visualize FDA Medical Device Adverse Event (MAUDE) data."
    ))
    story.append(spacer())
    story.append(bold_para("This SOP covers:"))
    story.append(bullet("User authentication and navigation workflow"))
    story.append(bullet("Fetching raw MAUDE data from the openFDA API"))
    story.append(bullet("Automated data cleaning and standardization"))
    story.append(bullet("IMDRF code mapping and insights analysis"))
    story.append(bullet("Device recall search"))
    story.append(bullet("Utility tools (TXT-to-CSV converter, CSV Viewer)"))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # 2. SYSTEM OVERVIEW
    # ══════════════════════════════════════════════════════════════════════════
    story.append(heading1("2. System Overview"))
    story.append(hr())
    story.append(para(
        "The platform is a <b>Flask web application</b> with Firebase authentication. "
        "It provides a modern, responsive UI with dark/light theme support."
    ))
    story.append(spacer())
    story.append(heading2("2.1 Technology Stack"))
    story.append(make_table(
        ["Component", "Technology"],
        [
            ["Backend", "Python / Flask"],
            ["Authentication", "Firebase Auth (Email/Password + Google OAuth)"],
            ["Data Source", "FDA openFDA REST API (device/event, device/recall, device/classification)"],
            ["Data Processing", "Pandas, OpenPyXL"],
            ["AI/ML Fallback", "Groq LLM API (for column identification &amp; IMDRF mapping fallback)"],
            ["IMDRF Reference", "IMDRF Annexes A\u2013G consolidated Excel file"],
            ["Visualization", "Plotly.js (interactive charts)"],
            ["PDF Reports", "ReportLab"],
            ["Hosting", "Render.com (production) / Local development"],
        ],
        col_widths=[1.8*inch, 5*inch],
    ))
    story.append(spacer())
    story.append(heading2("2.2 Pages / Modules"))
    story.append(make_table(
        ["Page", "URL Path", "Purpose"],
        [
            ["Login", "/login", "User sign-in (email or Google)"],
            ["Register", "/register", "New account creation"],
            ["Forgot Password", "/forgot-password", "Password reset via email"],
            ["Dashboard (Home)", "/", "Main page \u2014 pipeline controls, upload, bulk download"],
            ["IMDRF Insights", "/imdrf-insights", "IMDRF code analysis, trend charts, PDF reports"],
            ["Device Recall", "/device-recall", "Search FDA device recall records"],
            ["TXT to CSV", "/txt-to-csv", "Convert pipe-delimited MAUDE TXT files to CSV"],
            ["CSV Viewer", "/csv-viewer", "View &amp; explore large CSV/Excel files in-browser"],
        ],
        col_widths=[1.3*inch, 1.5*inch, 4*inch],
    ))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # 3. AUTHENTICATION
    # ══════════════════════════════════════════════════════════════════════════
    story.append(heading1("3. User Authentication"))
    story.append(hr())

    story.append(heading2("3.1 Login Page (/login)"))
    story.append(para("The login page is the entry point. All other pages require authentication."))
    story.append(spacer())
    story.append(heading3("Step-by-Step: Email/Password Login"))
    story.append(bullet("Navigate to the application URL. If not logged in, you are redirected to <b>/login</b>."))
    story.append(bullet("Enter your <b>Email Address</b> and <b>Password</b> in the respective fields."))
    story.append(bullet("Click the <b>\"Sign In\"</b> button."))
    story.append(bullet("On success, a green alert shows <i>\"Login successful! Redirecting...\"</i> and you are taken to the Dashboard."))
    story.append(bullet("On failure, a red alert shows the specific error (e.g., \"Invalid email or password\")."))
    story.append(spacer())
    story.append(heading3("Step-by-Step: Google Sign-In"))
    story.append(bullet("Click the <b>\"Sign in with Google\"</b> button below the divider."))
    story.append(bullet("A Google sign-in popup appears. Select your Google account."))
    story.append(bullet("On success, you are redirected to the Dashboard."))
    story.append(spacer())
    story.append(heading3("Additional Links on Login Page"))
    story.append(bullet("<b>\"Forgot Password?\"</b> \u2014 Takes you to the password reset page."))
    story.append(bullet("<b>\"Sign up\"</b> \u2014 Takes you to the registration page."))
    story.append(bullet("<b>Theme Toggle</b> (moon/sun icon, top-right) \u2014 Switches between dark and light mode."))

    story.append(spacer(0.3))
    story.append(heading2("3.2 Registration Page (/register)"))
    story.append(para("New users create an account here."))
    story.append(heading3("Step-by-Step"))
    story.append(bullet("Enter <b>Full Name</b>, <b>Email Address</b>, <b>Password</b> (min 8 characters), and <b>Confirm Password</b>."))
    story.append(bullet("Click <b>\"Create Account\"</b>."))
    story.append(bullet("Alternatively, click <b>\"Sign up with Google\"</b> for one-click registration."))
    story.append(bullet("On success, you are automatically logged in and redirected to the Dashboard."))
    story.append(note("Passwords must be at least 8 characters. Firebase enforces this server-side."))

    story.append(spacer(0.3))
    story.append(heading2("3.3 Forgot Password Page (/forgot-password)"))
    story.append(bullet("Enter your registered <b>Email Address</b>."))
    story.append(bullet("Click <b>\"Send Reset Link\"</b>."))
    story.append(bullet("Firebase sends a password reset email to the address (if an account exists)."))
    story.append(bullet("Click <b>\"Back to Login\"</b> to return."))
    story.append(note("For security, the system shows a generic success message even if the email is not registered."))

    story.append(spacer(0.3))
    story.append(heading2("3.4 Logout"))
    story.append(para(
        "Click the <b>\"Logout\"</b> button visible in the top-right user menu on any authenticated page. "
        "This clears your session and redirects to the Login page."
    ))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # 4. DASHBOARD & NAVIGATION
    # ══════════════════════════════════════════════════════════════════════════
    story.append(heading1("4. Dashboard &amp; Navigation"))
    story.append(hr())
    story.append(para(
        "After login, you land on the <b>Dashboard (Home Page)</b> at <b>/</b>. This is the central hub "
        "for all data operations."
    ))
    story.append(spacer())
    story.append(heading2("4.1 Top Bar / User Menu"))
    story.append(bullet("<b>User Avatar &amp; Name</b> \u2014 Shows your display name and email."))
    story.append(bullet("<b>Theme Toggle</b> (moon/sun icon) \u2014 Switch dark/light mode. Persists across sessions."))
    story.append(bullet("<b>Logout Button</b> \u2014 Ends your session."))
    story.append(spacer())

    story.append(heading2("4.2 Navigation Buttons"))
    story.append(para("Navigation buttons appear in the top bar on every page (except Login/Register):"))
    story.append(make_table(
        ["Button", "Icon", "Navigates To", "Purpose"],
        [
            ["Home / Dashboard", "fa-home", "/", "Main pipeline &amp; upload page"],
            ["IMDRF Insights", "fa-chart-bar", "/imdrf-insights", "IMDRF code analysis &amp; charting"],
            ["Device Recall", "fa-exclamation-triangle", "/device-recall", "FDA device recall search"],
            ["TXT to CSV", "fa-file-csv", "/txt-to-csv", "Convert MAUDE TXT files to CSV"],
            ["CSV Viewer", "fa-table", "/csv-viewer", "View large CSV/Excel files"],
        ],
        col_widths=[1.3*inch, 1.1*inch, 1.3*inch, 3.1*inch],
    ))
    story.append(spacer())
    story.append(heading2("4.3 Dashboard Layout"))
    story.append(para("The Dashboard has two main sections:"))
    story.append(bullet("<b>Main Card (Left)</b> \u2014 Contains the MAUDE Data Pipeline controls and the file upload area."))
    story.append(bullet("<b>Info Sidebar (Right)</b> \u2014 Displays quick-reference cards about data source, processing steps, and supported formats."))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # 5. MODULE 1 — MAUDE DATA PIPELINE
    # ══════════════════════════════════════════════════════════════════════════
    story.append(heading1("5. Module 1 \u2014 MAUDE Data Pipeline"))
    story.append(hr())
    story.append(para(
        "This is the primary module. It fetches adverse event data from the <b>FDA openFDA MAUDE database</b>, "
        "cleans/standardizes it, maps IMDRF codes, and produces downloadable outputs."
    ))

    story.append(heading2("5.1 Input Fields"))
    story.append(make_table(
        ["Field", "Required", "Description"],
        [
            ["Product Code", "Yes", "FDA product code for the device (e.g., LYZ, QBJ, DXY). This is a 3-letter code from the FDA device classification database."],
            ["Date From", "Yes", "Start date for the data range (YYYY-MM-DD format). The pipeline fetches records from this date onward."],
            ["Date To", "Yes", "End date for the data range (YYYY-MM-DD format). The pipeline fetches records up to and including this date."],
        ],
        col_widths=[1.3*inch, 0.8*inch, 4.7*inch],
    ))

    story.append(spacer())
    story.append(heading2("5.2 Output Selection"))
    story.append(para(
        "Before starting the pipeline, select one or more output types. You can select multiple outputs "
        "in a single run \u2014 the pipeline automatically runs to the highest level needed."
    ))
    story.append(make_table(
        ["Output Option", "Color", "What You Get"],
        [
            ["Raw CSV", "Blue", "Unprocessed MAUDE data downloaded directly from openFDA in CSV format. Contains all original fields."],
            ["Cleaned XLSX", "Green", "Processed &amp; standardized Excel file. Columns removed, dates standardized, manufacturers normalized, IMDRF codes mapped."],
            ["IMDRF Code Counts XLSX", "Purple", "Multi-sheet Excel workbook with IMDRF code counts at Level-1/2/3, monthly breakdowns per manufacturer, and patient problem E-codes."],
        ],
        col_widths=[1.5*inch, 0.7*inch, 4.6*inch],
    ))

    story.append(spacer())
    story.append(heading2("5.3 Step-by-Step: Running the Pipeline"))
    story.append(bullet("<b>Step 1:</b> Enter the <b>Product Code</b> (e.g., \"LYZ\")."))
    story.append(bullet("<b>Step 2:</b> Select <b>Date From</b> and <b>Date To</b>."))
    story.append(bullet("<b>Step 3:</b> Click one or more <b>output option cards</b> (Raw / Cleaned / Code Counts). Selected cards show a checkmark and full opacity."))
    story.append(bullet("<b>Step 4:</b> Click the <b>\"Start Pipeline\"</b> button."))
    story.append(spacer())
    story.append(bold_para("What happens after clicking Start:"))
    story.append(bullet("The pipeline progress timeline appears showing the steps."))
    story.append(bullet("A progress bar shows download progress with record counts."))
    story.append(bullet("The status area shows the current step (e.g., \"Connecting to openFDA...\", \"Downloading MAUDE data...\", \"Cleaning data...\")."))
    story.append(bullet("On completion, <b>Download</b> buttons appear for each selected output."))
    story.append(spacer())

    story.append(heading2("5.4 Pipeline Steps (Visual Timeline)"))
    story.append(make_table(
        ["Step", "Name", "Description"],
        [
            ["0", "Probe", "Connects to openFDA and determines the correct search field and date-field combination. Validates that the product code exists and has data in the date range."],
            ["1", "Download", "Fetches all matching MAUDE records page-by-page from openFDA. Uses adaptive windowing (year-level or monthly) to avoid the 26K record cap per query."],
            ["2", "Clean", "Runs the full cleaning pipeline: column removal, date standardization, manufacturer normalization, IMDRF code mapping, patient problem exploding, and validation."],
            ["3", "Code Counts", "Generates IMDRF code count summaries at all 3 levels with monthly breakdowns, per-manufacturer counts, and patient problem E-code analysis."],
        ],
        col_widths=[0.5*inch, 1*inch, 5.3*inch],
    ))

    story.append(spacer())
    story.append(heading2("5.5 Audit Report"))
    story.append(para(
        "After the pipeline completes, an <b>\"Audit Report\"</b> button appears. Clicking it downloads "
        "a plain-text audit report (.txt) containing:"
    ))
    story.append(bullet("Data source details (product code, date range, pipeline type)"))
    story.append(bullet("Total records found and downloaded from openFDA"))
    story.append(bullet("Data cleaning statistics (original rows, final rows, rows removed)"))
    story.append(bullet("Columns removed during cleaning"))
    story.append(bullet("Row removal reasons and counts"))
    story.append(bullet("Manufacturer list after normalization"))
    story.append(bullet("IMDRF mapping statistics (mapped vs. unmapped)"))
    story.append(bullet("Validation results (all checks passed/failed)"))

    story.append(spacer())
    story.append(heading2("5.6 Direct to IMDRF Insights"))
    story.append(para(
        "After a pipeline run that includes the Cleaned output, a button appears to <b>\"Open in IMDRF Insights\"</b>. "
        "This sends the cleaned file directly to the IMDRF Insights page without re-uploading."
    ))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # 6. MODULE 2 — FILE UPLOAD & PROCESSING
    # ══════════════════════════════════════════════════════════════════════════
    story.append(heading1("6. Module 2 \u2014 File Upload &amp; Processing"))
    story.append(hr())
    story.append(para(
        "If you already have a MAUDE data file (CSV or Excel), you can upload it directly "
        "for cleaning and IMDRF code mapping without fetching from openFDA."
    ))
    story.append(spacer())
    story.append(heading2("6.1 Step-by-Step"))
    story.append(bullet("In the Dashboard's <b>Upload section</b>, drag-and-drop or click to select a file."))
    story.append(bullet("Supported formats: <b>CSV</b>, <b>XLSX</b>, <b>XLS</b>."))
    story.append(bullet("Optionally upload a custom <b>IMDRF Annexure file</b> (Excel). If not provided, the bundled \"Annexes A-G consolidated.xlsx\" is used."))
    story.append(bullet("Click <b>\"Process File\"</b>."))
    story.append(bullet("The file is cleaned asynchronously. Poll the status until complete."))
    story.append(bullet("On success, a <b>cleaned XLSX file</b> is available for download."))
    story.append(spacer())
    story.append(heading2("6.2 Validation Checks (Hard Stops)"))
    story.append(para("The processor performs strict validation. A <b>HARD STOP</b> occurs if any of these fail:"))
    story.append(make_table(
        ["Check", "Description"],
        [
            ["Column Count", "Output must have the expected number of columns."],
            ["File Integrity", "Output file must be openable and parseable."],
            ["Date Format", 'All dates must be DD-MM-YYYY. No literal "nan" values allowed.'],
            ["IMDRF Adjacent", "IMDRF Code column must be immediately adjacent to Device Problem column."],
            ["IMDRF Codes Valid", "All IMDRF codes must exist in the loaded Annex file."],
        ],
        col_widths=[1.5*inch, 5.3*inch],
    ))
    story.append(note('A non-critical warning is issued if timestamps (time components) are still present in date fields. This does not block the output.'))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # 7. MODULE 3 — IMDRF INSIGHTS
    # ══════════════════════════════════════════════════════════════════════════
    story.append(heading1("7. Module 3 \u2014 IMDRF Insights &amp; Analysis"))
    story.append(hr())
    story.append(para(
        "The IMDRF Insights page (<b>/imdrf-insights</b>) provides deep analytical capabilities "
        "on cleaned MAUDE data using IMDRF code classifications."
    ))

    story.append(heading2("7.1 Getting Started"))
    story.append(bullet("<b>Option A:</b> Upload a cleaned CSV/XLSX file directly on the IMDRF Insights page."))
    story.append(bullet("<b>Option B:</b> After running the pipeline on the Dashboard, click \"Open in IMDRF Insights\" to use the pipeline output."))
    story.append(spacer())

    story.append(heading2("7.2 Analysis Levels"))
    story.append(para("IMDRF codes are analyzed at three hierarchical levels:"))
    story.append(make_table(
        ["Level", "Characters", "Example", "Description"],
        [
            ["Level-1", "3 chars", "A01", "Broad category (e.g., \"Adverse Event Type\")"],
            ["Level-2", "5 chars", "A0101", "Sub-category within a Level-1 group"],
            ["Level-3", "7 chars", "A010101", "Most specific classification"],
        ],
        col_widths=[0.8*inch, 1*inch, 0.9*inch, 4.1*inch],
    ))
    story.append(para("You can switch between levels using the <b>Level selector</b> on the page. The data re-loads automatically."))
    story.append(note("Minimum 1 year (365 days) of date coverage is required for meaningful trend analysis. Files with less will be rejected."))

    story.append(spacer())
    story.append(heading2("7.3 Features &amp; Outputs"))

    story.append(heading3("7.3.1 Prefix Overview"))
    story.append(bullet("After upload, the page shows all IMDRF prefixes found in the data with their counts."))
    story.append(bullet("Select a prefix to see its top manufacturers and detailed analysis."))

    story.append(heading3("7.3.2 Manufacturer Comparison Chart"))
    story.append(bullet("Select an IMDRF prefix and one or more manufacturers."))
    story.append(bullet("Choose time granularity: <b>Monthly (M)</b> or <b>Quarterly (Q)</b>."))
    story.append(bullet("Optionally filter by date range."))
    story.append(bullet("Click <b>\"Analyze\"</b> to generate an interactive Plotly line chart showing manufacturer trends vs. the threshold."))

    story.append(heading3("7.3.3 Top Manufacturers Drill-Down"))
    story.append(bullet("For any prefix, view the <b>Top 5 manufacturers</b> by event count."))
    story.append(bullet("View per-manufacturer counts and the average across all active manufacturers."))

    story.append(heading3("7.3.4 Downloadable Reports"))
    story.append(make_table(
        ["Download Button", "Output File", "Contents"],
        [
            ["Download Code Counts XLSX", "imdrf_code_counts_all_levels.xlsx",
             "Multi-sheet workbook: Summary (All Mfrs), Level-1/2/3 per-manufacturer monthly counts, Patient Problem E-Codes, Manufacturer Merges log"],
            ["Download Top-5 Grand Total XLSX", "imdrf_top5_grand_total.xlsx",
             "Top 5 IMDRF code families ranked by grand total (sum across all levels). Optionally split by year."],
            ["Generate PDF Trend Report", "IMDRF_Report_[mfr]_[dates].pdf",
             "Per-manufacturer trend analysis PDF with charts comparing selected manufacturer vs. peers over time."],
            ["Generate Detailed Report", "IMDRF_Detailed_Report_[years].pdf",
             "PSUR-style detailed PDF: Top-5 code families year-over-year, patient problem breakdown, device classification info from openFDA."],
        ],
        col_widths=[1.8*inch, 1.8*inch, 3.2*inch],
    ))

    story.append(heading3("7.3.5 Top-5 Yearly Trend Chart"))
    story.append(bullet("Specify a year range (e.g., 2020\u20132025)."))
    story.append(bullet("An interactive bar chart shows the top-5 IMDRF code families and their year-over-year trend."))

    story.append(heading3("7.3.6 Proportion Analysis"))
    story.append(bullet("For a specific IMDRF code, compute its proportion relative to all codes across a combined historical + current dataset."))

    story.append(heading3("7.3.7 Historical Code Table"))
    story.append(bullet("View the full IMDRF code distribution table from the historical dataset."))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # 8. MODULE 4 — DEVICE RECALL
    # ══════════════════════════════════════════════════════════════════════════
    story.append(heading1("8. Module 4 \u2014 Device Recall Search"))
    story.append(hr())
    story.append(para(
        "The Device Recall page (<b>/device-recall</b>) searches FDA device recall records via the openFDA "
        "<b>/device/recall.json</b> API."
    ))
    story.append(spacer())
    story.append(heading2("8.1 Step-by-Step"))
    story.append(bullet("Enter the <b>Product Code</b>."))
    story.append(bullet("Select <b>Date From</b> and <b>Date To</b>."))
    story.append(bullet("Click <b>\"Search Recalls\"</b>."))
    story.append(spacer())
    story.append(heading2("8.2 Output"))
    story.append(para("Results are displayed in a table with the following columns:"))
    story.append(make_table(
        ["Column", "Description"],
        [
            ["Recall Number", "Unique FDA recall identifier"],
            ["Recalling Firm", "Company that initiated the recall"],
            ["Product Description", "Description of the recalled device"],
            ["Reason for Recall", "Why the recall was initiated"],
            ["Status", "Current recall status (e.g., Ongoing, Terminated, Completed)"],
            ["Classification", "Recall classification (Class I = most serious, Class III = least)"],
            ["Voluntary/Mandated", "Whether the recall was voluntary or FDA-mandated"],
            ["Recall Initiation Date", "Date the recall was initiated"],
            ["Distribution Pattern", "Geographic distribution of affected devices"],
            ["Product Quantity", "Quantity of affected devices"],
        ],
        col_widths=[1.8*inch, 5*inch],
    ))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # 9. MODULE 5 — TXT TO CSV
    # ══════════════════════════════════════════════════════════════════════════
    story.append(heading1("9. Module 5 \u2014 TXT to CSV Converter"))
    story.append(hr())
    story.append(para(
        "The TXT to CSV page (<b>/txt-to-csv</b>) converts raw MAUDE text files (pipe-delimited \"|\") "
        "downloaded from the FDA MAUDE website into clean CSV format."
    ))
    story.append(spacer())
    story.append(heading2("9.1 Step-by-Step"))
    story.append(bullet("Drag-and-drop or click to upload a <b>.txt file</b>."))
    story.append(bullet("Supports <b>chunked upload</b> for very large files (10GB+). Files are uploaded in pieces to avoid browser memory issues."))
    story.append(bullet("After upload, click <b>\"Preview\"</b> to see the first few rows of parsed data."))
    story.append(bullet("Verify the columns look correct."))
    story.append(bullet("Click <b>\"Convert to CSV\"</b>."))
    story.append(bullet("Once conversion completes, click <b>\"Download CSV\"</b> to get the output file."))
    story.append(spacer())
    story.append(heading2("9.2 How It Works"))
    story.append(bullet("The converter reads the pipe-delimited text file line by line."))
    story.append(bullet("It auto-detects the delimiter (pipe \"|\") and header row."))
    story.append(bullet("Fields are cleaned of extraneous whitespace and quotes."))
    story.append(bullet("Output is a standard comma-separated CSV file."))
    story.append(note("This tool is specifically designed for raw MAUDE .txt files downloaded from the FDA MAUDE database at https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfmaude/search.cfm"))

    story.append(spacer(0.5))

    # ══════════════════════════════════════════════════════════════════════════
    # 10. MODULE 6 — CSV VIEWER
    # ══════════════════════════════════════════════════════════════════════════
    story.append(heading1("10. Module 6 \u2014 CSV Viewer"))
    story.append(hr())
    story.append(para(
        "The CSV Viewer (<b>/csv-viewer</b>) lets you explore large CSV or Excel files directly in the browser "
        "without downloading them to Excel."
    ))
    story.append(spacer())
    story.append(heading2("10.1 Step-by-Step"))
    story.append(bullet("Upload a <b>CSV</b> or <b>Excel</b> file (supports chunked upload for large files)."))
    story.append(bullet("The viewer shows file info: total rows, columns, file size."))
    story.append(bullet("Browse data <b>page-by-page</b> using pagination controls."))
    story.append(spacer())
    story.append(heading2("10.2 Features"))
    story.append(bullet("<b>Pagination</b> \u2014 Navigate through pages of data (configurable rows per page)."))
    story.append(bullet("<b>Search</b> \u2014 Search for specific text across all columns or within a specific column."))
    story.append(bullet("<b>Column Statistics</b> \u2014 View statistics for any column (unique values, top values, null counts, etc.)."))
    story.append(bullet("<b>Row Numbers</b> \u2014 Each row displays its original row number."))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # 11. BACKEND PROCESSING PIPELINE (TECHNICAL)
    # ══════════════════════════════════════════════════════════════════════════
    story.append(heading1("11. Backend Processing Pipeline"))
    story.append(hr())
    story.append(para(
        "This section describes the technical data processing steps performed by the backend. "
        "Understanding these helps interpret the output files and audit reports."
    ))

    story.append(heading2("11.1 Data Fetching from openFDA"))
    story.append(heading3("11.1.1 Probe Phase"))
    story.append(para(
        "Before downloading data, the system runs a <b>probe</b> to find the correct query parameters:"
    ))
    story.append(bullet("Tries multiple search fields: <code>device.device_report_product_code</code>, <code>device.product_code</code>, <code>device.openfda.product_code</code>, <code>openfda.product_code</code>."))
    story.append(bullet("Tries multiple date fields: <code>date_received</code>, <code>date_report</code>, <code>date_of_event</code>."))
    story.append(bullet("The first combination that returns results is used for the full fetch."))
    story.append(bullet("If all probes return 404, the system returns a helpful error: \"product code exists but not in date range\" or \"product code not found\"."))
    story.append(note("Probe calls use max_retries=0 (fail fast). Main fetch uses max_retries=4 with exponential backoff."))

    story.append(heading3("11.1.2 Download Phase"))
    story.append(para("The system fetches records page-by-page from openFDA:"))
    story.append(bullet("<b>Page size:</b> 1,000 records per API call (openFDA maximum)."))
    story.append(bullet("<b>Adaptive windowing:</b> For high-volume device codes, the system splits large date ranges into year-level or monthly sub-windows to avoid the openFDA 26,000 record cap per query."))
    story.append(bullet("<b>Threshold:</b> If a year has &gt;20,000 records, it is automatically split into monthly windows."))
    story.append(bullet("<b>Record filtering:</b> Each record is verified to contain the requested product code and fall within the date range (openFDA search is approximate)."))
    story.append(bullet("<b>API key:</b> Uses the configured openFDA API key for higher rate limits (120K requests/day). Falls back to anonymous (1K requests/day) if key is invalid."))

    story.append(heading3("11.1.3 CSV Output Fields"))
    story.append(para("The raw CSV contains these columns extracted from each MAUDE event record:"))
    story.append(make_table(
        ["Column", "Source"],
        [
            ["Report Number", "report_number"],
            ["Event Type", "event_type"],
            ["Date Received", "date_received (reformatted to YYYY-MM-DD)"],
            ["Date of Event", "date_of_event (reformatted)"],
            ["Product Code(s)", "device[].device_report_product_code"],
            ["Brand Name(s)", "device[].brand_name"],
            ["Generic Name(s)", "device[].generic_name"],
            ["Manufacturer Name(s)", "device[].manufacturer_d_name"],
            ["Model Number(s)", "device[].model_number"],
            ["Catalog Number(s)", "device[].catalog_number"],
            ["Device Problem(s)", "device[].openfda.device_name + device_problem codes"],
            ["Patient Problem(s)", "patient[].patient_problems"],
            ["Event Description", "mdr_text (Description of Event narrative)"],
            ["Manufacturer Narrative", "mdr_text (Additional Manufacturer Narrative)"],
        ],
        col_widths=[1.8*inch, 5*inch],
    ))

    story.append(spacer())
    story.append(heading2("11.2 Data Cleaning Pipeline"))
    story.append(para(
        "The <b>MAUDEProcessor</b> class (<code>backend/processor.py</code>) executes these phases in order:"
    ))
    story.append(make_table(
        ["Phase", "Name", "Description"],
        [
            ["0", "Column Identification",
             "Uses deterministic pattern matching (with Groq LLM fallback) to identify which columns represent Event Date, Date Received, Device Problem, Manufacturer, Patient Problem, etc. This handles variant column names across different MAUDE file formats."],
            ["1", "Missing Token Normalization",
             'Converts literal strings like "nan", "null", "none", "n/a", "na" to empty strings in date and key columns.'],
            ["2", "Column Removal",
             "Removes unnecessary columns defined in the configuration (e.g., administrative fields, internal IDs). Tracks removed columns in the audit log."],
            ["3", "Date Standardization",
             'Converts all dates to DD-MM-YYYY format. Handles multiple input formats (YYYYMMDD, YYYY-MM-DD, MM/DD/YYYY, etc.). Invalid dates become empty strings "".'],
            ["3.5", "Blank Date Row Removal",
             "Removes rows where BOTH Event Date AND Date Received are empty (no usable date information). Tracks removal count and reason."],
            ["4", "Manufacturer Normalization",
             "Standardizes manufacturer names: removes legal suffixes (Inc., LLC, Corp., etc.), normalizes whitespace and casing. Uses parent company mapping to merge subsidiaries."],
            ["5", "IMDRF Code Mapping",
             "Maps Device Problem text to IMDRF codes using the Annexes A\u2013G reference file. Uses 3-tier matching: Level-1 (3 chars), Level-2 (5 chars), Level-3 (7 chars). Deterministic keyword matching first; Groq LLM fallback for files &lt;1000 rows."],
            ["6", "Patient Problem Exploding",
             "Semicolon-separated Patient Problem values are exploded into separate rows so each problem gets its own IMDRF code."],
            ["7", "Final Validation",
             "Runs all validation checks (column count, file integrity, date format, IMDRF adjacency, IMDRF code validity). Generates pass/fail report."],
            ["7.5", "Excel Sanitization",
             "Removes illegal characters that would cause Excel to reject the file (control characters, etc.)."],
        ],
        col_widths=[0.5*inch, 1.6*inch, 4.7*inch],
    ))

    story.append(spacer())
    story.append(heading2("11.3 IMDRF Code Mapping Details"))
    story.append(para(
        "IMDRF (International Medical Device Regulators Forum) codes provide a standardized classification "
        "for device problems. The mapper works as follows:"
    ))
    story.append(bullet("<b>Annex Loading:</b> Reads the \"Annexes A-G consolidated.xlsx\" file to build lookup maps for Level-1, Level-2, and Level-3 codes."))
    story.append(bullet("<b>Deterministic Matching:</b> Normalizes device problem text and matches against known IMDRF terms using keyword overlap and similarity scoring."))
    story.append(bullet("<b>Groq LLM Fallback:</b> For files with &lt;1,000 rows, unmatched problems are sent to the Groq API in batches for AI-assisted classification. This is skipped for large files to maintain performance."))
    story.append(bullet("<b>Caching:</b> All mappings (deterministic and AI) are cached locally to avoid repeated API calls."))
    story.append(bullet("<b>Excluded Prefixes:</b> Codes with Level-1 prefixes A24 and A25 are excluded from all code-count outputs."))
    story.append(note("The IMDRF Code column is always placed immediately adjacent to the Device Problem column in the output."))

    story.append(spacer())
    story.append(heading2("11.4 Parent Company Normalization"))
    story.append(para(
        "The system maintains a parent company map (<code>backend/parent_company_map.py</code>) that merges "
        "subsidiary/variant manufacturer names into their parent company. For example, \"MEDTRONIC MINIMED\" "
        "and \"MEDTRONIC INC\" would both be mapped to \"MEDTRONIC\". This merge is tracked and included in "
        "the XLSX download as a \"Manufacturer Merges\" sheet."
    ))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # 12. DATA FLOW DIAGRAM
    # ══════════════════════════════════════════════════════════════════════════
    story.append(heading1("12. Data Flow Diagram"))
    story.append(hr())
    story.append(para("The following describes the end-to-end data flow through the platform:"))
    story.append(spacer())

    flow_data = [
        ["Step", "Process", "Input", "Output"],
        ["1", "User enters Product Code + Date Range", "User input via web form", "Product Code, Date From, Date To"],
        ["2", "Probe openFDA API", "Product Code + Dates", "Correct search field, total record count"],
        ["3", "Download MAUDE records (paginated)", "openFDA API queries", "Raw CSV file (all fields)"],
        ["4", "Column Identification", "Raw CSV columns", "Column mapping (which col = what field)"],
        ["5", "Data Cleaning", "Raw CSV + column map", "Cleaned data (removed cols, standardized dates)"],
        ["6", "Manufacturer Normalization", "Cleaned data", "Normalized manufacturer names"],
        ["7", "IMDRF Code Mapping", "Device Problem text + Annex file", "IMDRF codes (Level 1/2/3)"],
        ["8", "Patient Problem Exploding", "Semicolon-separated values", "One row per problem"],
        ["9", "Validation", "Final dataframe", "Validation report (pass/fail)"],
        ["10", "Export", "Validated data", "Cleaned XLSX / IMDRF Code Counts XLSX / Audit Report"],
    ]
    flow_t = Table(flow_data, colWidths=[0.5*inch, 2*inch, 2.2*inch, 2.1*inch])
    flow_t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1E40AF')),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#FFFFFF')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#FFFFFF'), HexColor('#F1F5F9')]),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(flow_t)
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # 13. GLOSSARY
    # ══════════════════════════════════════════════════════════════════════════
    story.append(heading1("13. Glossary"))
    story.append(hr())
    story.append(make_table(
        ["Term", "Definition"],
        [
            ["MAUDE", "Manufacturer and User Facility Device Experience \u2014 FDA's database of medical device adverse event reports."],
            ["openFDA", "FDA's public API for accessing structured FDA datasets (drugs, devices, food, etc.)."],
            ["IMDRF", "International Medical Device Regulators Forum \u2014 organization that publishes standardized codes for device problems and patient outcomes."],
            ["Product Code", "A 3-letter FDA code identifying a type of medical device (e.g., LYZ = Glucose Test System)."],
            ["Annex A\u2013G", "IMDRF reference document containing the full hierarchy of device problem codes (Annex A), patient problem codes (Annex E), and others."],
            ["Level-1 / L1", "First 3 alphanumeric characters of an IMDRF code (broadest classification)."],
            ["Level-2 / L2", "First 5 alphanumeric characters of an IMDRF code (mid-level classification)."],
            ["Level-3 / L3", "First 7 alphanumeric characters of an IMDRF code (most specific classification)."],
            ["MDR", "Medical Device Report \u2014 an individual adverse event report in the MAUDE system."],
            ["PSUR", "Periodic Safety Update Report \u2014 a regulatory document summarizing safety data over a period."],
            ["Groq", "LLM API service used as a fallback for AI-assisted column identification and IMDRF code mapping."],
            ["Pipeline", "The automated sequence of steps: Fetch \u2192 Clean \u2192 Map \u2192 Validate \u2192 Export."],
            ["Probe", "Initial API call to verify the product code exists and determine the correct search parameters."],
            ["Adaptive Windowing", "Strategy that splits large date ranges into smaller windows to avoid openFDA's 26K record limit per query."],
            ["Parent Company Map", "Lookup table that merges subsidiary manufacturer names into their parent company for consistent analysis."],
        ],
        col_widths=[1.8*inch, 5*inch],
    ))

    story.append(Spacer(1, 0.5*inch))
    story.append(HRFlowable(width="100%", thickness=2, color=HexColor('#2563EB'),
                             spaceBefore=12, spaceAfter=12))
    story.append(Paragraph("END OF DOCUMENT", ParagraphStyle(
        'EndDoc', parent=styles['Normal'],
        fontSize=11, alignment=TA_CENTER, textColor=HexColor('#6B7280'),
    )))

    # ── Build ─────────────────────────────────────────────────────────────────
    doc.build(story)
    print(f"\nPDF generated: {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == '__main__':
    build()
