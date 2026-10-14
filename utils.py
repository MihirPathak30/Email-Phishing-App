from __future__ import annotations

import base64
import io
import json
import re
from collections import Counter
from datetime import datetime
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import plotly.graph_objects as go
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
ASSETS_DIR = BASE_DIR / "assets"
HISTORY_FILE = OUTPUT_DIR / "analysis_history.json"

URL_REGEX = re.compile(r"https?://[^\s<>'\"]+|www\.[^\s<>'\"]+", re.IGNORECASE)
SHORT_URL_REGEX = re.compile(r"https?://(?:bit\.ly|tinyurl\.com|t\.co|goo\.gl|ow\.ly|buff\.ly|is\.gd|cutt\.ly|rebrand\.ly|shorturl\.at)/[^\s]+", re.IGNORECASE)
IP_REGEX = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_REGEX = re.compile(r"(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)?\d{3}[\s.-]?\d{4}\b")
HTML_REGEX = re.compile(r"<[^>]+>")
ATTACHMENT_REGEX = re.compile(r"\b(?:attached|attachment|invoice|statement|document|receipt|pdf|zip|docx?|xlsx?|pptx?)\b", re.IGNORECASE)
SPECIAL_CHAR_REGEX = re.compile(r"[^A-Za-z0-9\s]")
SUSPICIOUS_DOMAIN_REGEX = re.compile(r"(?:secure|verify|account|billing|update|login|wallet|alert|bonus|gift|crypto|pay|bank)[-.]", re.IGNORECASE)
TOKEN_REGEX = re.compile(r"[a-zA-Z]{2,}")

URGENT_WORDS = {
    "urgent", "immediately", "asap", "now", "today", "limited", "deadline", "suspend",
    "warning", "final", "act", "important", "attention", "alert", "instantly", "critical"
}
FINANCIAL_WORDS = {
    "bank", "payment", "invoice", "refund", "salary", "wire", "transfer", "bitcoin", "crypto",
    "wallet", "account", "billing", "tax", "fee", "purchase", "transaction", "debit", "credit"
}
PASSWORD_WORDS = {
    "password", "passcode", "otp", "pin", "credential", "login", "username", "verify", "authentication",
    "reset", "confirm", "security code", "2fa"
}
SUSPICIOUS_DOMAINS = {
    "bit.ly", "tinyurl.com", "goo.gl", "ow.ly", "dropbox-login.com", "paypal-verification.com",
    "secure-update.net", "account-verify.co", "wallet-alert.io", "login-check.org"
}

LEGIT_SAMPLE = """Subject: Team Lunch Confirmation\nFrom: hr@company.com\nTo: employee@company.com\n\nHello team,\n\nThis is a reminder that our department lunch is scheduled for Friday at 1:00 PM in Conference Room B. Please confirm your attendance by replying to this email.\n\nBest regards,\nHR Team"""

PHISHING_SAMPLE = """Subject: Urgent Account Suspension Notice\nFrom: security-update@account-verify.co\nTo: customer@example.com\n\nDear User,\n\nYour account will be suspended immediately unless you verify your password and banking details today. Click https://bit.ly/secure-login-reset to confirm your credentials and avoid permanent deactivation. You may also reply with your OTP for faster processing.\n\nSecurity Billing Team"""


def ensure_output_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def parse_uploaded_email(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    suffix = Path(uploaded_file.name).suffix.lower()
    raw = uploaded_file.read()
    uploaded_file.seek(0)
    if suffix == ".txt":
        return raw.decode("utf-8", errors="ignore")
    if suffix == ".eml":
        return parse_eml_bytes(raw)
    return raw.decode("utf-8", errors="ignore")


def parse_eml_bytes(raw: bytes) -> str:
    msg = BytesParser(policy=policy.default).parsebytes(raw)
    chunks = []
    subject = msg.get("subject", "")
    sender = msg.get("from", "")
    recipient = msg.get("to", "")
    if subject:
        chunks.append(f"Subject: {subject}")
    if sender:
        chunks.append(f"From: {sender}")
    if recipient:
        chunks.append(f"To: {recipient}")
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            disposition = str(part.get_content_disposition() or "")
            if "attachment" in disposition:
                filename = part.get_filename()
                if filename:
                    chunks.append(f"Attachment: {filename}")
            elif content_type in {"text/plain", "text/html"}:
                try:
                    content = part.get_content()
                except Exception:
                    content = ""
                if content_type == "text/html":
                    content = BeautifulSoup(str(content), "html.parser").get_text(" ")
                if content:
                    chunks.append(str(content))
    else:
        content = msg.get_content()
        if msg.get_content_type() == "text/html":
            content = BeautifulSoup(str(content), "html.parser").get_text(" ")
        chunks.append(str(content))
    return "\n\n".join(chunk for chunk in chunks if chunk)


def extract_matches(pattern: re.Pattern, text: str) -> List[str]:
    found = pattern.findall(text)
    cleaned = []
    for item in found:
        if isinstance(item, tuple):
            item = " ".join([x for x in item if x])
        cleaned.append(str(item).strip())
    return sorted(set([c for c in cleaned if c]))


def extract_keyword_hits(text: str, keywords: set) -> List[str]:
    lowered = text.lower()
    hits = [kw for kw in keywords if kw in lowered]
    return sorted(set(hits))


def extract_suspicious_domains(text: str) -> List[str]:
    urls = extract_matches(URL_REGEX, text)
    hits = set()
    for url in urls:
        for domain in SUSPICIOUS_DOMAINS:
            if domain in url.lower():
                hits.add(domain)
        if SUSPICIOUS_DOMAIN_REGEX.search(url.lower()):
            hits.add(url)
    emails = extract_matches(EMAIL_REGEX, text)
    for email in emails:
        domain = email.split("@")[-1].lower()
        if any(token in domain for token in ["verify", "secure", "billing", "wallet", "bonus", "crypto", "login"]):
            hits.add(domain)
    return sorted(hits)


def clean_visible_text(text: str) -> str:
    if not text:
        return ""
    soup_text = BeautifulSoup(text, "html.parser").get_text(" ")
    soup_text = re.sub(r"\s+", " ", soup_text)
    return soup_text.strip()


def analyze_email_indicators(text: str) -> Dict[str, List[str]]:
    indicators = {
        "URLs": extract_matches(URL_REGEX, text),
        "Shortened URLs": extract_matches(SHORT_URL_REGEX, text),
        "IP Addresses": extract_matches(IP_REGEX, text),
        "Email Addresses": extract_matches(EMAIL_REGEX, text),
        "Phone Numbers": extract_matches(PHONE_REGEX, text),
        "HTML Tags": extract_matches(HTML_REGEX, text),
        "Attachments Mentioned": extract_matches(ATTACHMENT_REGEX, text),
        "Urgent Words": extract_keyword_hits(text, URGENT_WORDS),
        "Financial Keywords": extract_keyword_hits(text, FINANCIAL_WORDS),
        "Password Keywords": extract_keyword_hits(text, PASSWORD_WORDS),
        "Suspicious Domains": extract_suspicious_domains(text),
    }
    return indicators


def compute_readability_score(text: str) -> float:
    words = re.findall(r"\b\w+\b", text)
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    syllables = sum(estimate_syllables(word) for word in words)
    if not words or not sentences:
        return 0.0
    asl = len(words) / max(len(sentences), 1)
    asw = syllables / max(len(words), 1)
    score = 206.835 - 1.015 * asl - 84.6 * asw
    return round(max(min(score, 100), 0), 2)


def estimate_syllables(word: str) -> int:
    word = word.lower()
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def compute_email_stats(text: str, indicators: Dict[str, List[str]]) -> Dict[str, object]:
    visible_text = clean_visible_text(text)
    words = re.findall(r"\b\w+\b", visible_text)
    sentences = [s for s in re.split(r"[.!?]+", visible_text) if s.strip()]
    stats = {
        "Word Count": len(words),
        "Character Count": len(text),
        "Sentence Count": len(sentences),
        "Number of URLs": len(indicators["URLs"]),
        "Number of Emails": len(indicators["Email Addresses"]),
        "Capital Letters": sum(1 for ch in text if ch.isupper()),
        "Special Characters": len(SPECIAL_CHAR_REGEX.findall(text)),
        "Numbers": sum(1 for ch in text if ch.isdigit()),
        "HTML Present": bool(indicators["HTML Tags"]),
        "Readability Score": compute_readability_score(visible_text),
        "Suspicious Keywords Present": bool(indicators["Urgent Words"] or indicators["Financial Keywords"] or indicators["Password Keywords"]),
    }
    return stats


def indicators_to_table(indicators: Dict[str, List[str]]) -> pd.DataFrame:
    rows = []
    for category, values in indicators.items():
        rows.append(
            {
                "Indicator": category,
                "Count": len(values),
                "Detected Items": ", ".join(values[:10]) if values else "—",
            }
        )
    return pd.DataFrame(rows)


def risk_level_from_score(phishing_probability: float, indicators: Dict[str, List[str]]) -> Tuple[str, str, int]:
    score = phishing_probability * 100
    risk_boost = 0
    risk_boost += min(len(indicators["URLs"]) * 3, 12)
    risk_boost += min(len(indicators["Shortened URLs"]) * 5, 10)
    risk_boost += min(len(indicators["Urgent Words"]) * 2, 10)
    risk_boost += min(len(indicators["Password Keywords"]) * 4, 12)
    risk_boost += min(len(indicators["Suspicious Domains"]) * 5, 15)
    adjusted = min(score + risk_boost, 100)
    if adjusted < 30:
        return "Low", "#23c55e", round(adjusted)
    if adjusted < 55:
        return "Medium", "#eab308", round(adjusted)
    if adjusted < 75:
        return "High", "#f97316", round(adjusted)
    return "Critical", "#ef4444", round(adjusted)


def build_reason_and_summary(prediction: str, confidence: float, indicators: Dict[str, List[str]], stats: Dict[str, object]) -> Tuple[str, str]:
    reasons = []
    if indicators["Shortened URLs"]:
        reasons.append("shortened links that can hide the true destination")
    if indicators["Suspicious Domains"]:
        reasons.append("suspicious or spoof-like sender/link domains")
    if indicators["Urgent Words"]:
        reasons.append("urgency-driven language")
    if indicators["Password Keywords"]:
        reasons.append("credential or password-related requests")
    if indicators["Financial Keywords"]:
        reasons.append("financial or payment-themed wording")
    if indicators["HTML Tags"]:
        reasons.append("HTML content often used in crafted phishing layouts")
    if not reasons and prediction == "Legitimate":
        reasons.append("a low density of risky patterns and a more routine communication style")
    if not reasons:
        reasons.append("multiple structural and linguistic anomalies")

    if prediction == "Phishing":
        summary = (
            f"This email is classified as phishing because it contains {', '.join(reasons[:4])}. "
            f"The model confidence is {confidence:.2f}% and the message shows {stats['Number of URLs']} URL(s)."
        )
    else:
        summary = (
            f"This email is classified as legitimate because it shows {', '.join(reasons[:3])}. "
            f"The model confidence is {confidence:.2f}% with limited suspicious indicators."
        )
    return "; ".join(reasons), summary


def build_probability_chart(phishing_probability: float) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Bar(
                x=["Legitimate", "Phishing"],
                y=[round((1 - phishing_probability) * 100, 2), round(phishing_probability * 100, 2)],
                marker_color=["#22c55e", "#ef4444"],
                text=[f"{(1 - phishing_probability) * 100:.2f}%", f"{phishing_probability * 100:.2f}%"],
                textposition="auto",
            )
        ]
    )
    fig.update_layout(template="plotly_dark", height=320, margin=dict(l=20, r=20, t=20, b=20), yaxis_title="Probability %")
    return fig


def build_risk_gauge(score: int, level: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "%"},
            title={"text": f"Risk Meter · {level}"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#60a5fa"},
                "steps": [
                    {"range": [0, 30], "color": "#14532d"},
                    {"range": [30, 55], "color": "#713f12"},
                    {"range": [55, 75], "color": "#7c2d12"},
                    {"range": [75, 100], "color": "#7f1d1d"},
                ],
            },
        )
    )
    fig.update_layout(template="plotly_dark", height=320, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def build_keyword_frequency_chart(indicators: Dict[str, List[str]]) -> go.Figure:
    categories = ["Urgent", "Financial", "Password", "Suspicious Domains", "URLs", "Emails"]
    values = [
        len(indicators["Urgent Words"]),
        len(indicators["Financial Keywords"]),
        len(indicators["Password Keywords"]),
        len(indicators["Suspicious Domains"]),
        len(indicators["URLs"]),
        len(indicators["Email Addresses"]),
    ]
    fig = go.Figure(
        data=[go.Bar(x=categories, y=values, marker_color=["#f43f5e", "#f59e0b", "#8b5cf6", "#ef4444", "#38bdf8", "#34d399"])]
    )
    fig.update_layout(template="plotly_dark", height=320, margin=dict(l=20, r=20, t=20, b=20), yaxis_title="Count")
    return fig


def build_wordcloud(text: str):
    from wordcloud import WordCloud

    wc = WordCloud(width=1200, height=500, background_color="#0b1220", colormap="cool", collocations=False)
    return wc.generate(clean_visible_text(text) or "empty")


def load_history() -> List[Dict[str, object]]:
    ensure_output_dirs()
    if not HISTORY_FILE.exists():
        return []
    try:
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_history_entry(entry: Dict[str, object]) -> None:
    history = load_history()
    history.insert(0, entry)
    HISTORY_FILE.write_text(json.dumps(history[:50], indent=2), encoding="utf-8")


def history_to_dataframe() -> pd.DataFrame:
    history = load_history()
    if not history:
        return pd.DataFrame(columns=["Timestamp", "Prediction", "Confidence", "Risk Level", "Preview"])
    return pd.DataFrame(history)


def create_txt_report(result: Dict[str, object]) -> bytes:
    lines = [
        "AI-Driven Phishing Email Detection Report",
        "=" * 48,
        f"Date: {result['timestamp']}",
        f"Prediction: {result['prediction']}",
        f"Confidence: {result['confidence']:.2f}%",
        f"Risk Level: {result['risk_level']}",
        f"Reason: {result['reason']}",
        f"Summary: {result['summary']}",
        "",
        "Detected Indicators",
        "-" * 20,
    ]
    for category, values in result["indicators"].items():
        lines.append(f"{category}: {', '.join(values) if values else 'None'}")
    lines.extend(["", "Email Statistics", "-" * 20])
    for key, value in result["stats"].items():
        lines.append(f"{key}: {value}")
    return "\n".join(lines).encode("utf-8")


def create_pdf_report(result: Dict[str, object]) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=12 * mm, bottomMargin=12 * mm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleDark", parent=styles["Title"], textColor=colors.HexColor("#0f172a"))
    body_style = ParagraphStyle("BodyDark", parent=styles["BodyText"], leading=14)
    story = [Paragraph("AI-Driven Phishing Email Detection Report", title_style), Spacer(1, 8)]
    meta_rows = [
        ["Date", result["timestamp"]],
        ["Prediction", result["prediction"]],
        ["Confidence", f"{result['confidence']:.2f}%"],
        ["Risk Level", result["risk_level"]],
    ]
    table = Table(meta_rows, colWidths=[50 * mm, 120 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbeafe")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.whitesmoke, colors.HexColor("#f8fafc")]),
    ]))
    story.extend([table, Spacer(1, 10)])
    story.append(Paragraph(f"<b>Reason:</b> {result['reason']}", body_style))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"<b>Summary:</b> {result['summary']}", body_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Detected Indicators", styles["Heading2"]))
    indicator_rows = [["Indicator", "Count", "Detected Items"]]
    for category, values in result["indicators"].items():
        indicator_rows.append([category, str(len(values)), ", ".join(values[:6]) if values else "None"])
    ind_table = Table(indicator_rows, colWidths=[45 * mm, 20 * mm, 105 * mm])
    ind_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#bfdbfe")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#94a3b8")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]))
    story.extend([ind_table, Spacer(1, 10), Paragraph("Email Statistics", styles["Heading2"])])
    stats_rows = [["Metric", "Value"]] + [[k, str(v)] for k, v in result["stats"].items()]
    stats_table = Table(stats_rows, colWidths=[70 * mm, 100 * mm])
    stats_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#bfdbfe")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#94a3b8")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
    ]))
    story.append(stats_table)
    doc.build(story)
    buffer.seek(0)
    return buffer.read()


def image_file_to_base64(path: Path) -> str:
    if not path.exists():
        return ""
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def get_theme_css(mode: str = "dark") -> str:
    dark = mode == "dark"
    bg = "#020617" if dark else "#f8fafc"
    panel = "rgba(15, 23, 42, 0.72)" if dark else "rgba(255,255,255,0.82)"
    text = "#e2e8f0" if dark else "#0f172a"
    sub = "#94a3b8" if dark else "#475569"
    border = "rgba(148,163,184,0.22)"
    return f"""
    <style>
    .stApp {{
        background:
            radial-gradient(circle at top left, rgba(59,130,246,0.22), transparent 22%),
            radial-gradient(circle at top right, rgba(168,85,247,0.20), transparent 20%),
            linear-gradient(135deg, {bg}, {'#0f172a' if dark else '#e2e8f0'});
        color: {text};
    }}
    section[data-testid='stSidebar'] > div {{
        background: linear-gradient(180deg, rgba(15,23,42,0.95), rgba(2,6,23,0.9));
    }}
    .glass {{
        background: {panel};
        border: 1px solid {border};
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 24px;
        padding: 1.2rem 1.3rem;
        box-shadow: 0 18px 40px rgba(15,23,42,0.24);
    }}
    .hero {{
        padding: 2rem 2.2rem;
        border-radius: 28px;
        background: linear-gradient(135deg, rgba(37,99,235,0.35), rgba(168,85,247,0.24));
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 25px 45px rgba(15,23,42,0.28);
    }}
    .hero h1 {{ font-size: 2.4rem; margin-bottom: 0.4rem; color: white; }}
    .hero p {{ color: #dbeafe; font-size: 1rem; margin-bottom: 0; }}
    .mini-badge {{
        display:inline-block; padding:0.35rem 0.65rem; margin-right:0.4rem; margin-top:0.4rem;
        border-radius:999px; background:rgba(255,255,255,0.08); color:#e0f2fe; font-size:0.82rem;
        border:1px solid rgba(255,255,255,0.09);
    }}
    .metric-card {{
        background: {panel}; color:{text}; border:1px solid {border}; border-radius:22px; padding:1rem;
        min-height:120px; box-shadow: 0 16px 36px rgba(15,23,42,0.18); transition: transform 0.2s ease;
    }}
    .metric-card:hover {{ transform: translateY(-3px); }}
    .metric-title {{ color: {sub}; font-size: 0.88rem; }}
    .metric-value {{ font-size: 1.8rem; font-weight: 700; margin-top: 0.25rem; }}
    .status-pill {{ display:inline-block; padding:0.45rem 0.9rem; border-radius:999px; font-weight:700; }}
    .section-title {{ color:{text}; font-size:1.2rem; font-weight:700; margin:1rem 0 0.8rem; }}
    .result-box {{
        background: linear-gradient(135deg, rgba(14,165,233,0.18), rgba(99,102,241,0.15));
        border: 1px solid rgba(125,211,252,0.18); border-radius:22px; padding:1rem 1.2rem;
    }}
    .small-muted {{ color:{sub}; font-size:0.88rem; }}
    .stButton > button, .stDownloadButton > button {{
        border-radius: 14px; border: 1px solid rgba(148,163,184,0.25); padding: 0.65rem 1rem;
        background: linear-gradient(135deg, rgba(37,99,235,0.88), rgba(147,51,234,0.88));
        color: white; font-weight: 600;
    }}
    .stTextArea textarea {{ border-radius: 18px !important; }}
    div[data-testid='stMetric'] {{ background: transparent !important; }}
    </style>
    """


def copy_button_component(text: str) -> str:
    payload = json.dumps(text)
    return f"""
    <div style='margin-top:8px;'>
      <button onclick='navigator.clipboard.writeText({payload})' style='padding:10px 14px;border:none;border-radius:12px;background:linear-gradient(135deg,#2563eb,#9333ea);color:white;font-weight:600;cursor:pointer;'>Copy Result</button>
    </div>
    """


def keyword_frequency_from_text(text: str, limit: int = 15) -> pd.DataFrame:
    tokens = [t.lower() for t in TOKEN_REGEX.findall(clean_visible_text(text)) if len(t) > 2]
    stop_words = {
        "the", "and", "for", "that", "this", "with", "your", "from", "have", "you", "are", "was", "but", "not",
        "our", "please", "will", "has", "all", "can", "out", "use", "www", "https", "http", "com", "org", "net"
    }
    filtered = [t for t in tokens if t not in stop_words]
    freq = Counter(filtered).most_common(limit)
    return pd.DataFrame(freq, columns=["Keyword", "Frequency"])


def timestamp_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
