from __future__ import annotations

import io
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from predict import analyze_email, ensure_artifacts
from utils import (
    ASSETS_DIR,
    LEGIT_SAMPLE,
    PHISHING_SAMPLE,
    build_keyword_frequency_chart,
    build_probability_chart,
    build_risk_gauge,
    build_wordcloud,
    copy_button_component,
    create_pdf_report,
    create_txt_report,
    get_theme_css,
    history_to_dataframe,
    image_file_to_base64,
    indicators_to_table,
    keyword_frequency_from_text,
    parse_uploaded_email,
    save_history_entry,
)

APP_TITLE = "AI-Driven Phishing Email Detection"
BASE_DIR = Path(__file__).resolve().parent
LOGO_PATH = ASSETS_DIR / "shield_logo.svg"

st.set_page_config(page_title=APP_TITLE, page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")


def init_state():
    defaults = {
        "theme_mode": "dark",
        "email_text": "",
        "analysis_result": None,
        "demo_trigger": False,
        "active_page": "Dashboard",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def set_sample(sample_text: str):
    st.session_state.email_text = sample_text


def render_metric_card(title: str, value, subtitle: str = ""):
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-title'>{title}</div>
            <div class='metric-value'>{value}</div>
            <div class='small-muted'>{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_logo():
    logo_b64 = image_file_to_base64(LOGO_PATH)
    if logo_b64:
        st.sidebar.markdown(
            f"<div style='text-align:center;margin-bottom:12px;'><img src='data:image/svg+xml;base64,{logo_b64}' width='84'></div>",
            unsafe_allow_html=True,
        )


def build_confusion_matrix_figure(metadata: dict) -> go.Figure:
    matrix = metadata.get("confusion_matrix", [[0, 0], [0, 0]])
    labels = metadata.get("labels", ["Legitimate", "Phishing"])
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            colorscale="Blues",
            text=matrix,
            texttemplate="%{text}",
        )
    )
    fig.update_layout(template="plotly_dark", title="Confusion Matrix", height=360, xaxis_title="Predicted", yaxis_title="Actual")
    return fig


def build_feature_importance_figure(metadata: dict) -> go.Figure:
    features = metadata.get("feature_importance", [])
    if not features:
        return go.Figure()
    df = pd.DataFrame(features).sort_values("importance", ascending=True)
    fig = go.Figure(
        data=[go.Bar(x=df["importance"], y=df["feature"], orientation="h", marker_color="#8b5cf6")]
    )
    fig.update_layout(template="plotly_dark", title="Feature Importance", height=480, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def build_distribution_figure(metadata: dict) -> go.Figure:
    dist = metadata.get("class_distribution", {"legitimate": 0, "phishing": 0})
    fig = px.pie(
        values=[dist.get("legitimate", 0), dist.get("phishing", 0)],
        names=["Legitimate", "Phishing"],
        hole=0.55,
        color_discrete_sequence=["#22c55e", "#ef4444"],
        title="Dataset Distribution",
    )
    fig.update_layout(template="plotly_dark", height=340, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def render_hero(metadata: dict):
    best_model = metadata.get("best_model_name", "Auto-selected ML model")
    dataset_size = metadata.get("dataset_size", "Auto")
    st.markdown(
        f"""
        <div class='hero'>
            <div class='mini-badge'>🛡️ Cybersecurity NLP Dashboard</div>
            <div class='mini-badge'>⚙️ Best Model: {best_model}</div>
            <div class='mini-badge'>📊 Dataset Size: {dataset_size}</div>
            <h1>{APP_TITLE}</h1>
            <p>Commercial-style phishing intelligence cockpit built with Streamlit, NLP, and Scikit-learn. Paste raw email content or upload .txt/.eml files for instant risk scoring, explanation, and evidence-backed pattern analysis.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def run_analysis_from_state():
    text = st.session_state.email_text.strip()
    if not text:
        st.warning("Please paste an email or upload a .txt/.eml file before analysis.")
        return
    with st.spinner("Analyzing email content, extracting indicators, and scoring phishing risk..."):
        result = analyze_email(text)
        st.session_state.analysis_result = result
        save_history_entry(
            {
                "Timestamp": result["timestamp"],
                "Prediction": result["prediction"],
                "Confidence": f"{result['confidence']:.2f}%",
                "Risk Level": result["risk_level"],
                "Preview": result["preview"],
            }
        )


def render_overview(result: dict):
    pred_color = "#ef4444" if result["prediction"] == "Phishing" else "#22c55e"
    st.markdown(
        f"""
        <div class='result-box'>
            <div class='small-muted'>Prediction</div>
            <div style='display:flex;align-items:center;gap:10px;flex-wrap:wrap;'>
                <h2 style='margin:0;color:white;'>{'🚨' if result['prediction']=='Phishing' else '✅'} {result['prediction']} Email</h2>
                <span class='status-pill' style='background:{pred_color}; color:white;'>{result['risk_level']} Risk</span>
            </div>
            <p style='margin-top:0.7rem;'>{result['summary']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(min(max(result["confidence"] / 100, 0.0), 1.0), text=f"Confidence: {result['confidence']:.2f}%")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Prediction", result["prediction"], "Model decision")
    with c2:
        render_metric_card("Confidence", f"{result['confidence']:.2f}%", "Prediction certainty")
    with c3:
        render_metric_card("Risk Level", result["risk_level"], "Behavioral + model score")
    with c4:
        render_metric_card("Best Model", result["metadata"].get("best_model_name", "N/A"), "Auto-selected during training")

    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.plotly_chart(build_risk_gauge(result["risk_score"], result["risk_level"]), use_container_width=True)
    with col_right:
        st.plotly_chart(build_probability_chart(result["phishing_probability"]), use_container_width=True)

    st.markdown("<div class='section-title'>AI Explanation</div>", unsafe_allow_html=True)
    st.info(result["summary"])
    st.caption(f"Reason for prediction: {result['reason']}")

    report_col1, report_col2, report_col3 = st.columns([1, 1, 1])
    with report_col1:
        st.download_button(
            "Download PDF Report",
            data=create_pdf_report(result),
            file_name="phishing_analysis_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    with report_col2:
        st.download_button(
            "Download TXT Report",
            data=create_txt_report(result),
            file_name="phishing_analysis_report.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with report_col3:
        components.html(copy_button_component(create_txt_report(result).decode("utf-8")), height=56)


def render_analysis_tables(result: dict):
    st.markdown("<div class='section-title'>Detected Indicators</div>", unsafe_allow_html=True)
    st.dataframe(indicators_to_table(result["indicators"]), use_container_width=True, hide_index=True)

    st.markdown("<div class='section-title'>Email Statistics</div>", unsafe_allow_html=True)
    stats_items = list(result["stats"].items())
    stat_cols = st.columns(5)
    for i, (title, value) in enumerate(stats_items):
        with stat_cols[i % 5]:
            render_metric_card(title, value, "")

    with st.expander("Expanded indicator details"):
        for category, values in result["indicators"].items():
            st.markdown(f"**{category}**")
            if values:
                st.write(values)
            else:
                st.write("No items detected.")


def render_visuals(result: dict):
    metadata = result["metadata"]
    top_row_1, top_row_2 = st.columns(2)
    with top_row_1:
        st.plotly_chart(build_confusion_matrix_figure(metadata), use_container_width=True)
    with top_row_2:
        st.plotly_chart(build_feature_importance_figure(metadata), use_container_width=True)

    mid_row_1, mid_row_2 = st.columns(2)
    with mid_row_1:
        st.plotly_chart(build_keyword_frequency_chart(result["indicators"]), use_container_width=True)
    with mid_row_2:
        st.plotly_chart(build_distribution_figure(metadata), use_container_width=True)

    keyword_df = keyword_frequency_from_text(st.session_state.email_text)
    if not keyword_df.empty:
        st.plotly_chart(
            px.bar(keyword_df, x="Keyword", y="Frequency", title="Keyword Frequency in Current Email", color="Frequency", color_continuous_scale="Turbo").update_layout(template="plotly_dark", height=340),
            use_container_width=True,
        )

    st.markdown("<div class='section-title'>Word Cloud</div>", unsafe_allow_html=True)
    wc = build_wordcloud(st.session_state.email_text)
    fig, ax = plt.subplots(figsize=(14, 5), facecolor="#0b1220")
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_history():
    st.markdown("<div class='section-title'>Recent Predictions</div>", unsafe_allow_html=True)
    df = history_to_dataframe()
    if df.empty:
        st.info("No saved analysis history yet. Run an email analysis to populate this table.")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)


def main():
    init_state()
    st.markdown(get_theme_css(st.session_state.theme_mode), unsafe_allow_html=True)
    metadata = ensure_artifacts()

    render_logo()
    st.sidebar.title("Control Center")
    theme_toggle = st.sidebar.toggle("Light mode", value=st.session_state.theme_mode == "light")
    st.session_state.theme_mode = "light" if theme_toggle else "dark"
    st.sidebar.caption("Upload email files, launch demos, and switch interface theme.")

    uploaded_file = st.sidebar.file_uploader(
        "Upload Email (.txt/.eml) or Dataset (.csv)",
        type=["txt", "eml", "csv"]
    )

    if uploaded_file and st.sidebar.button("Load Uploaded File", use_container_width=True):

        if uploaded_file.name.lower().endswith(".csv"):

            df = pd.read_csv(uploaded_file)

            st.success(f"Dataset Loaded Successfully ({len(df)} rows)")
            st.dataframe(df.head())

            possible_columns = ["text","body","email","message","content","mail"]

            email_column = next((c for c in possible_columns if c in df.columns), None)

            if email_column is None:
                st.error(f"No email text column found. Available columns: {list(df.columns)}")
            else:
                st.session_state.csv_data = df
                st.session_state.email_column = email_column

        else:
            st.session_state.email_text = parse_uploaded_email(uploaded_file)

    st.sidebar.markdown("### Demo Inputs")
    demo_col1, demo_col2 = st.sidebar.columns(2)
    with demo_col1:
        if st.button("Sample Legitimate", use_container_width=True):
            set_sample(LEGIT_SAMPLE)
    with demo_col2:
        if st.button("Sample Phishing", use_container_width=True):
            set_sample(PHISHING_SAMPLE)
    if st.sidebar.button("One-click Demo", use_container_width=True):
        set_sample(PHISHING_SAMPLE)
        st.session_state.demo_trigger = True
    if st.sidebar.button("Clear Text", use_container_width=True):
        st.session_state.email_text = ""
        st.session_state.analysis_result = None

    st.sidebar.markdown("### Navigation")
    st.session_state.active_page = st.sidebar.radio("Section", ["Dashboard", "Indicators", "Visuals", "History"], label_visibility="collapsed")

    render_hero(metadata)
    st.write("")

    left, right = st.columns([1.35, 0.65])
    with left:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.subheader("Email Input")
        st.caption("Paste the complete email body, headers, or upload a .txt/.eml file.")
        text_area = st.text_area(
            "Email content",
            value=st.session_state.email_text,
            height=260,
            placeholder="Paste suspicious email content here...",
            label_visibility="collapsed",
        )
        st.session_state.email_text = text_area
        analyze_col, spacer_col = st.columns([1, 4])
        with analyze_col:
            if st.button("Check Email", use_container_width=True) or st.session_state.demo_trigger:

                st.session_state.demo_trigger = False

                if "csv_data" in st.session_state:

                    df = st.session_state.csv_data.copy()
                    email_column = st.session_state.email_column

                    predictions = []
                    confidence = []

                    progress = st.progress(0)

                    for i, email in enumerate(df[email_column].fillna("")):
                        result = analyze_email(str(email))
                        predictions.append(result["prediction"])
                        confidence.append(result["confidence"])
                        progress.progress((i + 1) / len(df))

                    df["Prediction"] = predictions
                    df["Confidence"] = confidence

                    st.success("Dataset Analysis Completed")
                    st.dataframe(df, use_container_width=True)

                    st.download_button(
                        "📥 Download Prediction Results",
                        df.to_csv(index=False).encode("utf-8"),
                        "prediction_results.csv",
                        "text/csv",
                        use_container_width=True
                    )

                else:
                    run_analysis_from_state()
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.subheader("Quick Highlights")
        model_cards = st.columns(2)
        with model_cards[0]:
            render_metric_card("Best Model", metadata.get("best_model_name", "Auto"), "Selected via F1 score")
        with model_cards[1]:
            render_metric_card("Dataset Size", metadata.get("dataset_size", 0), "Training records")
        dist = metadata.get("class_distribution", {})
        stats_cards = st.columns(2)
        with stats_cards[0]:
            render_metric_card("Legitimate", dist.get("legitimate", 0), "Training samples")
        with stats_cards[1]:
            render_metric_card("Phishing", dist.get("phishing", 0), "Training samples")
        st.markdown("</div>", unsafe_allow_html=True)

    result = st.session_state.analysis_result
    if not result:
        st.info("Run an analysis to unlock the full cybersecurity dashboard, charts, and downloadable reports.")
        render_history()
        return

    if st.session_state.active_page == "Dashboard":
        render_overview(result)
    elif st.session_state.active_page == "Indicators":
        render_analysis_tables(result)
    elif st.session_state.active_page == "Visuals":
        render_visuals(result)
    else:
        render_history()

    st.markdown("---")
    st.caption("Tip: replace the bundled sample dataset with a larger real phishing dataset in /dataset for stronger production performance.")


if __name__ == "__main__":
    main()
