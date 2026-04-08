import streamlit as st
import sys
import os
import json
import tempfile
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

sys.path.append(".")
from utils.pdf_parser import parse_pdf
from utils.similarity import ClauseSimilarityChecker
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Insurance Document Intelligence",
    page_icon="🛡️",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.entity-tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 13px;
    font-weight: 600;
    margin: 2px;
}
.metric-card {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
}
.anomaly-high { background:#ffe5e5; border-left:4px solid #e53e3e; padding:8px 12px; border-radius:4px; margin:4px 0; }
.anomaly-medium { background:#fff8e1; border-left:4px solid #f6ad55; padding:8px 12px; border-radius:4px; margin:4px 0; }
.present { background:#e6ffed; border-left:4px solid #38a169; padding:8px 12px; border-radius:4px; margin:4px 0; }
</style>
""", unsafe_allow_html=True)

ENTITY_COLORS = {
    "INSURED":      "#bee3f8",
    "COVERAGE":     "#c6f6d5",
    "PREMIUM":      "#fefcbf",
    "POLICY_DATE":  "#e9d8fd",
    "EXCLUSION":    "#fed7d7",
    "POLICY_LIMIT": "#feebc8",
}

# ── Load models (cached) ──────────────────────────────────────────────────────
@st.cache_resource
def load_ner_model():
    model_path = "models/bert-insurance-ner/best"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    ner = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple"
    )
    return ner

@st.cache_resource
def load_similarity_model():
    return ClauseSimilarityChecker()

# ── Entity extraction ─────────────────────────────────────────────────────────
def extract_entities(ner_pipeline, sentences):
    all_entities = []
    for sent in sentences[:80]:
        try:
            results = ner_pipeline(sent)
            for r in results:
                label = r["entity_group"].replace("B-","").replace("I-","")
                if label != "O" and r["score"] > 0.5:
                    all_entities.append({
                        "text": r["word"].replace("##",""),
                        "label": label,
                        "score": round(r["score"], 4),
                        "sentence": sent[:120]
                    })
        except:
            continue
    return all_entities

def highlight_entities(text, entities):
    highlighted = text
    seen = set()
    for ent in entities:
        word = ent["text"]
        label = ent["label"]
        if word in seen or len(word) < 3:
            continue
        seen.add(word)
        color = ENTITY_COLORS.get(label, "#e2e8f0")
        highlighted = highlighted.replace(
            word,
            f'<span class="entity-tag" style="background:{color}">{word} <small>{label}</small></span>',
            1
        )
    return highlighted

# ── Plotly charts ─────────────────────────────────────────────────────────────
def entity_bar_chart(entities):
    if not entities:
        return None
    df = pd.DataFrame(entities)
    counts = df["label"].value_counts().reset_index()
    counts.columns = ["Entity Type", "Count"]
    colors = [ENTITY_COLORS.get(l, "#ccc") for l in counts["Entity Type"]]
    fig = go.Figure(go.Bar(
        x=counts["Entity Type"],
        y=counts["Count"],
        marker_color=colors,
        text=counts["Count"],
        textposition="outside"
    ))
    fig.update_layout(
        title="Extracted entities by type",
        xaxis_title="Entity Type",
        yaxis_title="Count",
        plot_bgcolor="white",
        height=320,
        margin=dict(t=40, b=20)
    )
    return fig

def similarity_gauge_chart(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score * 100, 1),
        title={"text": "Policy completeness %"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#38a169"},
            "steps": [
                {"range": [0, 40],  "color": "#fed7d7"},
                {"range": [40, 70], "color": "#fefcbf"},
                {"range": [70, 100],"color": "#c6f6d5"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 60
            }
        }
    ))
    fig.update_layout(height=280, margin=dict(t=30, b=10))
    return fig

def clause_heatmap(similarity_results):
    labels = [k.replace("_", " ").title() for k in similarity_results]
    scores = [v["score"] for v in similarity_results.values()]
    colors = ["#38a169" if s >= 0.45 else "#e53e3e" for s in scores]
    fig = go.Figure(go.Bar(
        x=scores,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{s:.2f}" for s in scores],
        textposition="outside"
    ))
    fig.add_vline(x=0.45, line_dash="dash", line_color="gray",
                  annotation_text="threshold")
    fig.update_layout(
        title="Clause similarity scores",
        xaxis=dict(range=[0, 1]),
        plot_bgcolor="white",
        height=380,
        margin=dict(t=40, b=20, l=180)
    )
    return fig

# ── Main app ──────────────────────────────────────────────────────────────────
def main():
    st.title("🛡️ Insurance Document Intelligence System")
    st.markdown("*NLP-powered entity extraction and risk coverage analysis*")
    st.divider()

    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This system uses:
        - **BERT** fine-tuned for insurance NER
        - **Sentence-BERT** for clause similarity
        - **6 entity types**: Insured, Coverage, Premium, Policy Date, Exclusion, Policy Limit
        - **12 standard clauses** checked per document
        """)
        st.divider()
        st.markdown("**Model:** `bert-base-uncased` fine-tuned")
        st.markdown("**Similarity:** `all-MiniLM-L6-v2`")
        st.markdown("**Threshold:** 0.45 cosine similarity")

    # Upload
    uploaded_file = st.file_uploader(
        "Upload an insurance policy PDF",
        type=["pdf"],
        help="Upload any insurance policy, certificate, or endorsement PDF"
    )

    if uploaded_file is None:
        st.info("Upload a PDF above to begin analysis.")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("#### 🔍 Entity Extraction\nIdentifies insured names, coverage types, premiums, dates, exclusions and policy limits using fine-tuned BERT.")
        with col2:
            st.markdown("#### 📊 Clause Analysis\nCompares your document against 12 standard insurance clauses using cosine similarity on sentence embeddings.")
        with col3:
            st.markdown("#### ⚠️ Anomaly Detection\nFlags missing critical clauses with severity levels so underwriters know what to review.")
        return

    # Process
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("Parsing PDF..."):
        doc = parse_pdf(tmp_path)
        sentences = doc["sentences"]

    os.unlink(tmp_path)

    if not sentences:
        st.error("Could not extract text from this PDF. It may be a scanned image.")
        return

    st.success(f"Parsed **{len(sentences)}** sentences from **{uploaded_file.name}**")

    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)

    with st.spinner("Running BERT NER..."):
        ner = load_ner_model()
        entities = extract_entities(ner, sentences)

    with st.spinner("Running clause similarity..."):
        checker = load_similarity_model()
        sim_results = checker.check_document(sentences)
        stats = checker.get_summary_stats(sim_results)
        anomalies = checker.get_anomalies(sim_results)

    with col1:
        st.metric("Sentences extracted", len(sentences))
    with col2:
        st.metric("Entities found", len(entities))
    with col3:
        st.metric("Clauses present", f"{stats['present']}/{stats['total_clauses_checked']}")
    with col4:
        st.metric("Anomalies flagged", len(anomalies),
                  delta=f"{len([a for a in anomalies if a['severity']=='HIGH'])} HIGH",
                  delta_color="inverse")

    st.divider()

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Entity Extraction",
        "📊 Clause Analysis",
        "⚠️ Anomalies",
        "📄 Raw Text"
    ])

    # Tab 1 — Entities
    with tab1:
        st.subheader("Extracted entities")

        if not entities:
            st.warning("No entities detected above confidence threshold.")
        else:
            col_chart, col_legend = st.columns([2, 1])
            with col_chart:
                fig = entity_bar_chart(entities)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            with col_legend:
                st.markdown("**Entity color legend**")
                for label, color in ENTITY_COLORS.items():
                    st.markdown(
                        f'<span class="entity-tag" style="background:{color}">{label}</span>',
                        unsafe_allow_html=True
                    )

            st.subheader("Highlighted sentences")
            shown = 0
            for sent in sentences[:60]:
                sent_ents = [e for e in entities if e["sentence"].startswith(sent[:50])]
                if sent_ents:
                    highlighted = highlight_entities(sent, sent_ents)
                    st.markdown(highlighted, unsafe_allow_html=True)
                    shown += 1
                if shown >= 15:
                    break

            st.subheader("Entity table")
            if entities:
                df = pd.DataFrame(entities)[["label","text","score","sentence"]]
                df.columns = ["Entity Type","Extracted Text","Confidence","Source Sentence"]
                st.dataframe(df, use_container_width=True, height=300)

    # Tab 2 — Clause Analysis
    with tab2:
        st.subheader("Clause coverage analysis")
        col_gauge, col_heat = st.columns([1, 2])
        with col_gauge:
            gauge = similarity_gauge_chart(stats["coverage_percentage"] / 100)
            st.plotly_chart(gauge, use_container_width=True)
            st.metric("Avg similarity score", stats["avg_similarity_score"])
        with col_heat:
            heat = clause_heatmap(sim_results)
            st.plotly_chart(heat, use_container_width=True)

        st.subheader("Clause-by-clause breakdown")
        for key, val in sim_results.items():
            label = key.replace("_", " ").title()
            css_class = "present" if val["found"] else "anomaly-high"
            icon = "✓" if val["found"] else "✗"
            match_text = f"<br><small><i>Best match: {val['best_match'][:100]}...</i></small>" if val["best_match"] else ""
            st.markdown(
                f'<div class="{css_class}"><b>{icon} {label}</b> — score: {val["score"]}{match_text}</div>',
                unsafe_allow_html=True
            )

    # Tab 3 — Anomalies
    with tab3:
        st.subheader("Anomaly report")
        if not anomalies:
            st.success("No anomalies detected. All standard clauses are present.")
        else:
            high = [a for a in anomalies if a["severity"] == "HIGH"]
            medium = [a for a in anomalies if a["severity"] == "MEDIUM"]

            if high:
                st.error(f"🚨 {len(high)} HIGH severity anomalies — critical clauses missing")
                for a in high:
                    st.markdown(
                        f'<div class="anomaly-high"><b>HIGH: {a["clause"]}</b><br>'
                        f'Similarity score: {a["score"]} (threshold: 0.45)<br>'
                        f'<small>This clause is critical and appears to be absent from the document.</small></div>',
                        unsafe_allow_html=True
                    )

            if medium:
                st.warning(f"⚠️ {len(medium)} MEDIUM severity anomalies")
                for a in medium:
                    st.markdown(
                        f'<div class="anomaly-medium"><b>MEDIUM: {a["clause"]}</b><br>'
                        f'Similarity score: {a["score"]} (threshold: 0.45)</div>',
                        unsafe_allow_html=True
                    )

            st.divider()
            st.markdown("**What this means for underwriters:**")
            st.markdown("""
            - HIGH anomalies indicate potentially missing mandatory clauses that could create coverage gaps
            - MEDIUM anomalies should be reviewed but may be present under different wording
            - Similarity scores below 0.30 strongly suggest the clause is absent
            - Scores between 0.30–0.45 suggest possible paraphrasing — manual review recommended
            """)

    # Tab 4 — Raw text
    with tab4:
        st.subheader("Extracted text")
        st.text_area("Full document text", doc["full_text"][:5000], height=400)
        st.caption(f"Showing first 5000 characters of {len(doc['full_text'])} total")

if __name__ == "__main__":
    main()