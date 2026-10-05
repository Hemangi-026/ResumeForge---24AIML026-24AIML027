import os
import sys
import streamlit as st
import pandas as pd
import json

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.inference import ResumeClassifierPipeline

# Streamlit Page Configuration - Simple, professional theme
st.set_page_config(
    page_title="ResumeForge 2026 — AI Resume Classifier",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Single-Color CSS Theme (Slate Blue / Corporate Minimalist)
st.markdown("""
<style>
    :root {
        --primary-color: #1E3A8A;
        --secondary-color: #3B82F6;
        --bg-color: #F8FAFC;
        --text-color: #0F172A;
    }
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .pred-card {
        background-color: #EFF6FF;
        border-left: 6px solid #1E3A8A;
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1.2rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E3A8A;
    }
    .skill-badge {
        display: inline-block;
        background-color: #E2E8F0;
        color: #1E293B;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 3px 4px 3px 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_pipeline():
    return ResumeClassifierPipeline()

pipeline = get_pipeline()

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown("### ⚙️ Pipeline Settings")
    model_choice = st.radio(
        "Classifier Model Engine",
        options=["Word2Vec + BiLSTM (Deep Learning)", "Linear SVM (Classical TF-IDF)"],
        index=0,
        help="Select between the neural BiLSTM model (84.3% Acc) or the calibrated classical SVM baseline."
    )
    model_type = "dl" if "BiLSTM" in model_choice else "classical"
    
    st.markdown("---")
    st.markdown("### 📊 System Metadata")
    st.markdown("""
    - **Classes**: 24 Job Categories
    - **Corpus**: 2,500 Resumes
    - **Tech Keyword Protection**: Active
    - **Macro-F1**: 80.11% (BiLSTM)
    """)

    st.markdown("---")
    st.markdown("### 🧪 Quick Test Samples")
    sample_text = ""
    if st.button("Sample 1: IT / Cloud Engineer"):
        sample_text = "Cloud Solutions Architect with 7 years experience in Python, AWS, Docker, Kubernetes, and SQL. Deployed scalable microservices and built CI/CD pipelines."
    if st.button("Sample 2: Financial Analyst"):
        sample_text = "Senior Financial Analyst specializing in financial modeling, balance sheet reconciliation, SAP ERP, corporate budgeting, variance analysis, and Excel macros."
    if st.button("Sample 3: Chef / Culinary"):
        sample_text = "Executive Chef with 10 years leading high-volume restaurant kitchens, food safety sanitation, inventory procurement, culinary staff training, and fine dining recipe design."

# ----------------- MAIN VIEW -----------------
st.markdown("<div class='main-header'>Samatrix ResumeForge 2026</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>End-to-End Multi-Class Resume Classification System (24 Categories)</div>", unsafe_allow_html=True)

tab_upload, tab_paste, tab_benchmark = st.tabs(["📤 Upload PDF Resume", "📝 Paste Resume Text", "📈 Model Benchmarks"])

# Tab 1: PDF Upload
with tab_upload:
    uploaded_file = st.file_uploader("Upload a candidate's resume (PDF format)", type=["pdf"])
    if uploaded_file is not None:
        with st.spinner("Extracting text and running classification..."):
            file_bytes = uploaded_file.read()
            result = pipeline.predict(file_bytes, is_pdf=True, model_type=model_type)

        if not result.get('success', False):
            st.error(result.get('error', 'Failed to process PDF.'))
        else:
            col1, col2 = st.columns([1.5, 1])
            with col1:
                st.markdown(f"""
                <div class='pred-card'>
                    <div style='color: #475569; font-size: 0.9rem; font-weight: 600;'>PREDICTED JOB CATEGORY</div>
                    <div class='metric-value'>{result['predicted_category']}</div>
                    <div style='color: #1E3A8A; font-weight: 600; font-size: 1.1rem; margin-top: 4px;'>Confidence: {result['confidence']}%</div>
                    <div style='color: #64748B; font-size: 0.85rem; margin-top: 6px;'>Engine: {result['model_used']}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("#### 🎯 Top 3 Category Probabilities")
                for item in result['top_3_predictions']:
                    st.write(f"**{item['category']}**: {item['probability']}%")
                    st.progress(float(item['probability']) / 100.0)

            with col2:
                st.markdown("#### 🛠️ Detected Technical Skills")
                skills = result['extracted_skills']
                if skills:
                    badges = "".join([f"<span class='skill-badge'>{s}</span>" for s in skills])
                    st.markdown(f"<div>{badges}</div>", unsafe_allow_html=True)
                else:
                    st.info("No standardized technical keywords detected in this resume.")

                st.markdown("---")
                st.markdown("#### 📑 Document Metrics")
                st.write(f"• **Raw Word Count**: {result['raw_word_count']}")
                st.write(f"• **Raw Character Count**: {result['raw_char_count']}")

            with st.expander("🔍 View Preprocessed Cleaned Text"):
                st.write(result['cleaned_text_preview'])

# Tab 2: Paste Text
with tab_paste:
    input_text = st.text_area(
        "Paste resume text below:",
        value=sample_text,
        height=180,
        placeholder="e.g. Experienced software engineer skilled in Python, SQL, C++, AWS..."
    )
    if st.button("Classify Text Resume", type="primary"):
        if not input_text.strip():
            st.warning("Please enter some resume text first.")
        else:
            with st.spinner("Processing text..."):
                result = pipeline.predict(input_text, is_pdf=False, model_type=model_type)

            if not result.get('success', False):
                st.error(result.get('error', 'Text too short.'))
            else:
                col1, col2 = st.columns([1.5, 1])
                with col1:
                    st.markdown(f"""
                    <div class='pred-card'>
                        <div style='color: #475569; font-size: 0.9rem; font-weight: 600;'>PREDICTED JOB CATEGORY</div>
                        <div class='metric-value'>{result['predicted_category']}</div>
                        <div style='color: #1E3A8A; font-weight: 600; font-size: 1.1rem; margin-top: 4px;'>Confidence: {result['confidence']}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("#### 🎯 Top 3 Category Probabilities")
                    for item in result['top_3_predictions']:
                        st.write(f"**{item['category']}**: {item['probability']}%")
                        st.progress(float(item['probability']) / 100.0)

                with col2:
                    st.markdown("#### 🛠️ Detected Technical Skills")
                    skills = result['extracted_skills']
                    if skills:
                        badges = "".join([f"<span class='skill-badge'>{s}</span>" for s in skills])
                        st.markdown(f"<div>{badges}</div>", unsafe_allow_html=True)
                    else:
                        st.info("No standardized technical keywords detected.")

# Tab 3: Model Benchmark Summary
with tab_benchmark:
    st.markdown("### 🏆 Official Model Comparison Benchmark (Test Set)")
    comp_csv = "models/model_comparison.csv"
    if os.path.exists(comp_csv):
        df_comp = pd.read_csv(comp_csv)
        st.dataframe(df_comp, use_container_width=True)
    else:
        st.info("Model comparison data not found. Run train.py first.")

    cm_path = "models/confusion_matrix.png"
    if os.path.exists(cm_path):
        st.markdown("### 📊 24-Class Normalized Confusion Matrix")
        st.image(cm_path, use_column_width=True)

    err_path = "models/error_analysis.csv"
    if os.path.exists(err_path):
        st.markdown("### 🔬 Test Set Error Analysis (Sample Failures)")
        err_df = pd.read_csv(err_path)
        st.write(f"Total misclassifications on unseen test set: **{len(err_df)}** / 375")
        st.dataframe(err_df[['actual_category', 'predicted_category', 'confidence', 'primary_reason']].head(10), use_container_width=True)
