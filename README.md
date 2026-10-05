# Samatrix ResumeForge 2026: End-to-End Resume Classification System

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14-EE4C2C.svg)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.65-FF4B4B.svg)](https://streamlit.io/)

A complete, production-grade NLP and Machine Learning solution for multi-class resume classification across **24 professional job categories**, developed for the **Samatrix ResumeForge 2026 Hackathon**.

---

## 📌 Project Overview & Highlights

- **24 Target Categories**: ACCOUNTANT, ADVOCATE, AGRICULTURE, APPAREL, ARTS, AUTOMOBILE, AVIATION, BANKING, BPO, BUSINESS-DEVELOPMENT, CHEF, CONSTRUCTION, CONSULTANT, DESIGNER, DIGITAL-MEDIA, ENGINEERING, FINANCE, FITNESS, HEALTHCARE, HR, INFORMATION-TECHNOLOGY, PUBLIC-RELATIONS, SALES, TEACHER.
- **Domain-Aware Preprocessing**: Preserves crucial technical terms (C++, .NET, C#, Python, SQL, AWS, TensorFlow, Java, NLP, PowerBI, Docker, PyTorch) using regex protection rules.
- **Data Integrity & Leakage Prevention**: Stratified 70/15/15 train-validation-test split strictly executed *before* TF-IDF vectorization or vocabulary fitting.
- **Dual Modeling Paradigm**:
  - **Classical ML Baselines**: Multinomial Naive Bayes, Balanced Logistic Regression, and Calibrated Linear SVM.
  - **Deep Learning Neural Model**: Word2Vec Continuous Skip-Gram embeddings + 2-layer Bidirectional LSTM with combined Global Max & Mean pooling and class-weighted CrossEntropyLoss.
- **Single Master Notebook**: Clean, executive-ready notebook covering all 10 stages (
otebooks/resume_classification_pipeline.ipynb).
- **Minimalist Streamlit UI**: Single-color theme focusing strictly on business logic (PDF upload, text parsing, predicted category, confidence score, top-3 breakdown, and extracted skills).

---

## 🏆 Model Benchmarking & Performance Comparison

Evaluated on the held-out, unseen test set (375 resumes across 24 classes):

| Model | Representation | Test Accuracy | Macro-F1 | Weighted-F1 | Macro-Precision | Macro-Recall |
|---|---|---|---|---|---|---|
| **Word2Vec + BiLSTM (DL)** | **Word2Vec (150-d) + BiLSTM** | **84.27%** | **80.11%** | **83.55%** | **80.92%** | **80.54%** |
| Linear SVM (Calibrated) | TF-IDF (1,2) N-Grams | 67.47% | 62.89% | 66.39% | 63.63% | 63.55% |
| Logistic Regression | TF-IDF (1,2) N-Grams | 65.07% | 62.09% | 63.88% | 67.35% | 62.14% |
| Multinomial Naive Bayes | TF-IDF (1,2) N-Grams | 55.47% | 49.85% | 53.21% | 53.08% | 51.65% |

> **Key Finding**: The neural **Word2Vec + BiLSTM** model achieved the highest score (**80.11% Macro-F1**, **84.27% Accuracy**), outperforming the strongest classical baseline by over **17% Macro-F1**.

---

## 📁 Repository Structure

`
samatrix/
├── data/
│   ├── raw/                           # 2,500 extracted PDFs organized in 24 category folders
│   └── processed/
│       ├── extracted_resumes.csv      # Raw text, counts, and metadata from all PDFs
│       ├── cleaned_resumes.csv        # Preprocessed text with protected tech tokens
│       └── eda_plots/                 # High-resolution EDA figures & reports
├── models/
│   ├── best_model.joblib              # Calibrated Linear SVM
│   ├── best_bilstm.pt                 # Best PyTorch BiLSTM neural checkpoint
│   ├── tfidf_vectorizer.joblib        # Fitted TF-IDF vectorizer (8,000 features)
│   ├── label_encoder.joblib           # Fitted LabelEncoder for 24 classes
│   ├── word2vec.model                 # Trained Word2Vec skip-gram embeddings
│   ├── word2idx.json                  # Vocabulary token index mapping
│   ├── model_comparison.csv           # Benchmark results table
│   ├── confusion_matrix.png           # 24x24 normalized confusion matrix
│   └── error_analysis.csv             # Test set misclassification audit
├── notebooks/
│   └── resume_classification_pipeline.ipynb # Single comprehensive master notebook (Stages 1-10)
├── src/
│   ├── __init__.py
│   ├── pdf_extractor.py               # Robust PDF text parser (pypdf + pdfplumber)
│   ├── preprocessing.py               # Tech token preservation, cleaning, and skill extraction
│   ├── features.py                    # Stratified splitting, TF-IDF, and Word2Vec sequences
│   ├── models_classical.py            # Naive Bayes, Logistic Regression, Linear SVM
│   ├── models_dl.py                   # PyTorch ResumeBiLSTM network architecture
│   ├── evaluate.py                    # Metric calculations, confusion matrix, error analysis
│   └── inference.py                   # Unified prediction pipeline for PDFs and raw text
├── app.py                             # Clean, single-color Streamlit demo
├── train.py                           # Full pipeline training runner
├── requirements.txt                   # Dependency specifications
└── README.md                          # Documentation
`

---

## 🚀 Setup & Execution Guide

### 1. Virtual Environment & Dependencies
`ash
# Create Python 3.12 virtual environment
py -3.12 -m venv venv

# Activate virtual environment
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
`

### 2. Run Data Quality Audit & EDA (Stages 3 & 4)
`ash
python src/eda.py
`
Outputs high-resolution charts in data/processed/eda_plots/:
- class_distribution.png: Category frequency plot
- esume_length_distribution.png: Word count histograms
- 	op_words_frequency.png: Top 30 global vocabulary words
- wordcloud_global.png: Corpus word cloud
- 
grams_analysis.png: Unigram, bigram, and trigram frequencies
- class_distinctive_terms.json: Top TF-IDF keywords per class

### 3. Run Model Training & Evaluation (Stages 6, 7, 8, 9)
`ash
python train.py
`
Trains all 4 models, calculates test metrics, saves models to models/, generates models/confusion_matrix.png, and exports models/error_analysis.csv.

### 4. Launch the Master Notebook
`ash
jupyter notebook notebooks/resume_classification_pipeline.ipynb
`

### 5. Launch the Streamlit Demo Application (Stage 10)
`ash
streamlit run app.py
`
Access the application at http://localhost:8501. Features include:
- PDF resume drag-and-drop file upload
- Instant text parsing and word count statistics
- Top predicted category badge with calibrated confidence score
- Top-3 probable categories breakdown with progress bars
- Extracted technical skills chips (Python, SQL, AWS, C++, etc.)

---

## 🔬 Error Analysis & Insights (Stage 9)

An audit of the 59 misclassifications on the held-out test set revealed:
1. **Cross-Functional & Hybrid Careers (94.9% of errors)**: Candidates possessing inter-disciplinary experience (e.g. an IT administrator working in a hospital or an accountant in a construction firm) contain overlapping vocabulary that creates natural ambiguity between their functional domain and industry domain.
2. **Commercial & Client-Facing Overlap**: High semantic proximity between BUSINESS-DEVELOPMENT, SALES, and CONSULTANT.
3. **Financial Role Convergence**: Shared terminology across ACCOUNTANT, BANKING, and FINANCE.
