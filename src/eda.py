import os
import sys
import json
import logging

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections import Counter

from src.preprocessing import clean_resume_text, assess_data_quality

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.autolayout'] = True

def run_data_quality_and_eda(
    input_csv: str = os.path.join("data", "processed", "extracted_resumes.csv"),
    output_cleaned_csv: str = os.path.join("data", "processed", "cleaned_resumes.csv"),
    plot_dir: str = os.path.join("data", "processed", "eda_plots")
):
    os.makedirs(plot_dir, exist_ok=True)
    logger.info(f"Loading extracted dataset from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # ---------------- STAGE 3: DATA QUALITY AUDIT ----------------
    logger.info("Executing Stage 3: Data Quality Audit...")
    quality_report = assess_data_quality(df)
    logger.info("Data Quality Metrics:")
    for k, v in quality_report.items():
        if k != "category_distribution":
            logger.info(f"  {k}: {v}")
            
    with open(os.path.join(plot_dir, "data_quality_report.json"), "w", encoding="utf-8") as f:
        json.dump(quality_report, f, indent=2)

    # ---------------- STAGE 5: PREPROCESSING ----------------
    logger.info("Executing Stage 5: Applying domain-aware preprocessing...")
    df['cleaned_text'] = df['raw_text'].astype(str).apply(clean_resume_text)
    df['cleaned_word_count'] = df['cleaned_text'].apply(lambda x: len(x.split()))
    df['cleaned_char_count'] = df['cleaned_text'].apply(len)
    
    initial_len = len(df)
    df = df[df['cleaned_word_count'] >= 10].reset_index(drop=True)
    logger.info(f"Filtered out {initial_len - len(df)} records with < 10 cleaned words. Retained {len(df)} records.")
    
    df.to_csv(output_cleaned_csv, index=False, encoding='utf-8')
    logger.info(f"Saved cleaned dataset to {output_cleaned_csv}")

    # ---------------- STAGE 4: EDA VISUALIZATIONS ----------------
    logger.info("Executing Stage 4: Generating EDA visualizations...")

    # 1. Class Distribution Plot
    plt.figure(figsize=(12, 7))
    cat_order = df['category'].value_counts()
    sns.barplot(x=cat_order.values, y=cat_order.index, palette='crest')
    plt.title("Resume Category Distribution (24 Classes)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Number of Resumes", fontsize=12)
    plt.ylabel("Category", fontsize=12)
    for idx, count in enumerate(cat_order.values):
        plt.text(count + 1, idx, f"{count}", va='center', fontsize=9, fontweight='semibold')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "class_distribution.png"), dpi=300)
    plt.close()

    # 2. Text Length Distribution (Raw vs Cleaned)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df['word_count'], bins=40, kde=True, ax=axes[0], color='#2563EB')
    axes[0].set_title("Raw Resume Word Count Distribution", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Word Count")
    axes[0].set_ylabel("Frequency")
    axes[0].axvline(df['word_count'].median(), color='red', linestyle='--', label=f'Median: {df["word_count"].median():.0f}')
    axes[0].legend()

    sns.histplot(df['cleaned_word_count'], bins=40, kde=True, ax=axes[1], color='#059669')
    axes[1].set_title("Cleaned Resume Word Count Distribution", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Cleaned Word Count")
    axes[1].set_ylabel("Frequency")
    axes[1].axvline(df['cleaned_word_count'].median(), color='red', linestyle='--', label=f'Median: {df["cleaned_word_count"].median():.0f}')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "resume_length_distribution.png"), dpi=300)
    plt.close()

    # 3. Top 30 Most Frequent Words
    all_words = " ".join(df['cleaned_text']).split()
    word_freq = Counter(all_words).most_common(30)
    words_df = pd.DataFrame(word_freq, columns=['Word', 'Frequency'])

    plt.figure(figsize=(10, 8))
    sns.barplot(data=words_df, x='Frequency', y='Word', palette='mako')
    plt.title("Top 30 Most Frequent Words Across All Resumes", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Frequency", fontsize=12)
    plt.ylabel("Word", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "top_words_frequency.png"), dpi=300)
    plt.close()

    # 4. Word Cloud (Global)
    wc = WordCloud(
        width=1200, height=600,
        background_color='white',
        colormap='Blues_r',
        max_words=150,
        random_state=42
    ).generate(" ".join(df['cleaned_text']))

    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title("Global Resume Corpus Word Cloud", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "wordcloud_global.png"), dpi=300)
    plt.close()

    # 5. N-gram Analysis (Unigrams, Bigrams, Trigrams)
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    for idx, (n, name) in enumerate([(1, "Unigrams"), (2, "Bigrams"), (3, "Trigrams")]):
        c_vec = CountVectorizer(ngram_range=(n, n), max_features=15)
        bow = c_vec.fit_transform(df['cleaned_text'])
        counts = bow.sum(axis=0).A1
        ngrams = c_vec.get_feature_names_out()
        ngram_df = pd.DataFrame({'ngram': ngrams, 'count': counts}).sort_values('count', ascending=True)

        axes[idx].barh(ngram_df['ngram'], ngram_df['count'], color=['#3B82F6', '#10B981', '#8B5CF6'][idx])
        axes[idx].set_title(f"Top 15 {name}", fontsize=12, fontweight='bold')
        axes[idx].set_xlabel("Count")

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "ngrams_analysis.png"), dpi=300)
    plt.close()

    # 6. Class-Specific Distinctive Vocabulary (TF-IDF per class)
    logger.info("Computing distinctive class-specific vocabularies...")
    class_texts = df.groupby('category')['cleaned_text'].apply(lambda x: " ".join(x)).reset_index()
    tfidf_class = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    matrix = tfidf_class.fit_transform(class_texts['cleaned_text'])
    terms = np.array(tfidf_class.get_feature_names_out())

    class_top_terms = {}
    for i, row in class_texts.iterrows():
        row_vec = matrix[i].toarray().flatten()
        top_indices = row_vec.argsort()[-8:][::-1]
        cat_name = row['category']
        class_top_terms[cat_name] = [str(terms[idx]) for idx in top_indices]

    with open(os.path.join(plot_dir, "class_distinctive_terms.json"), "w", encoding="utf-8") as f:
        json.dump(class_top_terms, f, indent=2)

    logger.info("All EDA plots and metrics successfully generated!")
    return df, quality_report, class_top_terms

if __name__ == "__main__":
    df, quality, vocab = run_data_quality_and_eda()
    print("EDA Complete. Sample class vocabularies:")
    for cat in ['INFORMATION-TECHNOLOGY', 'FINANCE', 'HEALTHCARE', 'CHEF']:
        if cat in vocab:
            print(f"  {cat}: {vocab[cat]}")
