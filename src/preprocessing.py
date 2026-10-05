import re
import string
import logging
from typing import Dict, List, Tuple, Set
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize NLTK components
try:
    STOP_WORDS: Set[str] = set(stopwords.words('english'))
except Exception:
    nltk.download('stopwords')
    STOP_WORDS = set(stopwords.words('english'))

try:
    LEMMATIZER = WordNetLemmatizer()
except Exception:
    nltk.download('wordnet')
    LEMMATIZER = WordNetLemmatizer()

# Domain and Tech Term Preservation Map
# Prevents punctuation removal and standard tokenizers from destroying crucial symbols
TECH_PRESERVE_RULES = [
    (re.compile(r'\bC\+\+(?!\w)', re.IGNORECASE), ' cplusplus '),
    (re.compile(r'\bC#(?!\w)', re.IGNORECASE), ' csharp '),
    (re.compile(r'(?:(?<=[\s,;.])|^)\.NET(?!\w)', re.IGNORECASE), ' dotnet '),
    (re.compile(r'\bNODE(?:\.JS)?(?!\w)', re.IGNORECASE), ' nodejs '),
    (re.compile(r'\bVUE(?:\.JS)?(?!\w)', re.IGNORECASE), ' vuejs '),
    (re.compile(r'\bPOWER\s*BI(?!\w)', re.IGNORECASE), ' powerbi '),
    (re.compile(r'\bPL[\/\\]SQL(?!\w)', re.IGNORECASE), ' plsql '),
    (re.compile(r'\bMS\s*EXCEL(?!\w)', re.IGNORECASE), ' msexcel '),
    (re.compile(r'\bMS\s*OFFICE(?!\w)', re.IGNORECASE), ' msoffice '),
    (re.compile(r'\bMACHINE\s+LEARNING(?!\w)', re.IGNORECASE), ' machine_learning '),
    (re.compile(r'\bDEEP\s+LEARNING(?!\w)', re.IGNORECASE), ' deep_learning '),
    (re.compile(r'\bDATA\s+SCIENCE(?!\w)', re.IGNORECASE), ' data_science '),
    (re.compile(r'\bNATURAL\s+LANGUAGE\s+PROCESSING(?!\w)', re.IGNORECASE), ' nlp '),
    (re.compile(r'\bAI[\/\\]ML(?!\w)', re.IGNORECASE), ' aiml '),
    (re.compile(r'\bCI[\/\\]CD(?!\w)', re.IGNORECASE), ' cicd '),
]

# Important technical keywords to highlight/extract during inference
KNOWN_TECH_SKILLS = {
    'python', 'cplusplus', 'csharp', 'dotnet', 'sql', 'plsql', 'aws', 'azure', 'gcp',
    'tensorflow', 'pytorch', 'keras', 'java', 'nlp', 'machine_learning', 'deep_learning',
    'data_science', 'powerbi', 'docker', 'kubernetes', 'html', 'css', 'javascript',
    'react', 'nodejs', 'vuejs', 'linux', 'git', 'spark', 'hadoop', 'tableau',
    'excel', 'msexcel', 'scikit-learn', 'pandas', 'numpy', 'sap', 'salesforce', 'jira'
}

def clean_resume_text(
    text: str,
    remove_stopwords: bool = True,
    lemmatize: bool = True,
    min_word_length: int = 2
) -> str:
    """
    Robust, reproducible resume text preprocessor that:
    1. Normalizes whitespace and line breaks
    2. Removes HTML tags and markup
    3. Cleans URLs, emails, and phone numbers
    4. PRESERVES specialized technical keywords (C++, .NET, C#, Python, SQL, AWS, etc.)
    5. Strips non-alphanumeric noise
    6. Performs optional stopword removal & lemmatization
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Normalize line breaks and tabs
    text = re.sub(r'[\r\n\t]+', ' ', text)

    # 2. Remove HTML tags and markup
    text = re.sub(r'<[^>]+>', ' ', text)

    # 3. Mask URLs and emails
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' ', text)
    
    # 4. Remove phone numbers and repeated special bullet characters
    text = re.sub(r'\+?\d[\d -]{8,}\d', ' ', text)
    text = re.sub(r'[\u2022\u2023\u25E6\u2043\u2219\u25ba\uf0b7\uf0a7\u00b7\x95\x96\x97]', ' ', text)

    # 5. Apply Tech Term Preservation Rules before general punctuation removal
    for pattern, replacement in TECH_PRESERVE_RULES:
        text = pattern.sub(replacement, text)

    # 6. Lowercase
    text = text.lower()

    # 7. Remove remaining punctuation and non-alphanumeric characters (keeping underscores for compound tokens)
    text = re.sub(r'[^a-z0-9_\s]', ' ', text)

    # 8. Collapse whitespace
    tokens = text.split()

    # 9. Stopword filtering & Lemmatization
    cleaned_tokens = []
    for token in tokens:
        if len(token) < min_word_length and token not in {'r'}:
            continue
        if remove_stopwords and token in STOP_WORDS:
            continue
        if lemmatize:
            token = LEMMATIZER.lemmatize(token)
        cleaned_tokens.append(token)

    return ' '.join(cleaned_tokens)

def extract_skills_from_text(raw_or_cleaned_text: str) -> List[str]:
    """
    Extracts recognized technical skills and domain tokens present in the text.
    Useful for business logic, explainability, and recruiter review.
    """
    cleaned = clean_resume_text(raw_or_cleaned_text, remove_stopwords=False, lemmatize=False)
    words = set(cleaned.split())
    found_skills = sorted(list(KNOWN_TECH_SKILLS.intersection(words)))
    # Map normalized names back to display friendly names
    display_map = {
        'cplusplus': 'C++',
        'csharp': 'C#',
        'dotnet': '.NET',
        'nodejs': 'Node.js',
        'vuejs': 'Vue.js',
        'powerbi': 'Power BI',
        'plsql': 'PL/SQL',
        'msexcel': 'MS Excel',
        'msoffice': 'MS Office',
        'machine_learning': 'Machine Learning',
        'deep_learning': 'Deep Learning',
        'data_science': 'Data Science',
        'nlp': 'NLP',
        'sql': 'SQL',
        'aws': 'AWS',
        'gcp': 'GCP',
        'azure': 'Azure',
        'html': 'HTML',
        'css': 'CSS',
        'javascript': 'JavaScript',
        'python': 'Python',
        'java': 'Java',
        'tensorflow': 'TensorFlow',
        'pytorch': 'PyTorch',
        'keras': 'Keras',
        'docker': 'Docker',
        'kubernetes': 'Kubernetes',
        'linux': 'Linux',
        'git': 'Git',
        'spark': 'Apache Spark',
        'hadoop': 'Hadoop',
        'tableau': 'Tableau',
        'sap': 'SAP',
        'salesforce': 'Salesforce',
        'jira': 'Jira'
    }
    return [display_map.get(s, s.upper()) for s in found_skills]

def assess_data_quality(df: pd.DataFrame) -> Dict:
    """
    Performs comprehensive data quality audit:
    - Missing/null records
    - Empty resume texts
    - Exact duplicate resumes
    - Short resumes (< 30 words or < 150 chars)
    - Class imbalance metrics
    """
    total = len(df)
    missing_category = int(df['category'].isnull().sum())
    missing_text = int(df['raw_text'].isnull().sum())
    empty_text = int((df['raw_text'].str.strip() == '').sum())
    
    # Duplicates based on raw text
    duplicate_texts = int(df.duplicated(subset=['raw_text']).sum())
    
    # Word count and char count stats
    word_counts = df['raw_text'].apply(lambda x: len(str(x).split()))
    short_resumes = int((word_counts < 30).sum())
    
    cat_counts = df['category'].value_counts()
    imbalance_ratio = float(cat_counts.max() / cat_counts.min()) if len(cat_counts) > 0 else 0.0

    return {
        'total_resumes': total,
        'missing_category': missing_category,
        'missing_text': missing_text,
        'empty_text': empty_text,
        'duplicate_texts': duplicate_texts,
        'short_resumes_lt_30_words': short_resumes,
        'unique_categories': len(cat_counts),
        'min_category_count': int(cat_counts.min()),
        'max_category_count': int(cat_counts.max()),
        'imbalance_ratio': round(imbalance_ratio, 2),
        'category_distribution': cat_counts.to_dict()
    }

if __name__ == "__main__":
    sample = "Senior Engineer skilled in C++, .NET, C#, Python, SQL, AWS, TensorFlow, Java, and NLP. Contact: dev@company.com at http://myrepo.com"
    cleaned = clean_resume_text(sample)
    skills = extract_skills_from_text(sample)
    print("--- Preprocessing Verification ---")
    print("Raw sample:    ", sample)
    print("Cleaned text:  ", cleaned)
    print("Skills found:  ", skills)
    
    assert "cplusplus" in cleaned, "C++ preservation failed!"
    assert "dotnet" in cleaned, ".NET preservation failed!"
    assert "csharp" in cleaned, "C# preservation failed!"
    assert "python" in cleaned, "Python preservation failed!"
    assert "sql" in cleaned, "SQL preservation failed!"
    assert "aws" in cleaned, "AWS preservation failed!"
    assert "tensorflow" in cleaned, "TensorFlow preservation failed!"
    assert "nlp" in cleaned, "NLP preservation failed!"
    print("ALL TECHNICAL TOKEN PRESERVATION TESTS PASSED WITH 100% SUCCESS!")
