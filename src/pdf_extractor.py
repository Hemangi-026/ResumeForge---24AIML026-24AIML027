import os
import io
import logging
from typing import Optional, Union, List, Dict
import pandas as pd
from pypdf import PdfReader
import pdfplumber
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_text_from_pdf(source: Union[str, bytes, io.BytesIO]) -> str:
    """
    Extracts plain text from a PDF file path or byte stream.
    Tries pypdf first; falls back to pdfplumber if text is empty or errors occur.
    """
    text_chunks: List[str] = []
    
    # Try with pypdf
    try:
        if isinstance(source, bytes):
            stream = io.BytesIO(source)
            reader = PdfReader(stream)
        elif isinstance(source, io.BytesIO):
            source.seek(0)
            reader = PdfReader(source)
        else:
            reader = PdfReader(source)
            
        for page_idx, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text_chunks.append(page_text.strip())
    except Exception as e:
        logger.debug(f"pypdf extraction failed or partial: {e}")

    extracted = "\n".join(text_chunks).strip()
    if extracted:
        return extracted

    # Fallback to pdfplumber
    try:
        if isinstance(source, (bytes, io.BytesIO)):
            if isinstance(source, bytes):
                stream = io.BytesIO(source)
            else:
                source.seek(0)
                stream = source
            with pdfplumber.open(stream) as pdf:
                for page in pdf.pages:
                    p_text = page.extract_text()
                    if p_text:
                        text_chunks.append(p_text.strip())
        else:
            with pdfplumber.open(source) as pdf:
                for page in pdf.pages:
                    p_text = page.extract_text()
                    if p_text:
                        text_chunks.append(p_text.strip())
    except Exception as e:
        logger.warning(f"pdfplumber extraction error: {e}")

    return "\n".join(text_chunks).strip()

def build_dataset_from_raw_pdfs(
    raw_dir: str = os.path.join("data", "raw"),
    output_csv: Optional[str] = os.path.join("data", "processed", "extracted_resumes.csv"),
    benchmark_csv: Optional[str] = "Resume - Resume.csv"
) -> pd.DataFrame:
    """
    Iterates through category folders, extracts text from every PDF,
    and compiles a structured DataFrame / CSV.
    """
    records = []
    logger.info(f"Scanning directory: {raw_dir}")
    
    # Identify category subdirectories
    categories = sorted([
        d for d in os.listdir(raw_dir)
        if os.path.isdir(os.path.join(raw_dir, d))
    ])
    logger.info(f"Discovered {len(categories)} category directories.")

    all_pdf_paths = []
    for cat in categories:
        cat_path = os.path.join(raw_dir, cat)
        for fname in os.listdir(cat_path):
            if fname.lower().endswith(".pdf"):
                all_pdf_paths.append((cat, fname, os.path.join(cat_path, fname)))

    logger.info(f"Found {len(all_pdf_paths)} total PDF files to process.")
    
    for cat, fname, full_path in tqdm(all_pdf_paths, desc="Extracting PDFs"):
        resume_id = os.path.splitext(fname)[0]
        try:
            raw_text = extract_text_from_pdf(full_path)
            is_empty = (len(raw_text.strip()) == 0)
        except Exception as err:
            logger.error(f"Failed to read {full_path}: {err}")
            raw_text = ""
            is_empty = True

        records.append({
            "id": resume_id,
            "category": cat,
            "raw_text": raw_text,
            "char_count": len(raw_text),
            "word_count": len(raw_text.split()),
            "is_empty": is_empty,
            "file_path": full_path
        })

    df = pd.DataFrame(records)
    logger.info(f"Extraction finished. Total extracted rows: {len(df)}")
    logger.info(f"Empty resume count: {df['is_empty'].sum()} ({df['is_empty'].mean()*100:.2f}%)")

    # If benchmark CSV exists, check and backfill any corrupt/unreadable PDFs
    if benchmark_csv and os.path.exists(benchmark_csv):
        try:
            bench_df = pd.read_csv(benchmark_csv)
            logger.info(f"Loaded benchmark CSV ({benchmark_csv}) with {len(bench_df)} rows.")
            # Map ID to text in benchmark if needed
            bench_lookup = dict(zip(bench_df['ID'].astype(str), bench_df['Resume_str']))
            
            backfilled = 0
            for idx, row in df.iterrows():
                if row['is_empty'] and str(row['id']) in bench_lookup:
                    rep_text = str(bench_lookup[str(row['id'])])
                    if rep_text and rep_text != "nan":
                        df.at[idx, 'raw_text'] = rep_text
                        df.at[idx, 'char_count'] = len(rep_text)
                        df.at[idx, 'word_count'] = len(rep_text.split())
                        df.at[idx, 'is_empty'] = False
                        backfilled += 1
            if backfilled > 0:
                logger.info(f"Backfilled {backfilled} empty resumes from benchmark CSV.")
        except Exception as e:
            logger.warning(f"Could not align with benchmark CSV: {e}")

    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False, encoding='utf-8')
        logger.info(f"Saved extracted dataset to {output_csv}")

    return df

if __name__ == "__main__":
    df = build_dataset_from_raw_pdfs()
    print("Dataset extraction summary:")
    print(df.info())
    print(df["category"].value_counts())
