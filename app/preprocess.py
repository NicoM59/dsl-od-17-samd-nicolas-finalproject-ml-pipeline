# app/preprocess.py
import pandas as pd

CANONICAL_LABELS = {
    "adhd": "ADHD", "add": "ADHD", "anxiety": "Anxiety",
    "autism": "Autism", "asd": "Autism", "autistic": "Autism",
    "bipolar": "Bipolar", "bpd": "BPD", "borderline": "BPD",
    "depression": "Depression", "depressed": "Depression",
    "schizophrenia": "schizophrenia", "schizo": "schizophrenia", "schizoaffective": "schizophrenia"
}

CANONICAL_CLASS_LIST = [
    "ADHD", "Anxiety", "Autism", "BPD", "Bipolar", "Depression", "schizophrenia"
]

def clean_body_text(text):
    """Nettoie le texte brut d'un post (Testable par Pytest sans S3)"""
    return str(text).strip()

def map_canonical_category(category_str):
    """Associe un label à sa catégorie canonique (Testable par Pytest sans S3)"""
    if pd.isna(category_str):
        return None
    lowered = str(category_str).strip().lower()
    return CANONICAL_LABELS.get(lowered, None)
