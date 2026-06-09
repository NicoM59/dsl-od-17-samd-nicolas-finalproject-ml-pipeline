# app/preprocess.py
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# 📋 Téléchargement sécurisé des briques NLTK pour le Runner GitHub et Render
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')  # 🌟 Ajout de la vérification de la nouvelle ressource
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)  # 🌟 Force le téléchargement si manquant
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

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
    """Nettoie le texte brut d'un post (Testable par Pytest sans S3) - RESTE INVARIANT"""
    return str(text).strip()

def map_canonical_category(category_str):
    """Associe un label à sa catégorie canonique (Testable par Pytest sans S3) - RESTE INVARIANT"""
    if pd.isna(category_str):
        return None
    lowered = str(category_str).strip().lower()
    return CANONICAL_LABELS.get(lowered, None)

def lemmatize_body_text(text):
    """
    🧪 NOUVELLE FONCTION FEATURE V2 :
    Nettoie le texte en profondeur et applique une lemmatisation complète pour le TF-IDF.
    """
    if not text or str(text).strip() == "":
        return ""
    
    # 1. Passage en minuscules et retrait des espaces aux extrémités
    text = str(text).lower().strip()
    
    # 2. Retrait des URL et des caractères spéciaux / chiffres via Regex
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 3. Tokenisation
    tokens = word_tokenize(text)
    
    # 4. Initialisation des stop words anglais et du lemmatiseur
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    # 5. Filtrage des mots vides + Lemmatisation
    cleaned_tokens = [
        lemmatizer.lemmatize(word) 
        for word in tokens 
        if word not in stop_words
    ]
    
    return " ".join(cleaned_tokens)