import os
import sys
import boto3
from io import BytesIO
import joblib
import hashlib
import mlflow
from mlflow.tracking import MlflowClient
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# Charge les variables d'environnement
load_dotenv()

# ==============================================================================
# 🌟 LE HACK DE GÉNIE POUR JOBLIB (Infaillible)
# ==============================================================================
def tfidf_anti_float_preprocessor(x):
    return str(x) if not isinstance(x, str) else x

def force_string_input(X):
    return [str(text) for text in X]

# On force l'injection des deux fonctions dans le cerveau principal de Python
sys.modules['__main__'].tfidf_anti_float_preprocessor = tfidf_anti_float_preprocessor
sys.modules['__main__'].force_string_input = force_string_input
# ==============================================================================

# ==============================================================================
# 🛠️ CONFIGURATION CLOUD & TEST
# ==============================================================================
TEST_MODE = os.getenv("TEST_MODE", "False").lower() in ("true", "1", "t")

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "eu-west-3")

app = FastAPI(title="Mental Health API - S3 Dynamic Router", version="1.0.0")
model = None

# ==============================================================================
# 🧪 MOCK MODEL POUR PYTEST
# ==============================================================================
class MockModel:
    def predict(self, texts):
        classes = ["ADHD", "Anxiety", "Autism", "BPD", "Bipolar", "Depression", "schizophrenia"]
        text_hash = int(hashlib.md5(str(texts[0]).strip().encode('utf-8')).hexdigest(), 16)
        return [classes[text_hash % len(classes)]]

# ==============================================================================
# ☁️ LOGIQUE ROUTEUR MLFLOW -> S3
# ==============================================================================
def get_latest_s3_path():
    """Utilise MLflow comme un annuaire pour trouver où se cache le 'latest' sur S3"""
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://13.39.248.239:5000")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    
    # 1. On cherche toutes les versions avec la nouvelle méthode
    versions = client.search_model_versions("name='MentalHealth_LinearSVC'")
    if not versions:
        raise ValueError("Aucune version trouvée sur MLflow.")
        
    # 2. On isole la toute dernière version (la plus grande)
    # 🎯 C'est ici qu'était l'erreur : on utilise bien la variable "versions" maintenant !
    latest_version = max(versions, key=lambda v: int(v.version))
    s3_uri = latest_version.source  # Ex: s3://bucket/5/hash/artifacts/model
    
    # 3. On extrait le bucket et le chemin pour boto3
    path_parts = s3_uri.replace("s3://", "").split("/")
    bucket = path_parts[0]
    key_prefix = "/".join(path_parts[1:])
    s3_key = f"{key_prefix}/model.pkl"  # On cible ton fichier exact
    
    return bucket, s3_key

def download_from_s3(bucket: str, key: str):
    """Télécharge le fichier S3 en mémoire vive"""
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )
    response = s3.get_object(Bucket=bucket, Key=key)
    return BytesIO(response["Body"].read())

# ==============================================================================
# 🚀 DÉMARRAGE DE L'API
# ==============================================================================
@app.on_event("startup")
def load_model():
    global model
    
    # Mode Test (Zéro réseau)
    if TEST_MODE:
        print("🧪 [TEST MODE] Activation du MockModel.")
        model = MockModel()
        return

    # Mode Production (Docker / S3 Automatique)
    try:
        print("📡 Interrogation du GPS MLflow pour localiser 'latest'...")
        bucket, s3_key = get_latest_s3_path()
        print(f"📍 'latest' trouvé sur S3 : s3://{bucket}/{s3_key}")
        
        print("📥 Téléchargement direct avec Boto3...")
        model_file = download_from_s3(bucket, s3_key)
        
        # Joblib respecte notre hack, lui !
        model = joblib.load(model_file)
        print("✅ Modèle 'latest' téléchargé et chargé avec succès !")
    except Exception as e:
        print(f"❌ Échec du chargement : {e}")
        model = None

# ==============================================================================
# 🚦 ENDPOINTS
# ==============================================================================
class PredictionInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {
        "status": "online",
        "model_loaded": model is not None,
        "mode": "test_mock" if TEST_MODE else "cloud_latest_s3"
    }

@app.post("/predict")
def predict(payload: PredictionInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle indisponible.")
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Texte vide.")

    try:
        prediction = model.predict([payload.text])
        return {
            "input_text": payload.text,
            "predicted_disorder": str(prediction[0]),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne : {str(e)}")