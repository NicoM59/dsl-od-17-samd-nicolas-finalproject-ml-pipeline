import os
import sys
import boto3
from io import BytesIO
import joblib
import hashlib
import mlflow
import numpy as np
import uuid
from datetime import datetime
from mlflow.tracking import MlflowClient
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from dotenv import load_dotenv

# Charge les variables d'environnement
load_dotenv()

# Si API_URL est défini dans le système, il l'utilise. 
# Sinon, il prend "http://localhost:8000" par défaut.
API_URL = os.getenv("API_URL", "http://localhost:8000")

print(f"Connexion à l'API sur : {API_URL}")

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
    
    # Eviter les erreurs 500
    def decision_function(self, texts):
        # Retourne une liste de scores factices (par exemple 0.5 pour chaque texte)
        # On renvoie une liste de la même taille que les entrées
        return [0.5] * len(texts)

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
# 🚦 ENDPOINTS & SAUVEGARDE DES REQUETES SUR S3
# ==============================================================================
def save_request_to_s3(payload: dict):
    """Envoie la requête utilisateur sur S3 pour le futur ETL."""
    try:
        s3 = boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
        # Nom du fichier : json_queries/AAAA/MM/JJ/uuid.json
        file_key = f"json_queries/{datetime.now().strftime('%Y/%m/%d')}/{uuid.uuid4()}.json"
        
        s3.put_object(
            # ⚠️ Remplace par ton vrai nom
            Bucket="dsl-od-17-samd-nicolas-finalproject", 
            Key=file_key,
            Body=json.dumps(payload),
            ContentType="application/json"
        )
    except Exception as e:
        print(f"❌ Erreur S3 : {e}")


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
def predict(payload: PredictionInput, background_tasks: BackgroundTasks):
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle indisponible.")
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Texte vide.")

    try:
        # 1. Prédiction
        prediction = model.predict([payload.text])
        
        # 2. Calcul du score de confiance (Sigmoïde)
        # decision_function donne la distance à la frontière de décision
        raw_scores = model.decision_function([payload.text])
        max_score = np.max(raw_scores)
        
        # Fonction sigmoïde : 1 / (1 + exp(-x))
        # Cela transforme la distance en probabilité (0 à 1)
        probability = 1 / (1 + np.exp(-max_score))
        
        # Conversion en pourcentage entier (0 à 100)
        conf_percentage = int(probability * 100)

        api_url = os.getenv("API_URL", "http://localhost:8000") 
           
        # 1. On prépare les données de log
        log_data = {
            "input_text": payload.text,
            "predicted_disorder": str(prediction[0]),
            "probability": conf_percentage,
            "timestamp": datetime.now().isoformat()
        }

    # 2. On déclenche la tâche de fond (elle se lancera en parallèle)
        background_tasks.add_task(save_request_to_s3, log_data)   
               
        return {
            "input_text": payload.text,
            "predicted_disorder": str(prediction[0]),
            "probability": conf_percentage, # On envoie le pourcentage
            "status": "success",
            "api_source": api_url
        }
        
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))