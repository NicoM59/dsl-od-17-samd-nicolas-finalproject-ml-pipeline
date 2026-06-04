# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import os
from dotenv import load_dotenv

# On charge les variables d'environnement (.env en local, secrets en prod)
load_dotenv()

app = FastAPI(
    title="Mental Health Disorder Detection API",
    description="API de prédiction clinique basée sur un modèle LinearSVC managé par MLflow",
    version="1.0.0"
)

# Configuration de la cible MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://13.39.248.239:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Variable globale pour stocker le modèle en mémoire vive
model = None

# Format attendu pour la requête de prédiction (input utilisateur)
class PredictionInput(BaseModel):
    text: str

@app.on_event("startup")
def load_model():
    """
    Cette fonction s'exécute AUTOMATIQUEMENT au démarrage de l'API.
    Elle va chercher la dernière version du modèle directement dans le registre MLflow.
    """
    global model
    try:
        print(f"📡 Connexion à MLflow ({MLFLOW_TRACKING_URI}) pour récupérer le modèle...")
        # On demande la version de Production (ou la dernière version du registre)
        model_uri = "models:/MentalHealth_LinearSVC/latest"
        model = mlflow.pyfunc.load_model(model_uri)
        print("✅ Modèle chargé en mémoire avec succès ! API prête à servir.")
    except Exception as e:
        print(f"❌ Erreur critique lors du chargement du modèle : {e}")
        # Option de secours locale si MLflow est inaccessible pendant tes tests
        model = None

@app.get("/")
def home():
    return {
        "status": "online", 
        "model_loaded": model is not None,
        "message": "Bienvenue sur l'API de détection des troubles de la santé mentale."
    }

@app.post("/predict")
def predict(payload: PredictionInput):
    if model == None:
        raise HTTPException(status_code=503, detail="Le modèle prédictif n'est pas disponible.")
    
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Le texte soumis est vide.")
        
    try:
        # 🌟 LA SÉCURITÉ : On force le casting en string propre pour éliminer tout résidu de type 'float'
        clean_input = str(payload.text)
        
        # Inférence (on passe une liste contenant notre string bien typée)
        prediction = model.predict([clean_input])
        
        predicted_class = str(prediction[0])
        
        return {
            "input_text": payload.text,
            "predicted_disorder": predicted_class,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'inférence : {str(e)}")
    