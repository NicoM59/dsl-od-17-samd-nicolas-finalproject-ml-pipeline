# app/main.py
import os
import __main__
import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 1. LE HACK DE GENIE : On greffe la fonction dans le namespace pour l'unpickling MLflow
def tfidf_anti_float_preprocessor(x):
    return str(x) if not isinstance(x, str) else x

setattr(__main__, "tfidf_anti_float_preprocessor", tfidf_anti_float_preprocessor)

# 2. Configuration FastAPI
app = FastAPI(
    title="Mental Health API - Mode Dynamic Pull",
    version="1.0.0"
)
model = None

# 3. Configuration MLflow Distant
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://13.39.248.239:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# 🎯 REMPLACE PAR LE BON NUMÉRO DE VERSION !
MODEL_URI = "models:/MentalHealth_LinearSVC/latest"

@app.on_event("startup")
def load_model():
    global model
    try:
        print(f"📡 Connexion à {MLFLOW_TRACKING_URI}...")
        print(f"📥 Téléchargement dynamique du modèle : {MODEL_URI}...")
        
        # Le SDK MLflow va télécharger l'artefact S3 et le dé-sérialiser
        # Grâce à notre setattr, il trouvera la fonction sans crasher !
        model = mlflow.pyfunc.load_model(MODEL_URI)
        print("✅ Modèle distant téléchargé et chargé avec succès !")
    except Exception as e:
        print(f"❌ Échec du chargement distant : {e}")
        model = None

class PredictionInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"status": "online", "model_loaded": model is not None, "mode": "dynamic_pull"}

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