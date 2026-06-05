import os
import pytest
from fastapi.testclient import TestClient

# 🚨 SÉCURITÉ ABSOLUE : On force le mode test AVANT d'importer l'API.
os.environ["TEST_MODE"] = "True"

from app.main import app

# 🌟 LA SOLUTION : Une fixture Pytest !
# Le bloc 'with' force FastAPI à exécuter le @app.on_event("startup")
@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client

def test_home_endpoint(client):
    """Vérifie que la route racine (/) fonctionne et que le mode Mock est actif."""
    response = client.get("/")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "online"
    assert data["model_loaded"] is True
    assert data["mode"] == "test_mock"  # Vérifie qu'on est bien sur le Mock !

def test_predict_valid_text(client):
    """Vérifie qu'une phrase valide renvoie bien une prédiction."""
    payload = {"text": "I feel quite stressed and tired lately."}
    response = client.post("/predict", json=payload)
    
    if response.status_code == 500:
        print("\n--- ERREUR DÉTECTÉE ---")
        print(response.json()) # Souvent, FastAPI renvoie le détail de l'erreur ici
        print("------------------------\n")

    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["input_text"] == payload["text"]
    assert "predicted_disorder" in data

def test_predict_empty_text(client):
    """Vérifie que l'API bloque intelligemment les textes vides."""
    payload = {"text": "   "}  # Juste des espaces
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "Texte vide" in data["detail"]