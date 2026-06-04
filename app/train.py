from app.preprocess import clean_body_text, map_canonical_category #on fait appel à preprocess.py dans le repo
import argparse
import os
import pickle
import shutil
from dotenv import load_dotenv

# --- SÉCURITÉ SYSTÈME ABSOLUE ---
load_dotenv()
os.environ["MLFLOW_AUTOLOGGING_DISABLE"] = "true"
os.environ["MLFLOW_SKLEARN_AUTOLOG"] = "false"

import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score

try:
    mlflow.end_run()
except:
    pass

# --- LOGIQUE DE NETTOYAGE ET S3 ---
def load_and_preprocess_data(path_or_url):
    print(f"📥 Chargement : {path_or_url}")
    df = pd.read_csv(path_or_url)
    
    # Étape de nettoyage qui utilise les fonctions déportées
    df = df.dropna(subset=['body', 'category']).copy()
    df['body'] = df['body'].apply(clean_body_text)
    df['category'] = df['category'].apply(map_canonical_category)
    
    df = df.dropna(subset=['category'])
    return df['body'], df['category']

def create_pipeline():
    custom_weights = {
        'schizophrenia': 2.1384418901660283,
        'Depression': 1.1239092495636998,
        'Autism': 1.0751252086811351,
        'Bipolar': 1.0593557230980124,
        'BPD': 0.9953632148377125,
        'Anxiety': 0.8396349413298566,
        'ADHD': 0.8039950062421972
    }
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=50000, sublinear_tf=True)),
        ("clf", LinearSVC(class_weight=custom_weights, random_state=42))
    ])

# 🌟 TOUTE LA LOGIQUE D'EXÉCUTION ET DE TRACKING EFFECTIVE ICI
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_url", type=str, default="s3://dsl-od-17-samd-nicolas-finalproject/Mental Health Disorder Detection Dataset.csv")
    args = parser.parse_args()

    # 🔐 CONFIGURATION MLFLOW SÉCURISÉE : Uniquement lue lors de l'entraînement, pas pendant les tests Pytest !
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://13.39.248.239:5000")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("mental_health_svc_final_demo")
    
    print(f"🎯 CIBLE MLFLOW : {mlflow.get_tracking_uri()}")

    # PHASE 1 : Entraînement pur (100% déconnecté de MLflow)
    X, y = load_and_preprocess_data(args.data_url)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    print("🏋️‍♂️ Lancement du GridSearchCV...")
    grid = GridSearchCV(create_pipeline(), {"clf__C": [0.5, 1.0]}, cv=3, scoring="f1_macro", n_jobs=-1, verbose=2)
    grid.fit(X_train, y_train)
    
    best_params = grid.best_params_
    print(f"🏆 Meilleurs hyperparamètres : {best_params}")
    
    final_pipeline = create_pipeline()
    final_pipeline.set_params(**best_params)
    print("🔄 Ré-entraînement du pipeline vainqueur...")
    final_pipeline.fit(X_train, y_train)
    
    y_pred = final_pipeline.predict(X_test)

    # Métriques
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_text = classification_report(y_test, y_pred)
    
    print("\n==================== REPORT DE CLASSIFICATION COMPLET ====================")
    print(report_text)
    print("=========================================================================\n")

    # PHASE 2 : Archivage final STRICTEMENT contrôlé
    with mlflow.start_run(run_name="mental_health_svc_tuning_3") as run:
        print(f"✅ Envoi des données dans l'unique Run ID : {run.info.run_id}")
        
        # 1. Hyperparamètres et Métriques
        mlflow.log_params(best_params)
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_f1_macro", f1_macro)
        
        # 2. Focus clinique
        target_focus = ["Bipolar", "schizophrenia"]
        for label in target_focus:
            if label in report_dict:
                metrics = report_dict[label]
                mlflow.log_metric(f"FOCUS_{label}_f1", metrics['f1-score'])
                mlflow.log_metric(f"FOCUS_{label}_precision", metrics['precision'])
                mlflow.log_metric(f"FOCUS_{label}_recall", metrics['recall'])
        
        # 3. Rapports textuels
        with open("classification_report.txt", "w") as f:
            f.write(report_text)
        mlflow.log_artifact("classification_report.txt")
        
        matrix = confusion_matrix(y_test, y_pred)
        np.savetxt("confusion_matrix.txt", matrix, fmt='%d')
        mlflow.log_artifact("confusion_matrix.txt")
        
        # 4. ENREGISTREMENT SÉCURISÉ SANS MLFLOW.SKLEARN
        print("📦 Préparation du package modèle...")
        model_dir = "temp_model_package"
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        os.makedirs(model_dir)
        
        # On sauvegarde le modèle en pickle classique
        with open(os.path.join(model_dir, "model.pkl"), "wb") as f:
            pickle.dump(final_pipeline, f)
            
        # On écrit le fichier d'environnement minimal requis par MLflow
        with open(os.path.join(model_dir, "MLmodel"), "w") as f:
            f.write(f"""artifact_path: model
flavors:
  python_function:
    env: null
    loader_module: mlflow.pyfunc.model_loader_utils
    python_version: 3.11
    data: model.pkl
run_id: {run.info.run_id}
""")

        print("📦 Envoi du modèle sur S3...")
        # On logue le dossier complet comme l'artefact "model"
        mlflow.log_artifacts(model_dir, artifact_path="model")
        
        # Nettoyage local du dossier temporaire
        try:
            shutil.rmtree(model_dir)
        except:
            pass
        
    print(f"🎉 Entrainement Terminé ! Run enregistré sur MLFLOW et artifacts déposés sur S3.")
    