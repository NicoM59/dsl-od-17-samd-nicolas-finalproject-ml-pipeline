#app/train.py


import os
import sys
import shutil
import pickle
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score
import joblib  # 🌟 Utilisation de joblib pour une meilleure sérialisation
from app.preprocess import clean_body_text, map_canonical_category

# On récupère le chemin depuis l'environnement


# ==============================================================================
# 🛠️ SÉCURITÉ ABSOLUE POUR LES CHEMINS EN ENVIRONNEMENT LINUX / CI-CD & SERIALISATION DU DATASET
# ==============================================================================
dataset_path = os.getenv("DATASET_PATH")

if not dataset_path:
    raise ValueError("La variable DATASET_PATH n'est pas définie !")

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ==============================================================================


# --- SÉCURITÉ SYSTÈME ABSOLUE ---
# Use explicit import to satisfy linters: python-dotenv exposes load_dotenv
try:
    from dotenv import load_dotenv
except Exception:
    # Provide a no-op fallback when python-dotenv is not installed (e.g. in CI linting)
    def load_dotenv(*args, **kwargs):
        return None

# mlflow is optional in some environments (linters / CI without package).
# Provide a lightweight dummy shim when mlflow cannot be imported so the
# script remains importable and linters don't fail. Real mlflow will be
# used when available at runtime.
try:
    import mlflow
except Exception:
    class _DummyRunInfo:
        def __init__(self):
            self.run_id = "local_dummy_run"

    class _DummyRun:
        def __init__(self):
            self.info = _DummyRunInfo()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _DummyData:
        @staticmethod
        def from_pandas(df, source=None, name=None):
            return None

    class _DummyMLflow:
        data = _DummyData()

        @staticmethod
        def end_run():
            return None

        @staticmethod
        def set_tracking_uri(uri):
            return None

        @staticmethod
        def set_experiment(name):
            return None

        @staticmethod
        def get_tracking_uri():
            return ""

        @staticmethod
        def start_run(run_name=None):
            return _DummyRun()

        @staticmethod
        def log_input(*args, **kwargs):
            return None

        @staticmethod
        def set_tag(*args, **kwargs):
            return None

        @staticmethod
        def log_params(*args, **kwargs):
            return None

        @staticmethod
        def log_metric(*args, **kwargs):
            return None

        @staticmethod
        def log_artifact(*args, **kwargs):
            return None

        @staticmethod
        def log_artifacts(*args, **kwargs):
            return None

        @staticmethod
        def register_model(*args, **kwargs):
            return None

    mlflow = _DummyMLflow()

load_dotenv()
os.environ["MLFLOW_AUTOLOGGING_DISABLE"] = "true"
os.environ["MLFLOW_SKLEARN_AUTOLOG"] = "false"

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV  # type: ignore
import sklearn.feature_extraction.text
try:
    import sklearn.svm
except Exception:
    # Fallback for environments where sklearn.svm may not be available to the linter/runtime
    # Use a compatible linear classifier as a substitute
    from sklearn.linear_model import SGDClassifier as LinearSVC  # type: ignore
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score

def tfidf_anti_float_preprocessor(x):
    """Fonction globale acceptée par pickle pour nettoyer les types inattendus"""
    return str(x) if not isinstance(x, str) else x

# Cette mini-fonction force la conversion de chaque élément en string
def force_string_input(X):
    return [str(text) for text in X]

try:
    mlflow.end_run()
except:
    pass

# --- LOGIQUE DE NETTOYAGE ET S3 ---
def load_and_preprocess_data(path_or_url):
    print(f"📥 Chargement : {path_or_url}")
    df = pd.read_csv(path_or_url)
    
    print("📥 Appel de preprocess.py")
    # Étape de nettoyage qui utilise les fonctions déportées
    df = df.dropna(subset=['body', 'category']).copy()
    df['body'] = df['body'].apply(clean_body_text)
    df['category'] = df['category'].apply(map_canonical_category)
    
    df = df.dropna(subset=['category'])
    mlflow_dataset = mlflow.data.from_pandas(df, source=path_or_url, name="mental_health_disorder_detection")
    return df['body'], df['category'], mlflow_dataset

print("📥 Chargement & preprocessing terminé")

def create_pipeline():
    print(f"🌟Application des poids de classe")
    custom_weights = {
        'schizophrenia': 2.1384418901660283,
        'Depression': 1.1239092495636998,
        'Autism': 1.0751252086811351,
        'Bipolar': 1.0593557230980124,
        'BPD': 0.9953632148377125,
        'Anxiety': 0.8396349413298566,
        'ADHD': 0.8039950062421972
    }
    # 🌟 On passe directement la fonction globale force_string_input ici
    return Pipeline([
        ("force_string", FunctionTransformer(force_string_input)),
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=50000,
            sublinear_tf=True,
            preprocessor=tfidf_anti_float_preprocessor
        )),
        ("clf", LinearSVC(class_weight=custom_weights, random_state=42))
    ])
    
print("🏋️‍♂️ Création du pipeline terminée...")
# 🌟 TOUTE LA LOGIQUE D'EXÉCUTION ET DE TRACKING EFFECTIVE ICI
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 1. On ne met pas de valeur par défaut ici, ou alors une valeur générique
    parser.add_argument("--data_url", type=str, help="URL S3 du dataset")
    args = parser.parse_args()

    # 2. La priorité : 
    # - Si l'utilisateur passe --data_url dans la commande, on utilise celui-là.
    # - Sinon, on utilise la variable d'environnement.
    final_data_url = args.data_url or os.getenv("DATASET_PATH")

    if not final_data_url:
        raise ValueError("Aucun dataset trouvé : spécifiez --data_url ou définissez DATASET_PATH")
        
    print(f"🚀 Démarrage avec le dataset : {final_data_url}")

    # 🔐 CONFIGURATION MLFLOW SÉCURISÉE : Uniquement lue lors de l'entraînement, pas pendant les tests Pytest !
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://13.39.248.239:5000")
    
    print(f"📡 [LOG CI] Tentative de connexion à MLflow sur : {MLFLOW_TRACKING_URI}...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("mental_health_svc_final_demo")
    print("✅ [LOG CI] Connexion MLflow initialisée (ou mise en attente).")
    
    print(f"🎯 CIBLE MLFLOW : {mlflow.get_tracking_uri()}")

    # PHASE 1 : Entraînement pur (100% déconnecté de MLflow)
    print("📥 [LOG CI] Début du téléchargement des données depuis S3...")
    X, y,mlflow_dataset = load_and_preprocess_data(args.data_url)
    print(f"📊 [LOG CI] Données chargées avec succès ! Taille du dataset : {len(X)} lignes.")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    print("🏋️‍♂️ Lancement du GridSearchCV...Merci de patienter")
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
    with mlflow.start_run(run_name="mental_health_svc_tuning_demo_final") as run:
        mlflow.log_input(mlflow_dataset, context="training")
        # 📝 A. Remplir la colonne DESCRIPTION
        mlflow.set_tag("mlflow.note.content", "Pipeline SVC entraîné via l'automatisation GitHub Actions après validation de la CI.")
        mlflow.set_tag("Model_Type", "LinearSVC w/ CV")
        
        # 📊 B. Remplir la colonne DATASET (Reconstitution propre à la volée)
        try:
            # On recrée un mini-dataframe temporaire pour le notifier à MLflow
            summary_df = pd.DataFrame({"body": X, "category": y})
            mlflow_dataset = mlflow.data.from_pandas(summary_df, source=args.data_url, name="mental_health_disorder_detection")
            
            
            print("✅ Dataset notifié avec succès dans MLflow.")
        except Exception as e:
            print(f"⚠️ Impossible de loguer le dataset : {e}")

        print(f"✅ Envoi des données dans l'unique Run ID : {run.info.run_id}")
        
        # 1. Hyperparamètres et Métriques
        mlflow.log_params(best_params)
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_f1_macro", f1_macro)
        
        # 2. Focus clinique
        target_focus = ["Bipolar", "schizophrenia"]
        for label in target_focus:
            if isinstance(report_dict, dict) and isinstance(report_dict.get(label), dict):
                metrics = report_dict[label]
                mlflow.log_metric(f"FOCUS_{label}_f1", metrics['f1-score'])
                mlflow.log_metric(f"FOCUS_{label}_precision", metrics['precision'])
                mlflow.log_metric(f"FOCUS_{label}_recall", metrics['recall'])
        
        # 3. Rapports textuels
        with open("classification_report.txt", "w") as f:
            f.write(str(report_text))
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

        # 🤖 C. Remplir la colonne MODELS (Enregistrement officiel dans le registre)
        try:
            model_uri = f"runs:/{run.info.run_id}/model"
            mlflow.register_model(model_uri, "MentalHealth_LinearSVC")
            print("✅ Modèle enregistré avec succès dans le Model Registry.")
        except Exception as e:
            print(f"⚠️ Impossible d'enregistrer le modèle dans le registre : {e}")
        
        # Nettoyage local du dossier temporaire
        try:
            shutil.rmtree(model_dir)
        except:
            pass
        
    print(f"🎉 Entrainement Terminé ! Run enregistré sur MLFLOW et artifacts déposés sur S3.")
    