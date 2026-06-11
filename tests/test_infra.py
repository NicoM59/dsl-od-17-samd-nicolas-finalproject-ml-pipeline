import os
import requests
import mlflow

def test_mlflow_and_neon_are_up():
    """Vérifie que le serveur MLflow sur EC2 répond et que son backend store Neon est accessible."""
    
    
    print("""
=========================================================
 ___  ____ ____    ___  ____    ___  _  _ ____    ____ _  _ ___ 
 |__] |__| [__     |  \ |___    |__] |  | | __    [__  |  | |__]
 |    |  | ___]    |__/ |___    |__] |__| |__]    ___]  \/  |   
=========================================================
""")
    
    
    print("\n" + "=" * 60)
    print("🚀 RUNNER GITHUB : DÉMARRAGE DU TEST D'INTEGRATION INFRASTRUCTURE")
    print("=" * 60)
    
    # 1. Récupération et vérification de la variable d'environnement
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    
    print("🔍 Étape 1 : Vérification de la variable d'environnement...")
    if tracking_uri is None:
        print("   ❌ ÉCHEC : La variable MLFLOW_TRACKING_URI n'est pas configurée dans l'environnement.")
    assert tracking_uri is not None, "La variable MLFLOW_TRACKING_URI est manquante."
    print(f"   🟢 SUCCÈS : URL cible détectée : {tracking_uri}\n")
    
    # 2. Test de niveau 1 : Connexion HTTP de base à l'EC2
    print("📡 Étape 2 : Tentative de 'Ping' HTTP sur l'instance EC2 (Port 5000)...")
    try:
        response = requests.get(tracking_uri, timeout=5)
        print(f"   🟢 SUCCÈS : L'instance EC2 a répondu (Code HTTP: {response.status_code}).")
        assert response.status_code == 200, f"Code HTTP inhabituel renvoyé par l'EC2 : {response.status_code}"
    except requests.exceptions.RequestException as e:
        print("   ❌ ÉCHEC : Impossible de joindre physiquement l'EC2.")
        print(f"   ➔ Erreur technique : {e}")
        assert False, f"Le serveur MLflow sur l'EC2 est injoignable : {e}"
        
    print("") # Ligne vide pour aérer les logs Pytest
    
    # 3. Test de niveau 2 : Requête de l'application MLflow vers Neon PostgreSQL
    print("🗄️ Étape 3 : Test de liaison MLflow Server ➔ Neon PostgreSQL (Backend Store)...")
    try:
        # On configure l'URI de tracking pour le client MLflow du Runner
        mlflow.set_tracking_uri(tracking_uri)
        print("   ➔ Envoi de la commande 'search_experiments' à l'EC2...")
        
        # Cette commande force l'EC2 à faire un SELECT dans Neon
        experiments = mlflow.search_experiments()
        
        print("   🟢 SUCCÈS : Le serveur MLflow a réussi à lire les données dans Neon DB !")
        print(f"   ➔ Nombre d'expériences actives trouvées en base : {len(experiments)}")
        for exp in experiments:
            print(f"      - Expérience détectée : '{exp.name}' (ID: {exp.experiment_id})")
            
        assert len(experiments) >= 0
        
    except Exception as e:
        print("   ❌ ÉCHEC : MLflow (EC2) est en ligne, mais il ne peut pas dialoguer avec Neon DB.")
        print(f"   ➔ Erreur renvoyée à travers l'API : {e}")
        assert False, f"Rupture de liaison entre MLflow et le Backend Store Neon : {e}"

    print("\n" + "=" * 60)
    print("🎉 TOUS LES FEUX SONT VERTS ! L'INFRASTRUCTURE COMMUNIQUE DE BOUT EN BOUT.")
    print("=" * 60)