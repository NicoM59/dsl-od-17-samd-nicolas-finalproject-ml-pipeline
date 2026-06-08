# "Bipolar": "1456478153"


import requests
import os
import json
from functools import lru_cache
from dotenv import load_dotenv

# Charge les variables d'environnement (assure-toi d'avoir un fichier .env)
load_dotenv()

# Mapping des IDs ICD-11
ICD_IDS = {
    # "ADHD": "1566416972",
    # "Anxiety": "1064505370",
    # "Autism": "1731674483",
    "Bipolar": "1456478153"
    # "BPD": "1198305607",
    # "Depression": "1537233306",
    # "schizophrenia": "1456478153"
}

@lru_cache(maxsize=32)
def get_icd_token():
    """Récupère le token d'accès avec mise en cache."""
    token_url = 'https://icdaccessmanagement.who.int/connect/token'
    payload = {
        'client_id': os.getenv("client_id"),
        'client_secret': os.getenv("client_secret"),
        'scope': 'icdapi_access',
        'grant_type': 'client_credentials'
    }
    
    response = requests.post(token_url, data=payload, timeout=5)
    if response.status_code != 200:
        raise Exception(f"Erreur authentification OMS : {response.text}")
    return response.json()['access_token']
print("✅ Token reçu avec succès.")

def find_id_by_name(token, query):
    search_url = "https://id.who.int/icd/release/11/2024-01/mms/search"
    headers = {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/json',
        'Accept-Language': 'en',
        'API-Version': 'v2'
    }
    params = {'q': query}
    r = requests.get(search_url, headers=headers, params=params)
    data = r.json()
    
    # Affiche le résultat pour que tu puisses copier le VRAI ID
    # Le champ 'id' est celui que tu dois mettre dans ton dico
    for item in data['destinationEntities']:
        print(f"Trouvé : {item['title']} -> ID: {item['id']}")
        
        # ... garde tout ton code en haut tel quel ...

if __name__ == "__main__":
    # 1. Récupère le token une fois
    token = get_icd_token()
    print("✅ Token prêt pour la recherche.")
    
    # 2. Liste de tes pathologies à chercher
    pathologies = ["Bipolar", "schizophrenia", "Depression", "ADHD", "Anxiety", "Autism", "BPD"]
    
    # 3. Boucle de recherche
    for p in pathologies:
        print(f"\n--- Recherche pour : {p} ---")
        find_id_by_name(token, p)
        
        
# def get_medical_definition(disorder):
#     """Récupère la définition officielle depuis l'API ICD-11."""
#     print(f"DEBUG: Je cherche l'ID pour le disorder : '{disorder}'")
#     print(f"DEBUG: Les clés disponibles sont : {list(ICD_IDS.keys())}")
#     entity_id = ICD_IDS.get(disorder)
#     if not entity_id:
#         return "Aucune référence OMS trouvée pour cette catégorie."

#     try:
#         token = get_icd_token()
#         uri = f'https://id.who.int/icd/entity/{entity_id}'
#         headers = {
#             'Authorization': f'Bearer {token}',
#             'Accept': 'application/json',
#             'Accept-Language': 'fr',
#             'API-Version': 'v2'
# #         }
        
#         response = requests.get(uri, headers=headers, timeout=5)
        
#         if response.status_code == 200:
#             data = response.json()
#             print(json.dumps(data, indent=2))
#             title = data.get('title', {}).get('@value', disorder)
#             # Priorité à 'definition', puis 'description'
#             text = data.get('definition', {}).get('@value') or \
#                    data.get('description', {}).get('@value') or \
#                    "Critères cliniques disponibles sur le portail officiel de l'OMS."
            
#             return f"**{title}**\n\n{text}"
#         else:
#             return "Connexion à la base de données OMS temporairement indisponible."
            
#     except Exception as e:
#         print(f"Erreur API OMS : {e}")
#         return "Service de documentation médicale indisponible."

# # --- TEST RAPIDE (pour vérifier que tout fonctionne) ---
# if __name__ == "__main__":
#     print(get_medical_definition("schizophrenia"))