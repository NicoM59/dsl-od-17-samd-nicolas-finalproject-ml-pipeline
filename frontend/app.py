import streamlit as st
import requests
import json

# ==========================================
# 🎨 CONFIGURATION DE LA PAGE
# ==========================================
st.set_page_config(
    page_title="IA Santé Mentale | Démo",
    page_icon="🧠",
    layout="centered"
)

# ==========================================
# 🔗 CONNEXION À L'API (BACKEND)
# ==========================================
# ⚠️ À REMPLACER : Mets l'URL que Render va te donner (sans le /docs à la fin)
# Exemple : "https://mental-health-api-nicolas.onrender.com"
API_URL = "http://127.0.0.1:8000"  

# ==========================================
# 🖥️ INTERFACE UTILISATEUR
# ==========================================
st.title("🧠 Analyseur de Santé Mentale par IA")
st.markdown("""
Bienvenue sur cet outil de détection développé dans le cadre du projet MLOps. 
Ce modèle de Machine Learning analyse la sémantique de votre texte pour identifier des signaux associés à certains troubles psychologiques.
""")

st.info("💡 **Note :** Ceci est un outil d'assistance au diagnostic basé sur du traitement du langage naturel (NLP). Il ne remplace en aucun cas l'avis d'un professionnel de santé.")

# Zone de saisie
user_text = st.text_area(
    "✍️ Entrez le texte à analyser (ex: journal intime, post de forum...) :",
    height=200,
    placeholder="Je me sens vraiment épuisé en ce moment, je n'arrive plus à me concentrer..."
)

# Bouton d'action
if st.button("🚀 Lancer l'analyse IA", use_container_width=True):
    if not user_text.strip():
        st.warning("⚠️ Veuillez entrer du texte avant de lancer l'analyse.")
    else:
        with st.spinner("L'API interroge le modèle S3 en temps réel..."):
            try:
                # Appel HTTP à ton API FastAPI
                response = requests.post(f"{API_URL}/predict", json={"text": user_text})
                
                if response.status_code == 200:
                    data = response.json()
                    disorder = data.get("predicted_disorder", "Inconnu")
                    
                    st.success("✅ Analyse terminée avec succès !")
                    
                    # Affichage du résultat en grand
                    st.metric(label="Diagnostic principal détecté", value=disorder)
                    
                    # Affichage des données brutes pour le côté "Tech" de la soutenance
                    with st.expander("🛠️ Voir la réponse brute de l'API (JSON)"):
                        st.json(data)
                        
                else:
                    st.error(f"❌ Erreur de l'API (Code {response.status_code})")
                    st.write(response.json())
                    
            except requests.exceptions.ConnectionError:
                st.error("🔌 Impossible de se connecter à l'API. Vérifiez que le serveur Render est bien allumé et que l'URL est correcte.")
            except Exception as e:
                st.error(f"⚠️ Une erreur inattendue est survenue : {e}")