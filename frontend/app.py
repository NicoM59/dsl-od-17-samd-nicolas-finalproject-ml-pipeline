import streamlit as st
import requests
import json

# ==========================================
# 1. CONFIGURATION DE LA PAGE
# ==========================================
st.set_page_config(
    page_title="IA Santé Mentale | Démo",
    page_icon="🩺",
    layout="wide", # On passe en format large pour mieux utiliser l'espace
    initial_sidebar_state="expanded"
)

# 🔗 URL de l'API (À modifier avec ton URL Render une fois prête)
# API_URL = "http://127.0.0.1:8000"
API_URL = "https://mental-health-api-nicolas-samd.onrender.com"  

# ==========================================
# 2. BARRE LATÉRALE (SIDEBAR)
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2875/2875348.png", width=100) # Icône d'illustration
    st.title("À propos")
    st.markdown("""
    Ce tableau de bord est une démonstration d'intégration **MLOps**. 
    
    Il communique avec une API FastAPI conteneurisée qui héberge un modèle de Machine Learning entraîné à détecter des signaux sémantiques.
    """)
    st.divider()
    st.warning("⚠️ **Avertissement**\n\nCeci est un outil d'assistance expérimental. Il ne remplace en aucun cas un diagnostic médical professionnel.")

# ==========================================
# 3. INTERFACE PRINCIPALE
# ==========================================
st.title("🧠 Interface d'Analyse Sémantique")
st.markdown("Saisissez un texte pour analyser les schémas linguistiques et obtenir une prédiction du modèle de classification.")

# On utilise des colonnes pour structurer l'écran
col1, col2 = st.columns([2, 1])

with col1:
    user_text = st.text_area(
        "📝 Texte à analyser :",
        height=250,
        placeholder="Décrivez votre état d'esprit, vos pensées récentes ou collez un extrait de texte ici..."
    )
    
    analyze_button = st.button("Lancer l'analyse 🚀", use_container_width=True, type="primary")

with col2:
    st.markdown("### 📊 Résultats")
    
    if analyze_button:
        if not user_text.strip():
            st.error("Veuillez saisir un texte à analyser.")
        else:
            with st.spinner("Analyse en cours par le modèle..."):
                try:
                    response = requests.post(f"{API_URL}/predict", json={"text": user_text})
                    
                    if response.status_code == 200:
                        data = response.json()
                        disorder = data.get("predicted_disorder", "Inconnu")
                        
                        # 1. On tente de récupérer le pourcentage envoyé par l'API
                        # S'il n'est pas encore envoyé, on met 0 par défaut pour éviter un plantage
                        probability = data.get("probability", 0) 
                        
                        st.success("Analyse terminée")
                        st.metric(label="Catégorie détectée", value=disorder)
                        
                        # 2. Nouvelle section : Affichage visuel du pourcentage
                        if probability > 0:
                            st.markdown(f"**Niveau de confiance : {probability}%**")
                            # st.progress génère une barre de remplissage dynamique
                            st.progress(int(probability))
                            
                        with st.expander("Voir le payload JSON (Debug)"):
                            st.json(data)
                    else:
                        st.error(f"Erreur API : {response.status_code}")
                
                except requests.exceptions.ConnectionError:
                    st.error("🔌 Impossible de joindre l'API Backend.")
    else:
        st.info("En attente de soumission du texte.")