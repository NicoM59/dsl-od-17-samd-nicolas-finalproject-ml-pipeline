# ========================================================================================
# 🚦 EXPLICABILITE via récupération des poids de chaque avec l'attribut coef de Linear_SVC
# ========================================================================================

import numpy as np

def get_feature_importance(model, vectorizer, top_n=5):
    # Sécurité : Vérifier si le modèle a été entraîné (présence de coef_)
    if not hasattr(model, "coef_"):
        return {"error": "Le modèle n'a pas de coefficients (probablement pas entraîné)."}
    
    # Sécurité : Vérifier si le vectorizer a un vocabulaire
    if not hasattr(vectorizer, "get_feature_names_out"):
        return {"error": "Le vectorizer est invalide."}

    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_.flatten()
    
    # Si le vocabulaire et les coefficients ne correspondent pas en taille
    if len(feature_names) != len(coefficients):
        return {"error": "Mismatch entre vocabulaire et poids du modèle."}

    top_indices = np.argsort(coefficients)[-top_n:]
    bottom_indices = np.argsort(coefficients)[:top_n]
    
    return {
        "positifs": [(feature_names[i], float(coefficients[i])) for i in reversed(top_indices)],
        "negatifs": [(feature_names[i], float(coefficients[i])) for i in bottom_indices]
    }