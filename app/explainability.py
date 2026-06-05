# ========================================================================================
# 🚦 EXPLICABILITE via récupération des poids de chaque avec l'attribut coef de Linear_SVC
# ========================================================================================

import numpy as np

def get_feature_importance(model, vectorizer, top_n=5):
    # Récupérer les noms des mots (features)
    feature_names = vectorizer.get_feature_names_out()
    # Récupérer les coefficients appris par le LinearSVC
    # (Note: si tu as plusieurs classes, coef_ est une matrice)
    coefficients = model.coef_.flatten()
    
    # Trier les mots par importance
    top_indices = np.argsort(coefficients)[-top_n:]
    bottom_indices = np.argsort(coefficients)[:top_n]
    
    # Créer un dictionnaire simple
    important_features = {
        "positifs": [(feature_names[i], coefficients[i]) for i in reversed(top_indices)],
        "negatifs": [(feature_names[i], coefficients[i]) for i in bottom_indices]
    }
    return important_features