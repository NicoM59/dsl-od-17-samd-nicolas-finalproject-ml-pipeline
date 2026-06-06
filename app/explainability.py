import numpy as np

def get_feature_importance(model, vectorizer, top_n=10, target_class_name=None, class_names=None):
    feature_names = vectorizer.get_feature_names_out()
    
    # Gestion du focus par classe avec sécurité
    coefficients = None
    if target_class_name and class_names:
        try:
            # On cherche l'index de la pathologie visée
            idx = class_names.index(target_class_name)
            coefficients = model.coef_[idx]
        except ValueError:
            # Si le nom n'est pas dans la liste, on bascule en mode global
            coefficients = np.sum(np.abs(model.coef_), axis=0)
    
    # Si aucun focus n'était demandé ou erreur de matching
    if coefficients is None:
        coefficients = np.sum(np.abs(model.coef_), axis=0)

    # Calcul des indices (reste inchangé)
    top_indices = np.argsort(coefficients)[-top_n:]
    bottom_indices = np.argsort(coefficients)[:top_n]
    
    return {
        "positifs": [(feature_names[i], float(coefficients[i])) for i in reversed(top_indices)],
        "negatifs": [(feature_names[i], float(coefficients[i])) for i in bottom_indices]
    }