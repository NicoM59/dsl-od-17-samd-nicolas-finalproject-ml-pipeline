# 1. Image de base Python allégée (Idéale pour la prod)
FROM python:3.11-slim

# 2. Définition du répertoire de travail dans le conteneur
WORKDIR /workspace

# 3. Optimisation du cache Docker : on copie d'abord SEULEMENT les requirements
COPY requirements.txt .

# 4. Installation des paquets sans garder le cache (pour réduire le poids de l'image)
RUN pip install --no-cache-dir -r requirements.txt

# 5. On copie le code source de l'API (AUCUN modèle lourd copié !)
COPY ./app ./app

# 6. On indique le port sur lequel l'API va écouter
EXPOSE 8000

# 7. La commande de démarrage (sans le mode "--reload" réservé au dev)
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
