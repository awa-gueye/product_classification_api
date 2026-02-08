# api/Dockerfile
FROM python:3.10-slim

# Définir le répertoire de travail
WORKDIR /api

# Copier requirements d'abord (optimisation cache Docker)
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code de l'application
COPY main.py .

# Exposer le port
EXPOSE 8000

# Variable d'environnement pour Railway
ENV PORT=8000
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=3

# Commande de démarrage
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]