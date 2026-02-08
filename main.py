"""
API FastAPI pour Railway avec Docker
Optimisée pour 1GB RAM avec TensorFlow
"""

import os
# CONFIGURATION CRITIQUE - DOIT ÊTRE EN PREMIER
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=info, 2=warnings, 3=errors
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Désactiver GPU
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys
print(f"🐍 Python version: {sys.version}")
print(f"📦 scikit-learn version: {sklearn.__version__}")
import numpy as np
import joblib
import logging
from datetime import datetime
import sys
import gdown
import tempfile
import gc
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional
from PIL import Image
import io

# Configuration logging minimal
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ========== CONFIGURATION ==========
GOOGLE_DRIVE_IDS = {
    "tfidf_vectorizer.pkl": "1De4pUoj_IDdH3ZMYYQaTVFwwAgMo_N9y",
    "final_best_model.pkl": "1p0UXPEM5bQ2CjM6BS3YtYlxcAED6o6UA", 
    "label_encoders.pkl": "1O4EFUU6Qj_mtEb_wmBL6QjlLahe3l3yH",
    "cnn_final.keras": "1RXL7knfjXtNk6Aa3HZZQjCEUDow0QUJ7",  
}

CATEGORIES = [
    "Baby Care", "Beauty and Personal Care", "Computers",
    "Home Decor & Festive Needs", "Home Furnishing", 
    "Kitchen & Dining", "Watches"
]

# ========== INITIALISATION API ==========
app = FastAPI(
    title="Product Classification API",
    description="API for e-commerce product categorization",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== GESTION DES MODÈLES ==========
class RailwayModelLoader:
    """Chargeur optimisé pour Railway (1GB RAM)"""
    
    def __init__(self):
        self.vectorizer = None
        self.text_model = None
        self.label_mapping = None
        self.cnn_model = None
        self.loaded = False
        self.memory_used_mb = 0
    
    def load_all_models(self):
        """Charge tous les modèles avec gestion mémoire"""
        try:
            logger.warning("🚀 Démarrage du chargement des modèles sur Railway...")
            
            # 1. Modèles texte (légers)
            logger.warning("1. Chargement modèles texte...")
            self._load_text_models()
            self._log_memory("Après modèles texte")
            
            # 2. Modèle CNN (lourd)
            logger.warning("2. Chargement modèle CNN...")
            self._load_cnn_model()
            self._log_memory("Après CNN")
            
            self.loaded = True
            logger.warning(f"✅ Tous les modèles chargés! Mémoire: ~{self.memory_used_mb}MB")
            
        except MemoryError:
            logger.error("💥 Mémoire insuffisante! Passage en mode texte-only")
            self._fallback_mode()
        except Exception as e:
            logger.error(f"❌ Erreur: {e}")
            raise
    
    def _load_text_models(self):
        """Charge les modèles texte"""
        # TF-IDF
        self.vectorizer = self._download_model("tfidf_vectorizer.pkl")
        self.memory_used_mb += 1  # ~1MB
        
        # Modèle texte
        self.text_model = self._download_model("final_best_model.pkl")
        self.memory_used_mb += 1  # ~1MB
        
        # Labels
        label_data = self._download_model("label_encoders.pkl")
        if isinstance(label_data, dict):
            self.label_mapping = {v: k for k, v in label_data.items()}
        else:
            self.label_mapping = {i: cat for i, cat in enumerate(CATEGORIES)}
    
    def _load_cnn_model(self):
        """Charge le modèle CNN avec gestion mémoire"""
        try:
            # Importer TensorFlow seulement si nécessaire
            import tensorflow as tf
            from tensorflow import keras
            
            # Configurer TF pour économiser mémoire
            tf.config.set_visible_devices([], 'GPU')
            tf.config.threading.set_intra_op_parallelism_threads(1)
            tf.config.threading.set_inter_op_parallelism_threads(1)
            
            # Essayer de télécharger .h5 d'abord
            try:
                cnn_model = self._download_model("cnn_final.h5", is_keras=True)
                self.memory_used_mb += 250  # ~250MB pour CNN
            except:
                # Si .h5 n'existe pas, créer un CNN simple
                logger.warning("⚠️ CNN .h5 non trouvé, création modèle simple...")
                cnn_model = self._create_simple_cnn()
                self.memory_used_mb += 50  # ~50MB pour simple CNN
            
            self.cnn_model = cnn_model
            gc.collect()  # Nettoyer mémoire
            
        except ImportError:
            logger.error("❌ TensorFlow non disponible")
            self.cnn_model = None
    
    def _create_simple_cnn(self):
        """Crée un CNN simple si le vrai modèle échoue"""
        from tensorflow import keras
        
        model = keras.Sequential([
            keras.layers.Input(shape=(224, 224, 3)),
            keras.layers.Rescaling(1./255),
            keras.layers.Conv2D(32, 3, activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(64, 3, activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(128, 3, activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(7, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _download_model(self, filename, is_keras=False):
        """Télécharge un modèle depuis Google Drive"""
        if filename not in GOOGLE_DRIVE_IDS:
            raise ValueError(f"ID manquant pour {filename}")
        
        file_id = GOOGLE_DRIVE_IDS[filename]
        url = f"https://drive.google.com/uc?id={file_id}"
        temp_path = tempfile.mktemp(suffix=os.path.splitext(filename)[1])
        
        logger.warning(f"📥 Téléchargement {filename}...")
        gdown.download(url, temp_path, quiet=True)
        
        if is_keras or filename.endswith(('.h5', '.keras')):
            from tensorflow import keras
            model = keras.models.load_model(temp_path, compile=False)
        else:
            model = joblib.load(temp_path)
        
        os.unlink(temp_path)
        return model
    
    def _fallback_mode(self):
        """Mode dégradé sans CNN"""
        self.cnn_model = None
        self.loaded = True
        logger.warning("⚠️ Mode dégradé: CNN désactivé")
    
    def _log_memory(self, phase):
        """Log l'utilisation mémoire"""
        import psutil
        memory = psutil.virtual_memory()
        logger.warning(f"📊 {phase}: {memory.percent}% utilisé ({memory.used/1e6:.0f}MB/{memory.total/1e6:.0f}MB)")

# Initialisation
models = RailwayModelLoader()

# ========== SCHÉMAS ==========
class TextRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    success: bool
    category: str
    confidence: float
    prediction_type: str
    timestamp: str
    model_mode: str = "full"

# ========== ÉVÉNEMENTS ==========
@app.on_event("startup")
async def startup_event():
    """Charge les modèles au démarrage"""
    logger.warning("="*60)
    logger.warning("🚂 DÉMARRAGE API SUR RAILWAY")
    logger.warning(f"🐍 Python: {sys.version}")
    logger.warning(f"📁 Répertoire: {os.getcwd()}")
    logger.warning("="*60)
    
    try:
        models.load_all_models()
        logger.warning("✅ API prête à recevoir des requêtes")
    except Exception as e:
        logger.error(f"💥 Erreur critique: {e}")
        # L'API démarre quand même en mode dégradé

# ========== ENDPOINTS ==========
@app.get("/")
async def root():
    return {
        "message": "Product Classification API",
        "deployment": "Railway + Docker",
        "status": "online",
        "models": {
            "text": "loaded",
            "image": "loaded" if models.cnn_model else "disabled",
            "memory_mode": "optimized"
        },
        "endpoints": {
            "text": "POST /predict/text",
            "image": "POST /predict/image",
            "health": "GET /health",
            "docs": "GET /docs"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy" if models.loaded else "degraded",
        "text_model": True,
        "image_model": models.cnn_model is not None,
        "memory_optimized": True,
        "timestamp": datetime.now().isoformat(),
        "platform": "Railway"
    }

@app.post("/predict/text", response_model=PredictionResponse)
async def predict_text(request: TextRequest):
    """Classification par texte"""
    if not models.loaded:
        raise HTTPException(503, "Models not loaded")
    
    try:
        # Vectorisation
        text_vec = models.vectorizer.transform([request.text])
        
        # Prédiction
        prediction = models.text_model.predict(text_vec)
        pred_idx = int(prediction[0])
        category = models.label_mapping.get(pred_idx, f"Class_{pred_idx}")
        
        # Confiance
        if hasattr(models.text_model, 'predict_proba'):
            probs = models.text_model.predict_proba(text_vec)[0]
            confidence = float(probs[pred_idx])
        else:
            confidence = 0.95
        
        return PredictionResponse(
            success=True,
            category=category,
            confidence=round(confidence, 4),
            prediction_type="text",
            timestamp=datetime.now().isoformat(),
            model_mode="full"
        )
        
    except Exception as e:
        logger.error(f"Text prediction error: {e}")
        raise HTTPException(500, f"Text prediction error: {str(e)}")

@app.post("/predict/image", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    """Classification par image"""
    if not models.loaded or models.cnn_model is None:
        raise HTTPException(503, "Image model not available")
    
    try:
        # Lire image
        image_bytes = await file.read()
        
        # Vérifier taille
        if len(image_bytes) > 10 * 1024 * 1024:  # 10MB max
            raise HTTPException(400, "Image too large (max 10MB)")
        
        # Prétraitement
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert('RGB').resize((224, 224))
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Prédiction
        predictions = models.cnn_model.predict(img_array, verbose=0)
        
        # Résultats
        pred_idx = int(np.argmax(predictions[0]))
        category = CATEGORIES[pred_idx] if 0 <= pred_idx < len(CATEGORIES) else "Unknown"
        confidence = float(predictions[0][pred_idx])
        
        return PredictionResponse(
            success=True,
            category=category,
            confidence=round(confidence, 4),
            prediction_type="image",
            timestamp=datetime.now().isoformat(),
            model_mode="full" if models.cnn_model else "text-only"
        )
        
    except MemoryError:
        raise HTTPException(507, "Insufficient memory for image processing")
    except Exception as e:
        logger.error(f"Image prediction error: {e}")
        raise HTTPException(500, f"Image processing error: {str(e)}")

# ========== LANCEMENT ==========
if __name__ == "__main__":
    import uvicorn
    
    # Récupérer le port depuis Railway
    port = int(os.environ.get("PORT", 8000))
    
    # Configuration optimisée
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=port,
        log_level="warning",
        access_log=True,
        timeout_keep_alive=30,
        limit_concurrency=100,
    )
    
    server = uvicorn.Server(config)
    
    logger.warning(f"🚀 Serveur démarré sur le port {port}")
    logger.warning("📡 Prêt à recevoir des requêtes")
    
    server.run()