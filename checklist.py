"""
Script de vérification et configuration des modèles
Adapté à vos fichiers spécifiques
"""

import os
from pathlib import Path
import sys

# Couleurs pour terminal
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_header(title):
    print("\n" + "=" * 70)
    print(f"{BLUE}{title}{RESET}")
    print("=" * 70)

def print_success(msg):
    print(f"{GREEN}✅ {msg}{RESET}")

def print_error(msg):
    print(f"{RED}❌ {msg}{RESET}")

def print_warning(msg):
    print(f"{YELLOW}⚠️  {msg}{RESET}")

def print_info(msg):
    print(f"{BLUE}ℹ️  {msg}{RESET}")

# Configuration des modèles
MODELS_CONFIG = {
    "Modèle Texte (SVM)": {
        "filename": "final_best_model.pkl",
        "description": "Modèle SVM entraîné pour classification texte",
        "required": True,
        "type": "text"
    },
    "Vectorizer TF-IDF": {
        "filename": "tfidf_vectorizer.pkl",
        "description": "Vectorizer TF-IDF pour transformer le texte",
        "required": True,
        "type": "text"
    },
    "Modèle Image (CNN)": {
        "filename": "cnn_final.keras",
        "description": "Modèle CNN pour classification d'images",
        "required": True,
        "type": "image"
    },
    "Label Encoder": {
        "filename": "label_encoders.pkl",  # Note: avec le typo de votre fichier
        "description": "Encoder pour les labels de catégories",
        "required": True,
        "type": "both"
    }
}

def check_models():
    """Vérifier la présence des modèles"""
    
    print_header("🔍 VÉRIFICATION DES MODÈLES")
    
    models_dir = Path("models")
    
    # Vérifier si le dossier existe
    if not models_dir.exists():
        print_error(f"Dossier 'models/' non trouvé")
        print_info("Création du dossier models/...")
        models_dir.mkdir(parents=True, exist_ok=True)
        print_success("Dossier créé")
    
    print()
    
    # Vérifier chaque modèle
    found_models = {}
    missing_models = []
    
    for name, config in MODELS_CONFIG.items():
        filepath = models_dir / config["filename"]
        
        print(f"\n📦 {name}")
        print(f"   Fichier: {config['filename']}")
        print(f"   Type: {config['type']}")
        
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print_success(f"Trouvé ({size_mb:.2f} MB)")
            found_models[name] = filepath
        else:
            print_error(f"Manquant")
            if config["required"]:
                missing_models.append(config["filename"])
    
    # Résumé
    print_header("📊 RÉSUMÉ")
    
    total = len(MODELS_CONFIG)
    found = len(found_models)
    missing = len(missing_models)
    
    print(f"\n✅ Modèles trouvés: {found}/{total}")
    print(f"❌ Modèles manquants: {missing}/{total}")
    
    if missing_models:
        print_warning("\nModèles à placer dans le dossier 'models/':")
        for filename in missing_models:
            print(f"   • {filename}")
    
    # Déterminer le mode disponible
    text_available = all(
        name in found_models 
        for name, conf in MODELS_CONFIG.items() 
        if conf['type'] in ['text', 'both']
    )
    
    image_available = all(
        name in found_models 
        for name, conf in MODELS_CONFIG.items() 
        if conf['type'] in ['image', 'both']
    )
    
    print_header("🚀 MODE DISPONIBLE")
    
    if text_available and image_available:
        print_success("Mode COMPLET : Texte ✅ + Images ✅")
        mode = "full"
    elif text_available:
        print_warning("Mode TEXTE UNIQUEMENT : Texte ✅ | Images ❌")
        mode = "text_only"
    elif image_available:
        print_warning("Mode IMAGE UNIQUEMENT : Texte ❌ | Images ✅")
        mode = "image_only"
    else:
        print_error("Mode SIMULATION : Aucun modèle disponible")
        mode = "simulation"
    
    return mode, found_models, missing_models

def test_models():
    """Tester le chargement des modèles"""
    
    print_header("🧪 TEST DE CHARGEMENT DES MODÈLES")
    
    models_dir = Path("models")
    
    # Test modèle texte
    print("\n📝 Test Modèle Texte...")
    try:
        import joblib
        
        text_model_path = models_dir / "final_best_model.pkl"
        vectorizer_path = models_dir / "tfidf_vectorizer.pkl"
        
        if text_model_path.exists() and vectorizer_path.exists():
            text_model = joblib.load(text_model_path)
            vectorizer = joblib.load(vectorizer_path)
            
            print_success(f"Modèle texte chargé: {type(text_model).__name__}")
            print_success(f"Vectorizer chargé: {type(vectorizer).__name__}")
            
            # Test prédiction
            test_text = ["soft baby diapers for newborns"]
            X = vectorizer.transform(test_text)
            pred = text_model.predict(X)
            
            print_success(f"Test prédiction: OK (classe {pred[0]})")
        else:
            print_warning("Fichiers modèle texte manquants - Test ignoré")
    
    except Exception as e:
        print_error(f"Erreur test modèle texte: {e}")
    
    # Test modèle image
    print("\n🖼️  Test Modèle Image...")
    try:
        import tensorflow as tf
        from tensorflow import keras
        
        image_model_path = models_dir / "cnn_final.keras"
        encoder_path = models_dir / "label_encoders.pkl"
        
        if image_model_path.exists() and encoder_path.exists():
            image_model = keras.models.load_model(image_model_path)
            label_encoder = joblib.load(encoder_path)
            
            print_success(f"Modèle image chargé: {len(image_model.layers)} couches")
            print_success(f"Label encoder chargé: {len(label_encoder.classes_)} classes")
            
            # Test prédiction
            import numpy as np
            test_img = np.random.rand(1, 224, 224, 3)
            pred = image_model.predict(test_img, verbose=0)
            
            print_success(f"Test prédiction: OK (shape {pred.shape})")
        else:
            print_warning("Fichiers modèle image manquants - Test ignoré")
    
    except ImportError:
        print_warning("TensorFlow non installé - Test modèle image ignoré")
    except Exception as e:
        print_error(f"Erreur test modèle image: {e}")

def show_instructions():
    """Afficher les instructions de placement"""
    
    print_header("📋 INSTRUCTIONS DE PLACEMENT")
    
    print("""
Vos fichiers de modèles doivent être placés dans le dossier 'models/' :

ecommerce_classification_project/
└── models/
    ├── final_best_model.pkl      ← Modèle SVM texte
    ├── tfidf_vectorizer.pkl      ← Vectorizer TF-IDF
    ├── cnn_final.keras           ← Modèle CNN images
    └── label_encoders.pkl       ← Label encoder

📍 Où sont vos modèles actuellement ?
   - Probablement dans le dossier de vos notebooks
   - Ou dans un dossier 'results/' ou 'outputs/'

🔧 Comment les copier :

   Option 1 - Ligne de commande :
   
   # Windows:
   copy "chemin\\vers\\final_best_model.pkl" models\\
   copy "chemin\\vers\\tfidf_vectorizer.pkl" models\\
   copy "chemin\\vers\\cnn_final.keras" models\\
   copy "chemin\\vers\\label_encoders.pkl" models\\
   
   # Linux/Mac:
   cp /chemin/vers/final_best_model.pkl models/
   cp /chemin/vers/tfidf_vectorizer.pkl models/
   cp /chemin/vers/cnn_final.keras models/
   cp /chemin/vers/label_encoders.pkl models/

   Option 2 - Interface graphique :
   
   1. Ouvrir l'explorateur de fichiers
   2. Localiser vos modèles
   3. Copier-coller dans le dossier 'models/'

✅ Une fois copiés, relancez ce script pour vérifier :
   
   python check_models.py
""")

def main():
    """Fonction principale"""
    
    print(f"{BLUE}")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "🔍 VÉRIFICATION DES MODÈLES" + " " * 26 + "║")
    print("║" + " " * 17 + "E-commerce Classifier" + " " * 30 + "║")
    print("╚" + "═" * 68 + "╝")
    print(f"{RESET}")
    
    # Vérifier les modèles
    mode, found, missing = check_models()
    
    print()
    
    # Si modèles trouvés, tester le chargement
    if found:
        test_models()
    
    # Si modèles manquants, afficher instructions
    if missing:
        print()
        show_instructions()
    else:
        print_header("🎉 SUCCÈS")
        print_success("Tous les modèles sont en place !")
        print_info("\nVous pouvez maintenant lancer l'API :")
        print(f"{BLUE}   cd api{RESET}")
        print(f"{BLUE}   uvicorn main_fixed:app --reload{RESET}")
    
    print("\n" + "=" * 70 + "\n")
    
    return 0 if not missing else 1

if __name__ == "__main__":
    sys.exit(main())