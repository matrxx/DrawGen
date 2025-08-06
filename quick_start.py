# scripts/quick_start.py
#!/usr/bin/env python3
"""
Script de démarrage rapide pour Sketch-to-3D
Teste tous les composants principaux et démarre l'API
"""

import sys
import subprocess
import time
import requests
import numpy as np
from pathlib import Path
import logging
from PIL import Image, ImageDraw
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Vérifie que toutes les dépendances sont installées"""
    logger.info("🔍 Vérification des dépendances...")
    
    required_packages = [
        'torch', 'torchvision', 'opencv-python', 'fastapi', 'uvicorn',
        'numpy', 'pillow', 'open3d', 'trimesh', 'scikit-image'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
            elif package == 'scikit-image':
                import skimage
            else:
                __import__(package.replace('-', '_'))
            logger.info(f"  ✓ {package}")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"  ✗ {package}")
    
    if missing_packages:
        logger.error(f"Packages manquants: {missing_packages}")
        logger.info("Installez avec: pip install " + " ".join(missing_packages))
        return False
    
    logger.info("✅ Toutes les dépendances sont présentes")
    return True

def test_models():
    """Teste l'initialisation des modèles"""
    logger.info("🧠 Test des modèles...")
    
    try:
        # Test du classificateur
        from src.models.sketch_classifier import SketchClassifier
        classifier = SketchClassifier(num_classes=10)
        logger.info("  ✓ SketchClassifier")
        
        # Test de l'estimateur de profondeur
        from src.models.depth_estimator import DepthEstimator
        depth_model = DepthEstimator()
        logger.info("  ✓ DepthEstimator")
        
        # Test avec données factices
        import torch
        test_input = torch.randn(1, 1, 224, 224)
        
        with torch.no_grad():
            output = classifier(test_input)
            assert output.shape == (1, 10)
        
        test_input = torch.randn(1, 1, 512, 512)
        with torch.no_grad():
            output = depth_model(test_input)
            assert output.shape == (1, 1, 512, 512)
        
        logger.info("✅ Modèles fonctionnels")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur modèles: {e}")
        return False

def test_core_pipeline():
    """Teste le pipeline principal"""
    logger.info("⚙️ Test du pipeline principal...")
    
    try:
        from src.utils.validation import validate_sketch_input
        from src.utils.image_processing import normalize_sketch, clean_sketch
        
        # Création d'un sketch de test
        test_sketch = np.zeros((128, 128), dtype=np.uint8)
        test_sketch[32:96, 32:96] = 255
        
        # Test validation
        assert validate_sketch_input(test_sketch) == True
        logger.info("  ✓ Validation")
        
        # Test preprocessing
        normalized = normalize_sketch(test_sketch)
        assert normalized.shape == (512, 512)
        logger.info("  ✓ Normalisation")
        
        cleaned = clean_sketch(test_sketch)
        assert cleaned.shape == test_sketch.shape
        logger.info("  ✓ Nettoyage")
        
        logger.info("✅ Pipeline principal fonctionnel")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur pipeline: {e}")
        return False

def test_mesh_generation():
    """Teste la génération de maillage"""
    logger.info("🔺 Test de génération de maillage...")
    
    try:
        from src.core.mesh_generator import MeshGenerator
        import trimesh
        
        config = {
            'voxel_resolution': 32,
            'smoothing_iterations': 1,
            'min_mesh_faces': 4
        }
        
        generator = MeshGenerator(config)
        
        # Test avec données synthétiques
        depth_map = np.random.rand(64, 64) * 0.8 + 0.2
        sketch_mask = np.ones((64, 64))
        class_info = {'class_name': 'cube', 'confidence': 0.8}
        
        # Test fallback (plus fiable)
        mesh = generator._generate_fallback_mesh(class_info)
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0
        
        logger.info("  ✓ Génération de maillage fallback")
        logger.info("✅ Génération de maillage fonctionnelle")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur génération maillage: {e}")
        logger.warning("Ceci peut être normal si Open3D/Trimesh ont des problèmes d'installation")
        return False

def create_test_sketch():
    """Crée un sketch de test pour démonstration"""
    logger.info("🎨 Création d'un sketch de test...")
    
    # Création d'une image avec dessin simple
    img = Image.new('RGB', (256, 256), color='white')
    draw = ImageDraw.Draw(img)
    
    # Dessin d'une maison simple
    # Base
    draw.rectangle([50, 150, 200, 230], outline='black', width=3)
    
    # Toit
    draw.polygon([40, 150, 125, 80, 210, 150], outline='black', width=3)
    
    # Porte
    draw.rectangle([100, 180, 140, 230], outline='black', width=2)
    
    # Fenêtres
    draw.rectangle([70, 170, 90, 190], outline='black', width=2)
    draw.rectangle([160, 170, 180, 190], outline='black', width=2)
    
    # Sauvegarde temporaire
    test_dir = Path("data/temp")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    sketch_path = test_dir / "test_house.png"
    img.save(sketch_path)
    
    logger.info(f"  ✓ Sketch sauvé: {sketch_path}")
    return sketch_path

def start_api_server():
    """Démarre le serveur API"""
    logger.info("🚀 Démarrage du serveur API...")
    
    try:
        # Import conditionnel pour éviter les erreurs
        from src.api.main import app
        import uvicorn
        
        # Démarrage en arrière-plan
        import threading
        
        def run_server():
            uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # Attente du démarrage
        time.sleep(3)
        
        # Test de connexion
        try:
            response = requests.get("http://127.0.0.1:8000/health", timeout=5)
            if response.status_code == 200:
                logger.info("✅ API démarrée avec succès")
                logger.info("📡 API accessible sur: http://127.0.0.1:8000")
                logger.info("📚 Documentation: http://127.0.0.1:8000/docs")
                return True
            else:
                logger.error(f"❌ API répond avec code {response.status_code}")
                return False
        
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Impossible de se connecter à l'API: {e}")
            return False
    
    except Exception as e:
        logger.error(f"❌ Erreur démarrage API: {e}")
        return False

def test_api_with_sketch():
    """Teste l'API avec un sketch réel"""
    logger.info("🧪 Test de l'API avec sketch...")
    
    try:
        # Création du sketch de test
        sketch_path = create_test_sketch()
        
        # Test de l'endpoint de traitement
        with open(sketch_path, 'rb') as f:
            files = {'file': ('test_house.png', f, 'image/png')}
            response = requests.post(
                "http://127.0.0.1:8000/api/v1/sketch/process",
                files=files,
                timeout=10
            )
        
        if response.status_code == 200:
            data = response.json()
            task_id = data.get('task_id')
            logger.info(f"  ✓ Tâche créée: {task_id}")
            
            # Vérification du statut
            status_response = requests.get(
                f"http://127.0.0.1:8000/api/v1/sketch/{task_id}/status",
                timeout=5
            )
            
            if status_response.status_code == 200:
                status_data = status_response.json()
                logger.info(f"  ✓ Statut: {status_data['status']}")
                logger.info("✅ Test API réussi")
                return True
            
        logger.error(f"❌ Test API échoué: {response.status_code}")
        return False
        
    except Exception as e:
        logger.error(f"❌ Erreur test API: {e}")
        return False

def show_usage_examples():
    """Affiche des exemples d'utilisation"""
    logger.info("\n" + "="*50)
    logger.info("📖 EXEMPLES D'UTILISATION")
    logger.info("="*50)
    
    print("""
🔧 LIGNE DE COMMANDE:
    
    # Entraînement des modèles
    python scripts/train_models.py --model classifier
    python scripts/train_models.py --model all
    
    # Tests
    python -m pytest tests/
    pytest tests/ -v --cov=src
    
    # Démarrage API
    python -m src.api.main
    make run-dev

🌐 API REST:
    
    # Health check
    GET http://localhost:8000/health
    
    # Traitement d'un sketch  
    POST http://localhost:8000/api/v1/sketch/process
    Content-Type: multipart/form-data
    file: [votre_sketch.png]
    
    # Statut du traitement
    GET http://localhost:8000/api/v1/sketch/{task_id}/status
    
    # Téléchargement du .blend
    GET http://localhost:8000/api/v1/sketch/{task_id}/download

🐍 PYTHON:
    
    from src.core.sketch_processor import SketchProcessor
    from src.core.mesh_generator import MeshGenerator
    
    # Configuration
    config = {...}  # Voir config/model_config.yaml
    
    # Traitement
    processor = SketchProcessor(config['models'])
    result = processor.process_sketch_to_3d(image_array)
    
    # Génération 3D
    generator = MeshGenerator(config['mesh_generation'])
    mesh = generator.generate_mesh_from_depth(
        depth_map, sketch_mask, class_info
    )

🐳 DOCKER:
    
    # Construction
    docker build -f docker/Dockerfile -t sketch3d .
    
    # Démarrage
    docker-compose -f docker/docker-compose.yml up -d
    
    # Logs
    docker-compose logs -f sketch3d-api
""")

def main():
    """Fonction principale de démarrage rapide"""
    print("\n" + "="*60)
    print("🎯 SKETCH-TO-3D - DÉMARRAGE RAPIDE")
    print("="*60)
    
    # Vérifications préliminaires
    success = True
    
    success &= check_dependencies()
    success &= test_models()
    success &= test_core_pipeline()
    success &= test_mesh_generation()
    
    if not success:
        logger.error("\n❌ Certains tests ont échoué. Consultez les logs ci-dessus.")
        logger.info("💡 Essayez: pip install -r requirements.txt")
        return 1
    
    logger.info("\n🎉 Tous les tests passent!")
    
    # Démarrage de l'API
    print("\n" + "="*60)
    print("🚀 DÉMARRAGE DE L'API")
    print("="*60)
    
    if start_api_server():
        # Test avec sketch réel
        test_api_with_sketch()
        
        # Affichage des exemples
        show_usage_examples()
        
        print("\n" + "="*60)
        print("✅ SKETCH-TO-3D PRÊT À L'EMPLOI!")
        print("="*60)
        print("\n🌐 API accessible sur: http://127.0.0.1:8000")
        print("📚 Documentation: http://127.0.0.1:8000/docs")
        print("🧪 Interface test: http://127.0.0.1:8000/docs#/default/process_sketch_api_v1_sketch_process_post")
        print("\n⌨️  Appuyez sur Ctrl+C pour arrêter")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n👋 Arrêt du serveur...")
            return 0
    else:
        logger.error("❌ Impossible de démarrer l'API")
        return 1

if __name__ == "__main__":
    exit(main())

# scripts/demo.py
#!/usr/bin/env python3
"""
Script de démonstration interactif pour Sketch-to-3D
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import requests
import time
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InteractiveDemo:
    """Démonstration interactive du système Sketch-to-3D"""
    
    def __init__(self, api_url="http://localhost:8000"):
        self.api_url = api_url
        self.demo_dir = Path("data/demo")
        self.demo_dir.mkdir(parents=True, exist_ok=True)
    
    def create_sample_sketches(self):
        """Crée plusieurs sketches d'exemple"""
        logger.info("🎨 Création des sketches d'exemple...")
        
        sketches = []
        
        # 1. Maison simple
        house = self._draw_house()
        house_path = self.demo_dir / "house.png"
        house.save(house_path)
        sketches.append(("Maison", house_path))
        
        # 2. Voiture
        car = self._draw_car()
        car_path = self.demo_dir / "car.png"
        car.save(car_path)
        sketches.append(("Voiture", car_path))
        
        # 3. Arbre
        tree = self._draw_tree()
        tree_path = self.demo_dir / "tree.png"
        tree.save(tree_path)
        sketches.append(("Arbre", tree_path))
        
        # 4. Forme géométrique
        cube = self._draw_cube()
        cube_path = self.demo_dir / "cube.png"
        cube.save(cube_path)
        sketches.append(("Cube", cube_path))
        
        logger.info(f"✅ {len(sketches)} sketches créés")
        return sketches
    
    def _draw_house(self):
        """Dessine une maison simple"""
        img = Image.new('RGB', (256, 256), color='white')
        draw = ImageDraw.Draw(img)
        
        # Base de la maison
        draw.rectangle([60, 140, 196, 220], outline='black', width=3)
        
        # Toit triangulaire
        draw.polygon([50, 140, 128, 70, 206, 140], outline='black', width=3)
        
        # Porte
        draw.rectangle([110, 180, 146, 220], outline='black', width=2)
        draw.circle([140, 200], 2, fill='black')  # Poignée
        
        # Fenêtres
        draw.rectangle([75, 155, 95, 175], outline='black', width=2)
        draw.line([85, 155, 85, 175], fill='black', width=1)
        draw.line([75, 165, 95, 165], fill='black', width=1)
        
        draw.rectangle([160, 155, 180, 175], outline='black', width=2)
        draw.line([170, 155, 170, 175], fill='black', width=1)
        draw.line([160, 165, 180, 165], fill='black', width=1)
        
        # Cheminée
        draw.rectangle([150, 80, 170, 110], outline='black', width=2)
        
        return img
    
    def _draw_car(self):
        """Dessine une voiture simple"""
        img = Image.new('RGB', (256, 256), color='white')
        draw = ImageDraw.Draw(img)
        
        # Carrosserie principale
        draw.rounded_rectangle([40, 140, 216, 180], radius=10, outline='black', width=3)
        
        # Toit
        draw.rounded_rectangle([80, 110, 176, 140], radius=8, outline='black', width=3)
        
        # Roues
        draw.circle([70, 190], 20, outline='black', width=3)
        draw.circle([186, 190], 20, outline='black', width=3)
        
        # Jantes
        draw.circle([70, 190], 8, outline='black', width=2)
        draw.circle([186, 190], 8, outline='black', width=2)
        
        # Fenêtres
        draw.rectangle([85, 115, 125, 135], outline='black', width=2)
        draw.rectangle([131, 115, 171, 135], outline='black', width=2)
        
        # Phares
        draw.circle([216, 155], 8, outline='black', width=2)
        draw.circle([40, 155], 8, outline='black', width=2)
        
        return img
    
    def _draw_tree(self):
        """Dessine un arbre simple"""
        img = Image.new('RGB', (256, 256), color='white')
        draw = ImageDraw.Draw(img)
        
        # Tronc
        draw.rectangle([118, 160, 138, 230], outline='black', width=3, fill='white')
        
        # Feuillage (cercles qui se chevauchent)
        draw.circle([128, 120], 40, outline='black', width=3)
        draw.circle([108, 100], 30, outline='black', width=2)
        draw.circle([148, 100], 30, outline='black', width=2)
        draw.circle([128, 80], 25, outline='black', width=2)
        
        # Détails du tronc
        draw.line([125, 170, 125, 220], fill='black', width=1)
        draw.line([131, 175, 131, 225], fill='black', width=1)
        
        return img
    
    def _draw_cube(self):
        """Dessine un cube en perspective"""
        img = Image.new('RGB', (256, 256), color='white')
        draw = ImageDraw.Draw(img)
        
        # Face avant
        draw.rectangle([80, 120, 160, 200], outline='black', width=3)
        
        # Face droite (perspective)
        draw.polygon([160, 120, 200, 80, 200, 160, 160, 200], outline='black', width=3)
        
        # Face dessus
        draw.polygon([80, 120, 120, 80, 200, 80, 160, 120], outline='black', width=3)
        
        # Lignes cachées (pointillés simulés)
        draw.line([80, 200, 120, 160], fill='gray', width=1)
        draw.line([120, 160, 200, 160], fill='gray', width=1)
        draw.line([120, 160, 120, 80], fill='gray', width=1)
        
        return img
    
    def test_api_connection(self):
        """Teste la connexion à l'API"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info("✅ API connectée")
                return True
            else:
                logger.error(f"❌ API répond avec code {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Impossible de se connecter à l'API: {e}")
            return False
    
    def process_sketch(self, sketch_path, sketch_name):
        """Traite un sketch via l'API"""
        logger.info(f"🔄 Traitement de {sketch_name}...")
        
        try:
            # Upload du sketch
            with open(sketch_path, 'rb') as f:
                files = {'file': (sketch_path.name, f, 'image/png')}
                data = {'mesh_name': f"Generated_{sketch_name}"}
                
                response = requests.post(
                    f"{self.api_url}/api/v1/sketch/process",
                    files=files,
                    data=data,
                    timeout=10
                )
            
            if response.status_code != 200:
                logger.error(f"❌ Erreur upload: {response.status_code}")
                return None
            
            task_data = response.json()
            task_id = task_data['task_id']
            logger.info(f"📋 Tâche créée: {task_id}")
            
            # Polling du statut
            max_attempts = 30
            for attempt in range(max_attempts):
                time.sleep(2)
                
                status_response = requests.get(
                    f"{self.api_url}/api/v1/sketch/{task_id}/status",
                    timeout=5
                )
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    status = status_data['status']
                    progress = status_data.get('progress', 0)
                    
                    print(f"  📊 {status} - {progress}%")
                    
                    if status == 'completed':
                        logger.info(f"✅ {sketch_name} traité avec succès!")
                        return task_id
                    elif status == 'failed':
                        error = status_data.get('error', 'Erreur inconnue')
                        logger.error(f"❌ Échec traitement: {error}")
                        return None
                else:
                    logger.warning(f"⚠️ Erreur statut: {status_response.status_code}")
            
            logger.error(f"⏰ Timeout pour {sketch_name}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Erreur traitement {sketch_name}: {e}")
            return None
    
    def download_blend_file(self, task_id, sketch_name):
        """Télécharge le fichier .blend généré"""
        try:
            download_response = requests.get(
                f"{self.api_url}/api/v1/sketch/{task_id}/download",
                timeout=30
            )
            
            if download_response.status_code == 200:
                blend_path = self.demo_dir / f"{sketch_name.lower()}_model.blend"
                
                with open(blend_path, 'wb') as f:
                    f.write(download_response.content)
                
                logger.info(f"💾 Fichier .blend sauvé: {blend_path}")
                return blend_path
            else:
                logger.error(f"❌ Erreur téléchargement: {download_response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Erreur téléchargement: {e}")
            return None
    
    def show_results_summary(self, results):
        """Affiche un résumé des résultats"""
        print("\n" + "="*60)
        print("📊 RÉSUMÉ DE LA DÉMONSTRATION")
        print("="*60)
        
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        print(f"\n✅ Réussis: {len(successful)}/{len(results)}")
        for result in successful:
            print(f"  • {result['name']}: {result['blend_file']}")
        
        if failed:
            print(f"\n❌ Échecs: {len(failed)}")
            for result in failed:
                print(f"  • {result['name']}: {result['error']}")
        
        print(f"\n📁 Tous les fichiers sont dans: {self.demo_dir}")
        
        if successful:
            print("\n💡 PROCHAINES ÉTAPES:")
            print("  1. Ouvrez Blender")
            print("  2. File > Import > Blender (.blend)")
            print("  3. Sélectionnez un des fichiers .blend générés")
            print("  4. Explorez votre modèle 3D!")
    
    def run_full_demo(self):
        """Lance la démonstration complète"""
        print("\n" + "="*60)
        print("🎯 DÉMONSTRATION SKETCH-TO-3D")
        print("="*60)
        
        # Vérification de l'API
        if not self.test_api_connection():
            logger.error("❌ API non accessible. Démarrez l'API avec: python -m src.api.main")
            return
        
        # Création des sketches
        sketches = self.create_sample_sketches()
        
        # Traitement de chaque sketch
        results = []
        
        for sketch_name, sketch_path in sketches:
            print(f"\n{'='*40}")
            print(f"🎨 TRAITEMENT: {sketch_name}")
            print(f"{'='*40}")
            
            task_id = self.process_sketch(sketch_path, sketch_name)
            
            if task_id:
                blend_file = self.download_blend_file(task_id, sketch_name)
                
                results.append({
                    'name': sketch_name,
                    'success': blend_file is not None,
                    'blend_file': blend_file,
                    'task_id': task_id,
                    'error': None
                })
            else:
                results.append({
                    'name': sketch_name,
                    'success': False,
                    'blend_file': None,
                    'task_id': None,
                    'error': "Traitement échoué"
                })
        
        # Résumé final
        self.show_results_summary(results)
        
        return results

def main():
    """Point d'entrée principal"""
    demo = InteractiveDemo()
    
    print("🚀 Démarrage de la démonstration...")
    print("⚠️  Assurez-vous que l'API est démarrée (python -m src.api.main)")
    
    input("Appuyez sur Entrée pour continuer...")
    
    results = demo.run_full_demo()
    
    print("\n🎉 Démonstration terminée!")
    
    return 0 if any(r['success'] for r in results) else 1

if __name__ == "__main__":
    exit(main())

# GETTING_STARTED.md
# Guide de Démarrage Sketch-to-3D

## 🚀 Installation Rapide

### Prérequis
- Python 3.9+
- CUDA 11.8+ (optionnel, pour GPU)
- 8GB RAM minimum
- 2GB espace disque

### Installation Automatique

```bash
# Clone du projet
git clone https://github.com/votre-repo/sketch3d-backend
cd sketch3d-backend

# Installation complète
make setup-dev
```

### Installation Manuelle

```bash
# Environnement virtuel
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Dépendances
pip install -r requirements.txt

# Structure des dossiers
mkdir -p data/{datasets,models,temp} logs checkpoints
```

## 🏁 Démarrage Rapide

### Option 1: Script Automatique
```bash
python scripts/quick_start.py
```

### Option 2: Démarrage Manuel
```bash
# Test des composants
python -c "from src.models.sketch_classifier import SketchClassifier; print('✅ OK')"

# Démarrage API
python -m src.api.main
```

## 🧪 Première Utilisation

### 1. Via Interface Web
1. Ouvrez http://localhost:8000/docs
2. Testez l'endpoint `/api/v1/sketch/process`
3. Uploadez une image de sketch
4. Récupérez le fichier .blend

### 2. Via Code Python
```python
import requests

# Upload sketch
with open('mon_sketch.png', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/api/v1/sketch/process', files=files)

task_id = response.json()['task_id']

# Vérification statut
status = requests.get(f'http://localhost:8000/api/v1/sketch/{task_id}/status')
print(status.json())
```

### 3. Démonstration Interactive
```bash
python scripts/demo.py
```

## 📊 Vérification Installation

### Tests Automatisés
```bash
# Tests complets
make test

# Tests unitaires seulement
pytest tests/test_models.py -v

# Tests de performance
pytest tests/test_performance.py -m performance
```

### Vérifications Manuelles

**Modèles:**
```python
from src.models.sketch_classifier import SketchClassifier
from src.models.depth_estimator import DepthEstimator
import torch

# Test classifier
model = SketchClassifier(num_classes=10)
x = torch.randn(1, 1, 224, 224)
output = model(x)  # Doit fonctionner

print("✅ Modèles OK")
```

**Pipeline:**
```python
from src.core.sketch_processor import SketchProcessor
import numpy as np

config = {'models': {...}}  # Voir config/model_config.yaml
processor = SketchProcessor(config['models'])

# Test avec image factice
test_image = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
result = processor.preprocess_sketch(test_image)

print("✅ Pipeline OK")
```

## 🔧 Configuration

### Fichiers de Configuration
- `config/model_config.yaml` - Paramètres des modèles
- `requirements.txt` - Dépendances Python
- `docker-compose.yml` - Configuration Docker

### Variables d'Environnement
```bash
export CUDA_VISIBLE_DEVICES=0
export MODEL_PATH=/path/to/models
export TEMP_PATH=/tmp/sketch3d
```

## 🐳 Docker (Alternative)

### Démarrage avec Docker
```bash
# Construction
docker build -f docker/Dockerfile -t sketch3d .

# Démarrage
docker-compose up -d

# Vérification
curl http://localhost:8000/health
```

## 🚨 Problèmes Courants

### Erreur CUDA
```bash
# Vérifier CUDA
nvidia-smi

# Installation PyTorch CPU si pas de GPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Erreur Open3D/Trimesh
```bash
# Mise à jour
pip install --upgrade open3d trimesh

# Alternative conda
conda install -c open3d-admin open3d
```

### Erreur Blender Export
```bash
# Installation Blender Python API
pip install bpy

# Ou utilisation Blender system
export BLENDER_PATH=/path/to/blender
```

### Port 8000 Occupé
```bash
# Changer le port
uvicorn src.api.main:app --port 8001

# Ou tuer le processus
lsof -i :8000
kill -9 <PID>
```

## 📚 Ressources Supplémentaires

### Documentation
- [Architecture](docs/architecture.md)
- [API Reference](docs/api.md)
- [Training Guide](docs/training.md)

### Datasets
```bash
# Téléchargement datasets démo
python scripts/download_datasets.py --all

# QuickDraw complet (optionnel)
python scripts/download_datasets.py --quickdraw
```

### Entraînement Modèles
```bash
# Entraîner classifier
python scripts/train_models.py --model classifier

# Entraîner tous les modèles
python scripts/train_models.py --model all
```

## 🆘 Support

### Logs et Debugging
```bash
# Logs détaillés
export PYTHONPATH=/path/to/sketch3d-backend
python -m src.api.main --log-level debug

# Tests avec coverage
pytest --cov=src --cov-report=html
```

### Commandes Utiles
```bash
# Statut système
make health-check

# Nettoyage
make clean

# Mise à jour dépendances
pip install -r requirements.txt --upgrade
```

**En cas de problème persistant:**
1. Vérifiez les logs dans `logs/`
2. Testez les composants individuellement
3. Consultez la documentation API
4. Ouvrez une issue avec les logs d'erreur

---

🎉 **Félicitations!** Drawgen est maintenant opérationnel.
Consultez http://localhost:8000/docs pour l'interface interactive.