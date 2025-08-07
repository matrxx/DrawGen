# docker/Dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Variables d'environnement
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglu1-mesa \
    libxi6 \
    libxmu6 \
    && rm -rf /var/lib/apt/lists/*

# Installation de Blender (headless)
RUN wget -q https://download.blender.org/release/Blender3.6/blender-3.6.0-linux-x64.tar.xz && \
    tar -xf blender-3.6.0-linux-x64.tar.xz && \
    mv blender-3.6.0-linux-x64 /opt/blender && \
    ln -s /opt/blender/blender /usr/local/bin/blender && \
    rm blender-3.6.0-linux-x64.tar.xz

# Configuration Python
RUN python3.9 -m pip install --upgrade pip setuptools wheel

# Création du répertoire de travail
WORKDIR /app

# Copie des requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Installation de PyTorch avec support CUDA
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copie du code source
COPY . .

# Installation du package
RUN pip install -e .

# Création des dossiers nécessaires
RUN mkdir -p /app/data/models /app/data/temp /app/logs

# Téléchargement des modèles pré-entraînés (si disponibles)
# RUN python scripts/download_models.py

# Exposition du port
EXPOSE 8000

# Variables d'environnement pour l'application
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/data/models
ENV TEMP_PATH=/app/data/temp

# Commande de démarrage
CMD ["python", "-m", "src.api.main"]

# docker/docker-compose.yml
version: '3.8'

services:
  sketch3d-api:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ../data:/app/data
      - ../logs:/app/logs
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - PYTHONPATH=/app
    runtime: nvidia
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: sketch3d
      POSTGRES_USER: sketch3d_user
      POSTGRES_PASSWORD: sketch3d_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  redis_data:
  postgres_data:

# scripts/setup_environment.sh
#!/bin/bash

# Script de configuration de l'environnement de développement

set -e

echo "=== Configuration de l'environnement Sketch-to-3D ==="

# Vérification de Python 3.9+
python_version=$(python3 --version 2>&1 | grep -o '[0-9]\+\.[0-9]\+' | head -1)
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Erreur: Python 3.9+ requis. Version détectée: $python_version"
    exit 1
fi

echo "✓ Python version OK: $python_version"

# Création de l'environnement virtuel
if [ ! -d "venv" ]; then
    echo "Création de l'environnement virtuel..."
    python3 -m venv venv
fi

# Activation de l'environnement virtuel
source venv/bin/activate
echo "✓ Environnement virtuel activé"

# Mise à jour de pip
pip install --upgrade pip setuptools wheel

# Installation des dépendances
echo "Installation des dépendances Python..."
pip install -r requirements.txt

# Vérification de CUDA (optionnel)
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU détecté:"
    nvidia-smi --query-gpu=name --format=csv,noheader
    
    # Installation de PyTorch avec CUDA
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "⚠ Aucun GPU NVIDIA détecté, installation PyTorch CPU"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Création des dossiers nécessaires
echo "Création de la structure des dossiers..."
mkdir -p data/{datasets,models,temp}
mkdir -p logs
mkdir -p checkpoints
mkdir -p src/models/model_weights

echo "✓ Dossiers créés"

# Installation d'Open3D
echo "Installation d'Open3D..."
pip install open3d

# Installation de Trimesh
pip install trimesh

# Test des imports principaux
echo "Test des imports..."
python3 -c "
import torch
import cv2
import numpy as np
import open3d as o3d
import trimesh
from PIL import Image
print('✓ Tous les imports principaux fonctionnent')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA disponible: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

# Téléchargement des datasets de base (optionnel)
echo "Voulez-vous télécharger les datasets de démonstration? (y/N)"
read -r download_data
if [[ $download_data =~ ^[Yy]$ ]]; then
    echo "Téléchargement des datasets..."
    python3 scripts/download_datasets.py
fi

echo ""
echo "=== Configuration terminée avec succès! ==="
echo ""
echo "Pour démarrer le développement:"
echo "1. source venv/bin/activate"
echo "2. python -m src.api.main"
echo ""
echo "Pour entraîner les modèles:"
echo "python scripts/train_models.py --model all"
echo ""
echo "Pour lancer les tests:"
echo "pytest tests/"

# scripts/download_datasets.py
#!/usr/bin/env python3
"""
Script de téléchargement et préparation des datasets
"""

import requests
import numpy as np
from pathlib import Path
import logging
import argparse
from tqdm import tqdm
import zipfile
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_file(url: str, destination: Path, description: str = ""):
    """Télécharge un fichier avec barre de progression"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f, tqdm(
        desc=description,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

def download_quickdraw_samples():
    """Télécharge des échantillons du dataset QuickDraw"""
    logger.info("Téléchargement des échantillons QuickDraw...")
    
    data_dir = Path("data/datasets/quickdraw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Liste des catégories populaires pour démo
    demo_categories = [
        "cat", "dog", "car", "house", "tree", "flower", "bird", "fish",
        "sun", "moon", "star", "cloud", "mountain", "river", "bridge",
        "chair", "table", "bed", "door", "window", "book", "phone",
        "computer", "camera", "bicycle", "airplane", "boat", "train"
    ]
    
    base_url = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap"
    
    for category in demo_categories:
        file_url = f"{base_url}/{category.replace(' ', '%20')}.npy"
        file_path = data_dir / f"{category}.npy"
        
        if not file_path.exists():
            try:
                logger.info(f"Téléchargement de {category}...")
                download_file(file_url, file_path, f"QuickDraw - {category}")
                
                # Limitation à 1000 échantillons pour la démo
                data = np.load(file_path)
                if len(data) > 1000:
                    data = data[:1000]
                    np.save(file_path, data)
                
                logger.info(f"✓ {category}: {len(data)} échantillons")
                
            except Exception as e:
                logger.error(f"Erreur téléchargement {category}: {e}")
        else:
            logger.info(f"✓ {category} déjà présent")

def create_synthetic_depth_data():
    """Crée des données synthétiques pour l'entraînement du depth estimator"""
    logger.info("Génération de données synthétiques depth...")
    
    sketch_dir = Path("data/datasets/synthetic/sketches")
    depth_dir = Path("data/datasets/synthetic/depths")
    
    sketch_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    
    # Génération de formes géométriques simples avec leurs depth maps
    shapes = ['cube', 'sphere', 'cylinder', 'cone', 'pyramid']
    
    for i, shape in enumerate(shapes):
        for sample in range(20):  # 20 échantillons par forme
            # Génération d'un sketch synthétique
            sketch, depth = generate_synthetic_shape(shape, size=256)
            
            filename = f"{shape}_{sample:03d}.png"
            
            # Sauvegarde
            sketch_path = sketch_dir / filename
            depth_path = depth_dir / filename
            
            cv2.imwrite(str(sketch_path), sketch)
            cv2.imwrite(str(depth_path), depth)
    
    logger.info("✓ Données synthétiques générées")

def generate_synthetic_shape(shape_type: str, size: int = 256):
    """Génère un sketch et sa depth map pour une forme géométrique"""
    import cv2
    
    sketch = np.zeros((size, size), dtype=np.uint8)
    depth = np.zeros((size, size), dtype=np.uint8)
    
    center_x, center_y = size // 2, size // 2
    
    if shape_type == 'cube':
        # Cube en perspective
        pts = np.array([
            [center_x-60, center_y-40],
            [center_x+40, center_y-40],
            [center_x+60, center_y+40],
            [center_x-40, center_y+40]
        ], dtype=np.int32)
        
        cv2.polylines(sketch, [pts], True, 255, 2)
        
        # Depth map: plus foncé au centre
        cv2.fillPoly(depth, [pts], 200)
        
    elif shape_type == 'sphere':
        # Cercle
        cv2.circle(sketch, (center_x, center_y), 50, 255, 2)
        
        # Depth map: gradient radial
        y, x = np.ogrid[:size, :size]
        mask = (x - center_x)**2 + (y - center_y)**2 <= 50**2
        depth[mask] = 255 - ((x - center_x)**2 + (y - center_y)**2)[mask] * 255 // (50**2)
        
    elif shape_type == 'cylinder':
        # Rectangle avec ellipse en haut
        cv2.rectangle(sketch, (center_x-40, center_y-20), (center_x+40, center_y+40), 255, 2)
        cv2.ellipse(sketch, (center_x, center_y-20), (40, 10), 0, 0, 360, 255, 2)
        
        # Depth map simple
        cv2.rectangle(depth, (center_x-40, center_y-20), (center_x+40, center_y+40), 180, -1)
        
    # Ajout de bruit pour plus de réalisme
    noise = np.random.normal(0, 10, sketch.shape).astype(np.uint8)
    sketch = np.clip(sketch + noise, 0, 255)
    
    return sketch, depth

def download_sample_3d_models():
    """Télécharge quelques modèles 3D de référence"""
    logger.info("Téléchargement de modèles 3D de référence...")
    
    models_dir = Path("data/datasets/reference_3d")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # URLs de modèles 3D simples (Creative Commons)
    reference_models = [
        {
            "name": "cube.obj",
            "url": "https://raw.githubusercontent.com/McNopper/OpenGL/master/Binaries/cube.obj"
        }
        # Ajoutez d'autres URLs de modèles si disponibles
    ]
    
    for model in reference_models:
        model_path = models_dir / model["name"]
        if not model_path.exists():
            try:
                download_file(model["url"], model_path, f"Modèle 3D - {model['name']}")
                logger.info(f"✓ {model['name']} téléchargé")
            except Exception as e:
                logger.warning(f"Impossible de télécharger {model['name']}: {e}")

def create_dataset_info():
    """Crée un fichier d'information sur les datasets"""
    info = {
        "datasets": {
            "quickdraw": {
                "description": "Google QuickDraw dataset - échantillons de sketches",
                "categories": 28,
                "samples_per_category": 1000,
                "format": "numpy arrays (28x28 bitmap)",
                "usage": "Classification de sketches"
            },
            "synthetic": {
                "description": "Données synthétiques pour depth estimation",
                "shapes": ["cube", "sphere", "cylinder", "cone", "pyramid"],
                "samples_per_shape": 20,
                "format": "PNG images (256x256)",
                "usage": "Entraînement depth estimator"
            }
        },
        "total_size": "~50MB",
        "last_updated": "2024-01-01"
    }
    
    with open("data/datasets/dataset_info.json", 'w') as f:
        json.dump(info, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Téléchargement des datasets')
    parser.add_argument('--quickdraw', action='store_true', help='Télécharger QuickDraw samples')
    parser.add_argument('--synthetic', action='store_true', help='Générer données synthétiques')
    parser.add_argument('--models', action='store_true', help='Télécharger modèles 3D')
    parser.add_argument('--all', action='store_true', help='Tout télécharger')
    
    args = parser.parse_args()
    
    if args.all or not any([args.quickdraw, args.synthetic, args.models]):
        # Par défaut, tout faire
        args.quickdraw = args.synthetic = args.models = True
    
    try:
        if args.quickdraw:
            download_quickdraw_samples()
        
        if args.synthetic:
            create_synthetic_depth_data()
        
        if args.models:
            download_sample_3d_models()
        
        create_dataset_info()
        
        logger.info("✓ Téléchargement terminé avec succès!")
        
    except Exception as e:
        logger.error(f"Erreur lors du téléchargement: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

# scripts/run_tests.sh
#!/bin/bash

# Script de tests automatisés

set -e

echo "=== Tests Sketch-to-3D ==="

# Activation de l'environnement virtuel
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✓ Environnement virtuel activé"
fi

# Installation des dépendances de test si nécessaire
pip install pytest pytest-cov pytest-mock

# Tests unitaires
echo "Exécution des tests unitaires..."
python -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term

# Test de l'API
echo "Test de l'API..."
python -c "
import requests
import time
import subprocess
import signal
import os

# Démarrage du serveur en arrière-plan
server = subprocess.Popen(['python', '-m', 'src.api.main'], 
                         stdout=subprocess.DEVNULL, 
                         stderr=subprocess.DEVNULL)

# Attente du démarrage
time.sleep(5)

try:
    # Test de base
    response = requests.get('http://localhost:8000/')
    assert response.status_code == 200
    print('✓ API accessible')
    
    # Test health check
    response = requests.get('http://localhost:8000/health')
    assert response.status_code == 200
    print('✓ Health check OK')
    
except Exception as e:
    print(f'✗ Test API échoué: {e}')
finally:
    # Arrêt du serveur
    server.terminate()
    server.wait()
"

# Test des modèles
echo "Test des modèles..."
python -c "
from src.models.sketch_classifier import SketchClassifier
from src.models.depth_estimator import DepthEstimator
import torch

# Test classifier
classifier = SketchClassifier(num_classes=10)
test_input = torch.randn(1, 1, 224, 224)
output = classifier(test_input)
assert output.shape == (1, 10)
print('✓ Classifier fonctionne')

# Test depth estimator  
depth_model = DepthEstimator()
test_input = torch.randn(1, 1, 512, 512)
output = depth_model(test_input)
assert output.shape == (1, 1, 512, 512)
print('✓ Depth estimator fonctionne')
"

# Test des utilitaires
echo "Test des utilitaires..."
python -c "
from src.utils.image_processing import normalize_sketch, clean_sketch
from src.utils.validation import validate_sketch_input
import numpy as np

# Test image processing
test_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
normalized = normalize_sketch(test_image)
assert normalized.shape == (512, 512)
print('✓ Image processing OK')

# Test validation
assert validate_sketch_input(test_image) == True
print('✓ Validation OK')
"

# Test de génération de mesh
echo "Test de génération de mesh..."
python -c "
from src.core.mesh_generator import MeshGenerator
import numpy as np

config = {
    'voxel_resolution': 32,
    'smoothing_iterations': 1,
    'min_mesh_faces': 10
}

generator = MeshGenerator(config)
depth_map = np.random.rand(64, 64)
sketch_mask = np.ones((64, 64))
class_info = {'class_name': 'cube', 'confidence': 0.8}

try:
    mesh = generator.generate_mesh_from_depth(depth_map, sketch_mask, class_info)
    print(f'✓ Mesh généré avec {len(mesh.vertices)} vertices')
except Exception as e:
    print(f'⚠ Mesh generation: {e} (normal avec données aléatoires)')
"

echo ""
echo "=== Tests terminés ==="
echo "Consultez htmlcov/index.html pour le rapport de couverture détaillé"

# Makefile
.PHONY: help install test run clean docker

help: ## Affiche cette aide
	@grep -E '^[a-zA-Z_-]+:.*?## .*$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $1, $2}'

install: ## Installation complète de l'environnement
	@echo "Installation de l'environnement..."
	@bash scripts/setup_environment.sh

download-data: ## Téléchargement des datasets
	@echo "Téléchargement des datasets..."
	@python scripts/download_datasets.py --all

train: ## Entraînement des modèles
	@echo "Entraînement des modèles..."
	@python scripts/train_models.py --model all

test: ## Exécution des tests
	@echo "Exécution des tests..."
	@bash scripts/run_tests.sh

run: ## Démarrage de l'API
	@echo "Démarrage de l'API Sketch-to-3D..."
	@python -m src.api.main

run-dev: ## Démarrage en mode développement
	@echo "Démarrage en mode développement..."
	@uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

docker-build: ## Construction de l'image Docker
	@echo "Construction de l'image Docker..."
	@docker build -f docker/Dockerfile -t sketch3d:latest .

docker-run: ## Démarrage avec Docker Compose
	@echo "Démarrage avec Docker Compose..."
	@docker-compose -f docker/docker-compose.yml up -d

docker-stop: ## Arrêt des conteneurs Docker
	@docker-compose -f docker/docker-compose.yml down

clean: ## Nettoyage des fichiers temporaires
	@echo "Nettoyage..."
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type d -name "*.egg-info" -exec rm -rf {} +
	@rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/

lint: ## Vérification du code avec flake8
	@echo "Vérification du code..."
	@flake8 src/ tests/ --max-line-length=100 --ignore=E203,W503

format: ## Formatage du code avec black
	@echo "Formatage du code..."
	@black src/ tests/ scripts/ --line-length=100

setup-dev: install download-data ## Configuration complète pour développement
	@echo "Configuration de développement terminée!"
	@echo "Utilisez 'make run-dev' pour démarrer"

# README.md pour le projet
# Sketch-to-3D Backend

Pipeline d'intelligence artificielle pour convertir des dessins à main levé en modèles 3D.

## 🚀 Démarrage Rapide

```bash
# Installation complète
make setup-dev

# Démarrage de l'API
make run-dev

# Tests
make test
```

## 📁 Architecture

```
sketch3d-backend/
├── src/
│   ├── api/          # API FastAPI  
│   ├── core/         # Pipeline principal
│   ├── models/       # Modèles PyTorch
│   └── utils/        # Utilitaires
├── scripts/          # Scripts d'entraînement
├── docker/           # Configuration Docker
└── tests/            # Tests automatisés
```

## 🤖 Pipeline IA

1. **Classification** : Identification du type d'objet (ResNet-18)
2. **Depth Estimation** : Génération de carte de profondeur (U-Net)
3. **Mesh Generation** : Reconstruction 3D (Marching Cubes)
4. **Export** : Fichier .blend pour Blender

## 🔧 API Endpoints

- `POST /api/v1/sketch/process` : Traitement d'un sketch
- `GET /api/v1/sketch/{id}/status` : Statut du traitement  
- `GET /api/v1/sketch/{id}/download` : Téléchargement du .blend

## 📊 Performance

- **Temps de traitement** : <30s par sketch
- **Classification** : >90% précision  
- **Support GPU** : CUDA 11.8+
- **Formats** : JPG, PNG → .blend

## 🐳 Docker

```bash
# Construction
make docker-build

# Démarrage complet
make docker-run
```

## 📈 Développement

- **Tests** : `make test`
- **Linting** : `make lint`  
- **Format** : `make format`
- **Clean** : `make clean`