# src/api/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from PIL import Image
import io
import uuid
import tempfile
import asyncio
from pathlib import Path
import logging
from typing import Dict, Any, Optional

from ..core.sketch_processor import SketchProcessor
from ..core.mesh_generator import MeshGenerator
from ..core.file_handler import BlenderExporter
from ..utils.validation import validate_sketch_input, validate_mesh_output
from ..config.settings import get_config

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
config = get_config()

# Initialisation de l'application
app = FastAPI(
    title="Sketch-to-3D API",
    description="API pour convertir des sketches en modèles 3D",
    version="1.0.0"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialisation des composants
sketch_processor = SketchProcessor(config['models'])
mesh_generator = MeshGenerator(config['mesh_generation'])
blender_exporter = BlenderExporter()

# Stockage temporaire des tâches
processing_tasks = {}

class ProcessingStatus:
    def __init__(self):
        self.status = "queued"
        self.progress = 0
        self.message = ""
        self.result = None
        self.error = None

@app.get("/")
async def root():
    """Endpoint de base"""
    return {"message": "Sketch-to-3D API", "status": "running", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Check de santé de l'API"""
    return {
        "status": "healthy",
        "components": {
            "sketch_processor": "loaded",
            "mesh_generator": "loaded",
            "blender_exporter": "loaded"
        },
        "device": str(sketch_processor.device)
    }

@app.post("/api/v1/sketch/process")
async def process_sketch(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    mesh_name: str = "Generated_Mesh"
):
    """
    Endpoint principal pour traiter un sketch et générer un modèle 3D
    """
    # Génération d'un ID unique pour la tâche
    task_id = str(uuid.uuid4())
    
    try:
        # Validation du fichier
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Le fichier doit être une image")
        
        # Lecture de l'image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Conversion en array numpy
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_array = np.array(image)
        
        # Validation de l'image
        if not validate_sketch_input(image_array):
            raise HTTPException(status_code=400, detail="Image de sketch invalide")
        
        # Initialisation du statut de traitement
        processing_tasks[task_id] = ProcessingStatus()
        
        # Lancement du traitement en arrière-plan
        background_tasks.add_task(
            process_sketch_background,
            task_id,
            image_array,
            mesh_name
        )
        
        return {
            "task_id": task_id,
            "status": "queued",
            "message": "Traitement démarré"
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation du traitement: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_sketch_background(task_id: str, image_array: np.ndarray, mesh_name: str):
    """
    Traitement en arrière-plan du sketch
    """
    task_status = processing_tasks[task_id]
    
    try:
        # Étape 1: Traitement du sketch
        task_status.status = "processing"
        task_status.progress = 10
        task_status.message = "Analyse du sketch..."
        
        sketch_result = sketch_processor.process_sketch_to_3d(image_array)
        
        if sketch_result['status'] != 'success':
            raise Exception(f"Erreur de traitement: {sketch_result.get('error_message', 'Erreur inconnue')}")
        
        # Étape 2: Génération du maillage
        task_status.progress = 50
        task_status.message = "Génération du maillage 3D..."
        
        depth_map = sketch_result['depth_map'].squeeze()
        sketch_mask = (sketch_result['sketch_tensor'].squeeze() > 0.1).astype(np.float32)
        
        mesh = mesh_generator.generate_mesh_from_depth(
            depth_map,
            sketch_mask,
            sketch_result['classification']
        )
        
        # Validation du maillage
        mesh_validation = validate_mesh_output(mesh)
        if not mesh_validation['is_valid']:
            logger.warning(f"Maillage invalide: {mesh_validation['errors']}")
        
        # Étape 3: Export vers .blend
        task_status.progress = 80
        task_status.message = "Export vers Blender..."
        
        # Création d'un fichier temporaire
        temp_dir = Path(tempfile.gettempdir()) / "sketch3d_results"
        temp_dir.mkdir(exist_ok=True)
        
        blend_path = temp_dir / f"{task_id}.blend"
        
        export_result = blender_exporter.export_mesh_to_blend(
            mesh,
            str(blend_path),
            mesh_name
        )
        
        if not export_result['success']:
            raise Exception(f"Erreur d'export: {export_result.get('error', 'Erreur inconnue')}")
        
        # Finalisation
        task_status.status = "completed"
        task_status.progress = 100
        task_status.message = "Traitement terminé avec succès"
        task_status.result = {
            "blend_file_path": str(blend_path),
            "mesh_stats": mesh_validation['stats'],
            "classification": sketch_result['classification'],
            "export_info": export_result
        }
        
        logger.info(f"Traitement terminé pour la tâche {task_id}")
        
    except Exception as e:
        task_status.status = "failed"
        task_status.error = str(e)
        task_status.message = f"Erreur: {str(e)}"
        logger.error(f"Erreur lors du traitement de la tâche {task_id}: {str(e)}")

@app.get("/api/v1/sketch/{task_id}/status")
async def get_task_status(task_id: str):
    """
    Récupère le statut d'une tâche de traitement
    """
    if task_id not in processing_tasks:
        raise HTTPException(status_code=404, detail="Tâche non trouvée")
    
    task_status = processing_tasks[task_id]
    
    response = {
        "task_id": task_id,
        "status": task_status.status,
        "progress": task_status.progress,
        "message": task_status.message
    }
    
    if task_status.status == "completed":
        response["result"] = task_status.result
    elif task_status.status == "failed":
        response["error"] = task_status.error
    
    return response

@app.get("/api/v1/sketch/{task_id}/download")
async def download_blend_file(task_id: str):
    """
    Télécharge le fichier .blend généré
    """
    if task_id not in processing_tasks:
        raise HTTPException(status_code=404, detail="Tâche non trouvée")
    
    task_status = processing_tasks[task_id]
    
    if task_status.status != "completed":
        raise HTTPException(status_code=400, detail="Traitement non terminé")
    
    blend_file_path = task_status.result["blend_file_path"]
    
    if not Path(blend_file_path).exists():
        raise HTTPException(status_code=404, detail="Fichier .blend non trouvé")
    
    return FileResponse(
        path=blend_file_path,
        filename=f"model_{task_id}.blend",
        media_type="application/octet-stream"
    )

@app.delete("/api/v1/sketch/{task_id}")
async def delete_task(task_id: str):
    """
    Supprime une tâche et nettoie les fichiers associés
    """
    if task_id not in processing_tasks:
        raise HTTPException(status_code=404, detail="Tâche non trouvée")
    
    task_status = processing_tasks[task_id]
    
    # Suppression des fichiers si ils existent
    if task_status.result and "blend_file_path" in task_status.result:
        blend_path = Path(task_status.result["blend_file_path"])
        if blend_path.exists():
            blend_path.unlink()
    
    # Suppression de la tâche de la mémoire
    del processing_tasks[task_id]
    
    return {"message": f"Tâche {task_id} supprimée avec succès"}

@app.get("/api/v1/models/status")
async def get_models_status():
    """
    Retourne le statut des modèles IA
    """
    return {
        "sketch_classifier": {
            "loaded": hasattr(sketch_processor.classifier, 'parameters'),
            "device": str(sketch_processor.device),
            "num_classes": config['models']['classifier']['num_classes']
        },
        "depth_estimator": {
            "loaded": hasattr(sketch_processor.depth_estimator, 'parameters'),
            "device": str(sketch_processor.device),
            "input_size": "512x512"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

# src/config/settings.py
import yaml
from pathlib import Path
from typing import Dict, Any

def get_config() -> Dict[str, Any]:
    """
    Charge la configuration depuis le fichier YAML
    """
    config_path = Path(__file__).parent / "model_config.yaml"
    
    # Configuration par défaut
    default_config = {
        "models": {
            "classifier": {
                "num_classes": 345,
                "backbone": "resnet18",
                "weights_path": "src/models/model_weights/classifier.pth",
                "class_names": _get_default_class_names()
            },
            "depth_estimator": {
                "features": 64,
                "weights_path": "src/models/model_weights/depth_model.pth"
            },
            "min_confidence_threshold": 0.3
        },
        "mesh_generation": {
            "voxel_resolution": 64,
            "smoothing_iterations": 3,
            "min_mesh_faces": 100
        },
        "api": {
            "max_file_size": 10 * 1024 * 1024,  # 10 MB
            "allowed_formats": ["jpg", "jpeg", "png", "bmp"],
            "max_concurrent_tasks": 5
        }
    }
    
    # Chargement du fichier de config si il existe
    if config_path.exists():
        with open(config_path, 'r') as f:
            file_config = yaml.safe_load(f)
        
        # Fusion des configurations
        default_config.update(file_config)
    
    return default_config

def _get_default_class_names():
    """
    Retourne la liste des noms de classes par défaut (QuickDraw categories)
    """
    return [
        "aircraft carrier", "airplane", "alarm clock", "ambulance", "angel", "animal migration",
        "ant", "anvil", "apple", "arm", "asparagus", "axe", "backpack", "banana", "bandage",
        "barn", "baseball", "baseball bat", "basket", "basketball", "bat", "bathtub", "beach",
        "bear", "beard", "bed", "bee", "belt", "bench", "bicycle", "binoculars", "bird",
        "birthday cake", "blackberry", "blueberry", "book", "boomerang", "bottlecap", "bowtie",
        "bracelet", "brain", "bread", "bridge", "broccoli", "broom", "bucket", "bulldozer",
        "bus", "bush", "butterfly", "cactus", "cake", "calculator", "calendar", "camel",
        "camera", "camouflage", "campfire", "candle", "cannon", "canoe", "car", "carrot",
        "castle", "cat", "ceiling fan", "cell phone", "cello", "chair", "chandelier",
        "church", "circle", "clarinet", "clock", "cloud", "coffee cup", "compass", "computer",
        "cookie", "cooler", "couch", "cow", "crab", "crayon", "crocodile", "crown", "cruise ship",
        "cup", "diamond", "dishwasher", "diving board", "dog", "dolphin", "donut", "door",
        "dragon", "dresser", "drill", "drums", "duck", "dumbbell", "ear", "elbow", "elephant",
        "envelope", "eraser", "eye", "eyeglasses", "face", "fan", "feather", "fence", "finger",
        "fire hydrant", "fireplace", "firetruck", "fish", "flamingo", "flashlight", "flip flops",
        "floor lamp", "flower", "flying saucer", "foot", "fork", "frog", "frying pan",
        "garden", "garden hose", "giraffe", "goatee", "golf club", "grapes", "grass", "guitar",
        "hamburger", "hammer", "hand", "harp", "hat", "headphones", "hedgehog", "helicopter",
        "helmet", "hexagon", "hockey puck", "hockey stick", "horse", "hospital", "hot air balloon",
        "hot dog", "hot tub", "hourglass", "house", "house plant", "hurricane", "ice cream",
        "jacket", "jail", "kangaroo", "key", "keyboard", "knee", "knife", "ladder", "lantern",
        "laptop", "leaf", "leg", "light bulb", "lighter", "lighthouse", "lightning", "line",
        "lion", "lipstick", "lobster", "lollipop", "mailbox", "map", "marker", "matches",
        "megaphone", "mermaid", "microphone", "microwave", "monkey", "moon", "mosquito",
        "motorbike", "mountain", "mouse", "moustache", "mouth", "mug", "mushroom", "nail",
        "necklace", "nose", "ocean", "octagon", "octopus", "onion", "oven", "owl", "paintbrush",
        "paint can", "palm tree", "panda", "pants", "paper clip", "parachute", "parrot",
        "passport", "peanut", "pear", "peas", "pencil", "penguin", "piano", "pickup truck",
        "picture frame", "pie", "pig", "pillow", "pineapple", "pizza", "pliers", "police car",
        "pond", "pool", "popsicle", "postcard", "potato", "power outlet", "purse", "rabbit",
        "raccoon", "radio", "rain", "rainbow", "rake", "remote control", "rhinoceros", "rifle",
        "river", "roller coaster", "rollerskates", "sailboat", "sandwich", "saw", "saxophone",
        "school bus", "scissors", "scorpion", "screwdriver", "sea turtle", "see saw", "shark",
        "sheep", "shoe", "shorts", "shovel", "sink", "skateboard", "skull", "skyscraper",
        "sleeping bag", "smiley face", "snail", "snake", "snorkel", "snowflake", "snowman",
        "soccer ball", "sock", "speedboat", "spider", "spoon", "spreadsheet", "square",
        "squiggle", "squirrel", "stairs", "star", "steak", "stereo", "stethoscope", "stitches",
        "stop sign", "stove", "strawberry", "streetlight", "string bean", "submarine", "suitcase",
        "sun", "swan", "sweater", "swing set", "sword", "syringe", "table", "teapot", "teddy-bear",
        "telephone", "television", "tennis racquet", "tent", "The Eiffel Tower", "The Great Wall of China",
        "The Mona Lisa", "tiger", "toaster", "toe", "toilet", "tooth", "toothbrush", "toothpaste",
        "tornado", "tractor", "traffic light", "train", "tree", "triangle", "trombone", "truck",
        "trumpet", "umbrella", "underwear", "van", "vase", "violin", "washing machine", "watermelon",
        "waterslide", "whale", "wheel", "windmill", "wine bottle", "wine glass", "wristwatch",
        "yoga", "zebra", "zigzag"
    ]

# config/model_config.yaml
models:
  classifier:
    num_classes: 345
    backbone: "resnet18"
    weights_path: "src/models/model_weights/classifier.pth"
    batch_size: 32
    learning_rate: 0.001
    epochs: 100
    
  depth_estimator:
    features: 64
    weights_path: "src/models/model_weights/depth_model.pth"
    batch_size: 16
    learning_rate: 0.0001
    epochs: 150
    
  min_confidence_threshold: 0.3

mesh_generation:
  voxel_resolution: 64
  smoothing_iterations: 3
  min_mesh_faces: 100

training:
  data_path: "src/data/datasets/"
  checkpoint_dir: "checkpoints/"
  log_dir: "logs/"
  device: "cuda"
  num_workers: 4
  
datasets:
  quickdraw:
    url: "https://console.cloud.google.com/storage/browser/quickdraw_dataset"
    categories: 345
    samples_per_category: 10000
    
  custom:
    path: "src/data/datasets/custom/"
    train_split: 0.8
    val_split: 0.2

# scripts/train_models.py
#!/usr/bin/env python3
"""
Script d'entraînement des modèles Sketch-to-3D
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from pathlib import Path
import numpy as np
import logging
import argparse
from tqdm import tqdm
import yaml
import cv2
from PIL import Image

# Import des modèles
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.sketch_classifier import SketchClassifier
from src.models.depth_estimator import DepthEstimator
from src.utils.image_processing import normalize_sketch, clean_sketch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuickDrawDataset(Dataset):
    """
    Dataset pour les données QuickDraw
    """
    def __init__(self, data_path: Path, categories: list, transform=None, samples_per_category=1000):
        self.data_path = data_path
        self.categories = categories
        self.transform = transform
        self.samples_per_category = samples_per_category
        
        # Chargement des données
        self.data, self.labels = self._load_data()
        
    def _load_data(self):
        """Charge les données QuickDraw depuis les fichiers .npy"""
        all_data = []
        all_labels = []
        
        for idx, category in enumerate(self.categories):
            category_file = self.data_path / f"{category.replace(' ', '_')}.npy"
            
            if category_file.exists():
                # Chargement des données de cette catégorie
                category_data = np.load(category_file)
                
                # Limitation du nombre d'échantillons
                if len(category_data) > self.samples_per_category:
                    indices = np.random.choice(len(category_data), self.samples_per_category, replace=False)
                    category_data = category_data[indices]
                
                all_data.append(category_data)
                all_labels.extend([idx] * len(category_data))
                
                logger.info(f"Chargé {len(category_data)} échantillons pour {category}")
            else:
                logger.warning(f"Fichier non trouvé: {category_file}")
        
        return np.vstack(all_data), np.array(all_labels)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Les données QuickDraw sont des vecteurs de strokes, conversion en image
        sketch_data = self.data[idx]
        label = self.labels[idx]
        
        # Conversion des strokes en image
        image = self._strokes_to_image(sketch_data)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def _strokes_to_image(self, strokes, image_size=256):
        """Convertit les strokes QuickDraw en image"""
        image = np.zeros((image_size, image_size), dtype=np.uint8)
        
        # Parsing des strokes (format QuickDraw)
        for stroke in strokes:
            if len(stroke) >= 2:
                x_coords = stroke[0]
                y_coords = stroke[1]
                
                # Normalisation des coordonnées
                if len(x_coords) > 1 and len(y_coords) > 1:
                    x_coords = np.array(x_coords) * image_size / 255
                    y_coords = np.array(y_coords) * image_size / 255
                    
                    # Dessin des lignes
                    points = np.column_stack((x_coords, y_coords)).astype(int)
                    for i in range(len(points) - 1):
                        cv2.line(image, tuple(points[i]), tuple(points[i + 1]), 255, 2)
        
        return Image.fromarray(image)

class SketchDepthDataset(Dataset):
    """
    Dataset pour l'entraînement de l'estimateur de profondeur
    """
    def __init__(self, sketch_path: Path, depth_path: Path, transform=None):
        self.sketch_path = sketch_path
        self.depth_path = depth_path
        self.transform = transform
        
        # Liste des fichiers
        self.sketch_files = list(sketch_path.glob("*.png")) + list(sketch_path.glob("*.jpg"))
        self.sketch_files = [f for f in self.sketch_files if (depth_path / f.name).exists()]
        
        logger.info(f"Trouvé {len(self.sketch_files)} paires sketch-depth")
    
    def __len__(self):
        return len(self.sketch_files)
    
    def __getitem__(self, idx):
        sketch_file = self.sketch_files[idx]
        depth_file = self.depth_path / sketch_file.name
        
        # Chargement du sketch
        sketch = Image.open(sketch_file).convert('L')  # Niveaux de gris
        
        # Chargement de la carte de profondeur
        depth = Image.open(depth_file).convert('L')
        
        if self.transform:
            sketch = self.transform(sketch)
            depth = self.transform(depth)
        
        return sketch, depth

def train_classifier(config):
    """
    Entraîne le classificateur de sketches
    """
    logger.info("Début de l'entraînement du classificateur")
    
    # Paramètres
    num_classes = config['models']['classifier']['num_classes']
    batch_size = config['models']['classifier']['batch_size']
    learning_rate = config['models']['classifier']['learning_rate']
    epochs = config['models']['classifier']['epochs']
    
    # Transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Dataset (simulé pour cet exemple)
    # En production, utilisez le vrai dataset QuickDraw
    dataset = QuickDrawDataset(
        data_path=Path(config['training']['data_path']) / 'quickdraw',
        categories=config['models']['classifier']['class_names'][:50],  # Sous-ensemble pour test
        transform=transform,
        samples_per_category=100
    )
    
    # Séparation train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Modèle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SketchClassifier(num_classes=len(dataset.categories), backbone='resnet18')
    model.to(device)
    
    # Optimiseur et loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # Boucle d'entraînement
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        # Phase d'entraînement
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * train_correct / train_total:.2f}%'
            })
        
        # Phase de validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calcul des métriques
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        logger.info(f'Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
        
        # Sauvegarde du meilleur modèle
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = Path(config['training']['checkpoint_dir']) / 'classifier_best.pth'
            checkpoint_path.parent.mkdir(exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, checkpoint_path)
            
            logger.info(f'Nouveau meilleur modèle sauvé: {val_acc:.2f}%')
        
        scheduler.step()
    
    logger.info(f'Entraînement terminé. Meilleure précision: {best_val_acc:.2f}%')

def train_depth_estimator(config):
    """
    Entraîne l'estimateur de profondeur
    """
    logger.info("Début de l'entraînement de l'estimateur de profondeur")
    
    # Paramètres
    batch_size = config['models']['depth_estimator']['batch_size']
    learning_rate = config['models']['depth_estimator']['learning_rate']
    epochs = config['models']['depth_estimator']['epochs']
    
    # Transformations
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    
    # Pour cet exemple, on simule un dataset
    # En production, utilisez des vrais pairs sketch-depth
    logger.warning("Dataset depth simulé - implémentez votre dataset réel")
    
    # Modèle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DepthEstimator(input_channels=1, output_channels=1, features=64)
    model.to(device)
    
    # Optimiseur et loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()  # Pour la régression de profondeur
    
    # Simulation d'entraînement (remplacez par votre vraie boucle)
    logger.info("Simulation d'entraînement terminée")
    
    # Sauvegarde
    checkpoint_path = Path(config['training']['checkpoint_dir']) / 'depth_estimator_best.pth'
    checkpoint_path.parent.mkdir(exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)

def main():
    parser = argparse.ArgumentParser(description='Entraînement des modèles Sketch-to-3D')
    parser.add_argument('--model', choices=['classifier', 'depth', 'all'], 
                       default='all', help='Modèle à entraîner')
    parser.add_argument('--config', type=str, default='config/model_config.yaml',
                       help='Chemin vers le fichier de configuration')
    
    args = parser.parse_args()
    
    # Chargement de la configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Création des dossiers nécessaires
    Path(config['training']['checkpoint_dir']).mkdir(exist_ok=True)
    Path(config['training']['log_dir']).mkdir(exist_ok=True)
    
    # Entraînement
    if args.model in ['classifier', 'all']:
        train_classifier(config)
    
    if args.model in ['depth', 'all']:
        train_depth_estimator(config)
    
    logger.info("Entraînement terminé !")

if __name__ == "__main__":
    main()