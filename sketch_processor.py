# sketch_processor.py
"""Processeur principal pour convertir les sketches en modèles 3D"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import logging
from typing import Tuple, Dict, Any

from models import SketchClassifier, DepthEstimator
from utils import normalize_sketch, clean_sketch, validate_sketch_input
from config import DEVICE, MODEL_CONFIG

logger = logging.getLogger(__name__)

class SketchProcessor:
    """Pipeline: Sketch → Classification → Depth → 3D Mesh"""
    
    def __init__(self, model_config: Dict[str, Any]):
        self.config = model_config
        self.device = DEVICE
        
        # Chargement des modèles
        self.classifier = self._load_classifier()
        self.depth_estimator = self._load_depth_estimator()
        
        logger.info(f"SketchProcessor initialisé sur {self.device}")
    
    def _load_classifier(self) -> SketchClassifier:
        """Charge le modèle de classification"""
        model = SketchClassifier(
            num_classes=self.config['classifier']['num_classes'],
            backbone=self.config['classifier']['backbone']
        )
        model.to(self.device)
        model.eval()
        return model
    
    def _load_depth_estimator(self) -> DepthEstimator:
        """Charge le modèle d'estimation de profondeur"""
        model = DepthEstimator(
            input_channels=1,
            output_channels=1,
            features=self.config['depth_estimator']['features']
        )
        model.to(self.device)
        model.eval()
        return model
    
    def preprocess_sketch(self, sketch_image: np.ndarray) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Préprocessing du sketch d'entrée"""
        if not validate_sketch_input(sketch_image):
            raise ValueError("Sketch d'entrée invalide")
        
        # Conversion en niveaux de gris si nécessaire
        if len(sketch_image.shape) == 3:
            sketch_gray = cv2.cvtColor(sketch_image, cv2.COLOR_RGB2GRAY)
        else:
            sketch_gray = sketch_image.copy()
        
        # Nettoyage du sketch
        cleaned_sketch = clean_sketch(sketch_gray)
        
        # Normalisation et redimensionnement
        normalized_sketch = normalize_sketch(cleaned_sketch, target_size=512)
        
        # Conversion en tensor PyTorch
        sketch_tensor = torch.from_numpy(normalized_sketch).float()
        sketch_tensor = sketch_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        sketch_tensor = sketch_tensor.to(self.device)
        
        # Métadonnées
        metadata = {
            'original_shape': sketch_image.shape,
            'preprocessed_shape': normalized_sketch.shape,
            'has_content': np.sum(normalized_sketch > 0.1) > 100
        }
        
        return sketch_tensor, metadata
    
    def classify_sketch(self, sketch_tensor: torch.Tensor) -> Dict[str, Any]:
        """Classification du type de sketch"""
        with torch.no_grad():
            logits = self.classifier(sketch_tensor)
            probabilities = F.softmax(logits, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            
            class_names = self.config['classifier']['class_names']
            
            return {
                'class_id': predicted_class.item(),
                'confidence': confidence.item(),
                'class_name': class_names[predicted_class.item()],
                'probabilities': probabilities.cpu().numpy().flatten()
            }
    
    def estimate_depth(self, sketch_tensor: torch.Tensor, class_info: Dict[str, Any]) -> torch.Tensor:
        """Estimation de la carte de profondeur"""
        with torch.no_grad():
            depth_map = self.depth_estimator(sketch_tensor)
            
            # Post-traitement spécifique à la classe
            depth_map = self._postprocess_depth(depth_map, class_info)
            
            return depth_map
    
    def _postprocess_depth(self, depth_map: torch.Tensor, class_info: Dict[str, Any]) -> torch.Tensor:
        """Post-traitement de la carte de profondeur selon la classe"""
        class_name = class_info['class_name']
        
        # Normalisation de base
        depth_map = torch.clamp(depth_map, 0, 1)
        
        # Ajustements spécifiques par classe
        if class_name in ['cube', 'box']:
            depth_map = torch.where(depth_map > 0.1, 
                                  torch.clamp(depth_map * 1.2, 0, 1), 
                                  depth_map)
        elif class_name in ['sphere', 'circle']:
            depth_map = F.gaussian_blur(depth_map, kernel_size=5, sigma=1.0)
        
        return depth_map
    
    def process_sketch_to_3d(self, sketch_image: np.ndarray) -> Dict[str, Any]:
        """Pipeline complet de conversion sketch vers 3D"""
        logger.info("Début du traitement sketch-to-3D")
        
        try:
            # Étape 1: Préprocessing
            sketch_tensor, preprocessing_meta = self.preprocess_sketch(sketch_image)
            
            if not preprocessing_meta['has_content']:
                raise ValueError("Le sketch ne contient pas suffisamment de contenu")
            
            # Étape 2: Classification
            classification_result = self.classify_sketch(sketch_tensor)
            
            if classification_result['confidence'] < self.config['min_confidence_threshold']:
                logger.warning(f"Confiance faible: {classification_result['confidence']:.2f}")
            
            # Étape 3: Estimation de profondeur
            depth_map = self.estimate_depth(sketch_tensor, classification_result)
            
            # Compilation des résultats
            results = {
                'status': 'success',
                'preprocessing': preprocessing_meta,
                'classification': classification_result,
                'depth_map': depth_map.cpu().numpy(),
                'sketch_tensor': sketch_tensor.cpu().numpy(),
                'processing_device': str(self.device)
            }
            
            logger.info("Traitement terminé avec succès")
            return results
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement: {str(e)}")
            return {
                'status': 'error',
                'error_message': str(e),
                'error_type': type(e).__name__
            }