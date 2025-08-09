# sketch_processor.py
"""Main processorto convert sketches to 3D Models"""

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
        
        self.classifier = self._load_classifier()
        self.depth_estimator = self._load_depth_estimator()
        
        logger.info(f"SketchProcessor initialisé sur {self.device}")
    
    def _load_classifier(self) -> SketchClassifier:
        """Load the classification model"""
        model = SketchClassifier(
            num_classes=self.config['classifier']['num_classes'],
            backbone=self.config['classifier']['backbone']
        )
        model.to(self.device)
        model.eval()
        return model
    
    def _load_depth_estimator(self) -> DepthEstimator:
        """Load the estimation depth model"""
        model = DepthEstimator(
            input_channels=1,
            output_channels=1,
            features=self.config['depth_estimator']['features']
        )
        model.to(self.device)
        model.eval()
        return model
    
    def preprocess_sketch(self, sketch_image: np.ndarray) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Preprocessing sketches entries"""
        if not validate_sketch_input(sketch_image):
            raise ValueError("Invalid Sketch entry")
        
        if len(sketch_image.shape) == 3:
            sketch_gray = cv2.cvtColor(sketch_image, cv2.COLOR_RGB2GRAY)
        else:
            sketch_gray = sketch_image.copy()
        
        cleaned_sketch = clean_sketch(sketch_gray)
        
        normalized_sketch = normalize_sketch(cleaned_sketch, target_size=512)
        
        sketch_tensor = torch.from_numpy(normalized_sketch).float()
        sketch_tensor = sketch_tensor.unsqueeze(0).unsqueeze(0)
        sketch_tensor = sketch_tensor.to(self.device)
        
        metadata = {
            'original_shape': sketch_image.shape,
            'preprocessed_shape': normalized_sketch.shape,
            'has_content': np.sum(normalized_sketch > 0.1) > 100
        }
        
        return sketch_tensor, metadata
    
    def classify_sketch(self, sketch_tensor: torch.Tensor) -> Dict[str, Any]:
        """Sketch type classification"""
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
        """Depth map estimation"""
        with torch.no_grad():
            depth_map = self.depth_estimator(sketch_tensor)
            
            depth_map = self._postprocess_depth(depth_map, class_info)
            
            return depth_map
    
    def _postprocess_depth(self, depth_map: torch.Tensor, class_info: Dict[str, Any]) -> torch.Tensor:
        """Post-treatment of the map depending on class"""
        class_name = class_info['class_name']
        
        depth_map = torch.clamp(depth_map, 0, 1)
        
        if class_name in ['cube', 'box']:
            depth_map = torch.where(depth_map > 0.1, 
                                  torch.clamp(depth_map * 1.2, 0, 1), 
                                  depth_map)
        elif class_name in ['sphere', 'circle']:
            depth_map = F.gaussian_blur(depth_map, kernel_size=5, sigma=1.0)
        
        return depth_map
    
    def process_sketch_to_3d(self, sketch_image: np.ndarray) -> Dict[str, Any]:
        """Full Pipeline conversion sketchs to 3D"""
        logger.info("Start of the processing DrawGen")
        
        try:
            sketch_tensor, preprocessing_meta = self.preprocess_sketch(sketch_image)
            
            if not preprocessing_meta['has_content']:
                raise ValueError("The sketch has not enough content")
            
            classification_result = self.classify_sketch(sketch_tensor)
            
            if classification_result['confidence'] < self.config['min_confidence_threshold']:
                logger.warning(f"Low confidence: {classification_result['confidence']:.2f}")
            
            depth_map = self.estimate_depth(sketch_tensor, classification_result)
            
            results = {
                'status': 'success',
                'preprocessing': preprocessing_meta,
                'classification': classification_result,
                'depth_map': depth_map.cpu().numpy(),
                'sketch_tensor': sketch_tensor.cpu().numpy(),
                'processing_device': str(self.device)
            }
            
            logger.info("Treatment ended successfully")
            return results
            
        except Exception as e:
            logger.error(f"Erreur during treatment: {str(e)}")
            return {
                'status': 'error',
                'error_message': str(e),
                'error_type': type(e).__name__

            }
