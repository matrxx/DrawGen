# utils.py
"""Utilitaires pour le traitement d'images et validation"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

def validate_sketch_input(image: np.ndarray) -> bool:
    """Valide qu'une image peut être utilisée comme sketch"""
    if image is None or image.size == 0:
        return False
    
    if len(image.shape) < 2 or len(image.shape) > 3:
        return False
    
    h, w = image.shape[:2]
    if h < 32 or w < 32 or h > 4096 or w > 4096:
        return False
    
    # Vérification du contenu
    if len(image.shape) == 2:
        content_ratio = np.sum(image > 10) / image.size
    else:
        gray = np.mean(image, axis=2)
        content_ratio = np.sum(gray > 10) / gray.size
    
    if content_ratio < 0.01:
        return False
    
    return True

def normalize_sketch(image: np.ndarray, target_size: int = 512) -> np.ndarray:
    """Normalise un sketch: redimensionnement, padding, normalisation"""
    h, w = image.shape
    max_dim = max(h, w)
    scale = target_size / max_dim
    
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Padding pour obtenir un carré
    pad_h = (target_size - new_h) // 2
    pad_w = (target_size - new_w) // 2
    
    padded = np.pad(resized, 
                   ((pad_h, target_size - new_h - pad_h), 
                    (pad_w, target_size - new_w - pad_w)), 
                   mode='constant', 
                   constant_values=0)
    
    # Normalisation entre 0 et 1
    normalized = padded.astype(np.float32) / 255.0
    return normalized

def clean_sketch(image: np.ndarray) -> np.ndarray:
    """Nettoie un sketch: débruitage, amélioration du contraste"""
    # Débruitage
    denoised = cv2.bilateralFilter(image, 9, 75, 75)
    
    # Amélioration du contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    # Seuillage adaptatif
    binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Inversion pour avoir lignes en blanc
    inverted = cv2.bitwise_not(binary)
    
    # Morphologie pour nettoyer
    kernel = np.ones((2,2), np.uint8)
    cleaned = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    return cleaned

def validate_mesh_output(mesh) -> dict:
    """Valide un maillage 3D généré"""
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    try:
        num_vertices = len(mesh.vertices)
        num_faces = len(mesh.faces)
        
        validation_results['stats'] = {
            'num_vertices': num_vertices,
            'num_faces': num_faces,
            'is_watertight': mesh.is_watertight,
            'is_valid': mesh.is_valid,
            'volume': float(mesh.volume) if mesh.is_watertight else None
        }
        
        if num_vertices < 4:
            validation_results['errors'].append("Trop peu de vertices")
            validation_results['is_valid'] = False
        
        if num_faces < 4:
            validation_results['errors'].append("Trop peu de faces")
            validation_results['is_valid'] = False
        
        if not mesh.is_valid:
            validation_results['warnings'].append("Maillage marqué comme invalide")
        
    except Exception as e:
        validation_results['is_valid'] = False
        validation_results['errors'].append(f"Erreur validation: {str(e)}")
    
    return validation_results