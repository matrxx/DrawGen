# config.py
"""Configuration centralisée pour drawgen"""

import torch

# Configuration générale
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configuration des modèles
MODEL_CONFIG = {
    'classifier': {
        'num_classes': 50,
        'backbone': 'resnet18',
        'min_confidence': 0.3,
        'class_names': [
            'cube', 'sphere', 'cylinder', 'cone', 'house', 'car', 'tree', 'flower',
            'cat', 'dog', 'bird', 'fish', 'airplane', 'boat', 'chair', 'table',
            'cup', 'bottle', 'phone', 'computer', 'book', 'clock', 'lamp', 'door',
            'window', 'star', 'sun', 'moon', 'cloud', 'mountain', 'bridge', 'tower',
            'castle', 'robot', 'heart', 'diamond', 'triangle', 'circle', 'square',
            'line', 'arrow', 'key', 'shoe', 'hat', 'glasses', 'umbrella', 'sword',
            'guitar', 'piano', 'camera', 'bicycle'
        ]
    },
    'depth_estimator': {
        'features': 64,
        'input_size': 512
    },
    'min_confidence_threshold': 0.3
}

# Configuration génération 3D
MESH_CONFIG = {
    'voxel_resolution': 64,
    'min_mesh_faces': 100,
    'smoothing_iterations': 2
}

# Configuration API
API_CONFIG = {
    'host': '127.0.0.1',
    'port': 8000,
    'max_file_size': 10 * 1024 * 1024  # 10MB
}

def get_config():
    """Retourne la configuration complète"""
    return {
        'models': MODEL_CONFIG,
        'mesh_generation': MESH_CONFIG,
        'api': API_CONFIG
    }