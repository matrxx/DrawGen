# mesh_generator.py
"""Générateur de maillages 3D à partir de cartes de profondeur"""

import numpy as np
import trimesh
from skimage import measure
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class MeshGenerator:
    """Générateur de maillages 3D à partir de cartes de profondeur et sketches"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.voxel_resolution = config.get('voxel_resolution', 64)
        self.smoothing_iterations = config.get('smoothing_iterations', 3)
        self.min_mesh_faces = config.get('min_mesh_faces', 100)
        
    def generate_mesh_from_depth(self, depth_map: np.ndarray, sketch_mask: np.ndarray, 
                                class_info: Dict[str, Any]) -> trimesh.Trimesh:
        """Méthode principale pour générer un maillage"""
        logger.info("Génération du maillage 3D...")
        
        try:
            # Méthode principale: Marching Cubes
            mesh = self._voxel_to_mesh_marching_cubes(depth_map, sketch_mask)
            
        except Exception as e:
            logger.warning(f"Marching Cubes échoué: {e}")
            # Fallback: génération basique selon la classe
            mesh = self._generate_fallback_mesh(class_info)
        
        # Post-traitement
        mesh = self._clean_mesh(mesh)
        mesh = self._apply_class_specific_processing(mesh, class_info)
        
        logger.info(f"Maillage généré avec {len(mesh.vertices)} vertices et {len(mesh.faces)} faces")
        return mesh
    
    def _voxel_to_mesh_marching_cubes(self, depth_map: np.ndarray, sketch_mask: np.ndarray) -> trimesh.Trimesh:
        """Génère un maillage via l'algorithme Marching Cubes"""
        # Création d'un volume 3D
        volume = self._create_3d_volume(depth_map, sketch_mask)
        
        try:
            vertices, faces, normals, values = measure.marching_cubes(
                volume, 
                level=0.5,
                spacing=(1.0, 1.0, 1.0)
            )
        except ValueError:
            # Fallback avec volume simplifié
            volume = self._create_simplified_volume(depth_map, sketch_mask)
            vertices, faces, normals, values = measure.marching_cubes(
                volume,
                level=0.3,
                spacing=(1.0, 1.0, 1.0)
            )
        
        # Normalisation des vertices
        vertices = self._normalize_vertices(vertices)
        
        # Création du mesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
        return mesh
    
    def _create_3d_volume(self, depth_map: np.ndarray, sketch_mask: np.ndarray) -> np.ndarray:
        """Crée un volume 3D à partir d'une carte de profondeur"""
        height, width = depth_map.shape
        depth_layers = 32
        
        volume = np.zeros((height, width, depth_layers), dtype=np.float32)
        
        for i in range(height):
            for j in range(width):
                if sketch_mask[i, j] > 0.1:
                    depth_val = depth_map[i, j]
                    depth_layer = int(depth_val * (depth_layers - 1))
                    volume[i, j, :depth_layer+1] = 1.0
        
        return volume
    
    def _create_simplified_volume(self, depth_map: np.ndarray, sketch_mask: np.ndarray) -> np.ndarray:
        """Version simplifiée du volume pour fallback"""
        small_depth = depth_map[::2, ::2]
        small_mask = sketch_mask[::2, ::2]
        
        height, width = small_depth.shape
        depth_layers = 16
        
        volume = np.zeros((height, width, depth_layers), dtype=np.float32)
        
        mask_indices = np.where(small_mask > 0.1)
        for i, j in zip(mask_indices[0], mask_indices[1]):
            depth_val = small_depth[i, j]
            depth_layer = min(int(depth_val * depth_layers), depth_layers - 1)
            volume[i, j, :depth_layer+1] = 1.0
        
        return volume
    
    def _normalize_vertices(self, vertices: np.ndarray) -> np.ndarray:
        """Normalise les vertices du maillage"""
        center = np.mean(vertices, axis=0)
        vertices_centered = vertices - center
        
        max_extent = np.max(np.abs(vertices_centered))
        if max_extent > 0:
            vertices_normalized = vertices_centered / max_extent
        else:
            vertices_normalized = vertices_centered
        
        return vertices_normalized
    
    def _generate_fallback_mesh(self, class_info: Dict[str, Any]) -> trimesh.Trimesh:
        """Génère un maillage de base selon la classe d'objet"""
        class_name = class_info.get('class_name', 'unknown')
        
        if class_name in ['cube', 'box', 'square']:
            mesh = trimesh.creation.box(extents=[2, 2, 2])
        elif class_name in ['sphere', 'circle', 'ball']:
            mesh = trimesh.creation.icosphere(radius=1, subdivisions=2)
        elif class_name in ['cylinder', 'tube']:
            mesh = trimesh.creation.cylinder(radius=1, height=2)
        elif class_name in ['cone', 'triangle']:
            mesh = trimesh.creation.cone(radius=1, height=2)
        else:
            mesh = trimesh.creation.box(extents=[1.5, 1.5, 1.5])
        
        logger.info(f"Mesh de fallback généré pour: {class_name}")
        return mesh
    
    def _clean_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Nettoie et optimise le maillage"""
        if len(mesh.faces) < self.min_mesh_faces:
            logger.warning(f"Maillage avec seulement {len(mesh.faces)} faces")
        
        # Suppression des composants déconnectés
        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            largest_component = max(components, key=lambda x: len(x.faces))
            mesh = largest_component
        
        # Réparation basique
        mesh.remove_duplicate_faces()
        mesh.remove_degenerate_faces()
        mesh.remove_unreferenced_vertices()
        
        return mesh
    
    def _apply_class_specific_processing(self, mesh: trimesh.Trimesh, 
                                       class_info: Dict[str, Any]) -> trimesh.Trimesh:
        """Applique des traitements spécifiques selon la classe"""
        confidence = class_info.get('confidence', 0.5)
        
        # Ajustement de l'échelle selon la confiance
        if confidence < 0.5:
            mesh.apply_scale(0.9)
        
        return mesh