# mesh_generator.py
"""3D Mesh Generator"""

import numpy as np
import trimesh
from skimage import measure
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class MeshGenerator:
    """3D Mesh Generator For Deepeness"""
    
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
            mesh = self._voxel_to_mesh_marching_cubes(depth_map, sketch_mask)
            
        except Exception as e:
            logger.warning(f"Marching Cubes échoué: {e}")
            mesh = self._generate_fallback_mesh(class_info)
        
        mesh = self._clean_mesh(mesh)
        mesh = self._apply_class_specific_processing(mesh, class_info)
        
        logger.info(f"Maillage généré avec {len(mesh.vertices)} vertices et {len(mesh.faces)} faces")
        return mesh
    
    def _voxel_to_mesh_marching_cubes(self, depth_map: np.ndarray, sketch_mask: np.ndarray) -> trimesh.Trimesh:
        """Generate Meshs Trough Marching Cubes"""
        volume = self._create_3d_volume(depth_map, sketch_mask)
        
        try:
            vertices, faces, normals, values = measure.marching_cubes(
                volume, 
                level=0.5,
                spacing=(1.0, 1.0, 1.0)
            )
        except ValueError:
            volume = self._create_simplified_volume(depth_map, sketch_mask)
            vertices, faces, normals, values = measure.marching_cubes(
                volume,
                level=0.3,
                spacing=(1.0, 1.0, 1.0)
            )
        
        vertices = self._normalize_vertices(vertices)
        
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
        return mesh
    
    def _create_3d_volume(self, depth_map: np.ndarray, sketch_mask: np.ndarray) -> np.ndarray:
        """3D Volume Through Depth Map"""
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
        """Simplified Fallback Version"""
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
        """Standard Vertice"""
        center = np.mean(vertices, axis=0)
        vertices_centered = vertices - center
        
        max_extent = np.max(np.abs(vertices_centered))
        if max_extent > 0:
            vertices_normalized = vertices_centered / max_extent
        else:
            vertices_normalized = vertices_centered
        
        return vertices_normalized
    
    def _generate_fallback_mesh(self, class_info: Dict[str, Any]) -> trimesh.Trimesh:
        """Generate meshs through object class"""
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
            logger.warning(f"Meshs with only {len(mesh.faces)} faces")
        
        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            largest_component = max(components, key=lambda x: len(x.faces))
            mesh = largest_component
        
        mesh.remove_duplicate_faces()
        mesh.remove_degenerate_faces()
        mesh.remove_unreferenced_vertices()
        
        return mesh
    
    def _apply_class_specific_processing(self, mesh: trimesh.Trimesh, 
                                       class_info: Dict[str, Any]) -> trimesh.Trimesh:
        """Applys specific class treatment"""
        confidence = class_info.get('confidence', 0.5)
        
        if confidence < 0.5:
            mesh.apply_scale(0.9)
        

        return mesh
