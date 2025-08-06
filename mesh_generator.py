# src/core/mesh_generator.py
import numpy as np
import torch
import open3d as o3d
import trimesh
from skimage import measure
from typing import Tuple, Optional, Dict, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class MeshGenerator:
    """
    Générateur de maillages 3D à partir de cartes de profondeur et sketches
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.voxel_resolution = config.get('voxel_resolution', 64)
        self.smoothing_iterations = config.get('smoothing_iterations', 3)
        self.min_mesh_faces = config.get('min_mesh_faces', 100)
        
    def depth_to_pointcloud(self, depth_map: np.ndarray, sketch_mask: np.ndarray) -> o3d.geometry.PointCloud:
        """
        Convertit une carte de profondeur en nuage de points 3D
        
        Args:
            depth_map: Carte de profondeur (H, W)
            sketch_mask: Masque binaire du sketch (H, W)
            
        Returns:
            Nuage de points Open3D
        """
        height, width = depth_map.shape
        
        # Création des coordonnées de grille
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        
        # Normalisation des coordonnées
        x_norm = (xx - width / 2) / (width / 2)
        y_norm = (yy - height / 2) / (height / 2)
        
        # Application du masque pour ne garder que les pixels du sketch
        valid_mask = (sketch_mask > 0.1) & (depth_map > 0.01)
        
        if np.sum(valid_mask) < 10:
            raise ValueError("Pas assez de points valides pour générer un maillage")
        
        # Extraction des points 3D
        x_3d = x_norm[valid_mask]
        y_3d = y_norm[valid_mask]
        z_3d = depth_map[valid_mask]
        
        # Création du nuage de points
        points_3d = np.column_stack([x_3d, y_3d, z_3d])
        
        # Création de l'objet Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d)
        
        # Estimation des normales
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        
        return pcd
    
    def pointcloud_to_voxel_grid(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.VoxelGrid:
        """
        Convertit un nuage de points en grille de voxels
        """
        # Calcul de la taille de voxel basée sur la résolution souhaitée
        bbox = pcd.get_axis_aligned_bounding_box()
        bbox_size = bbox.get_max_bound() - bbox.get_min_bound()
        voxel_size = max(bbox_size) / self.voxel_resolution
        
        # Création de la grille de voxels
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
        
        return voxel_grid
    
    def voxel_to_mesh_marching_cubes(self, depth_map: np.ndarray, sketch_mask: np.ndarray) -> trimesh.Trimesh:
        """
        Génère un maillage via l'algorithme Marching Cubes
        """
        # Création d'un volume 3D à partir de la carte de profondeur
        volume = self._create_3d_volume(depth_map, sketch_mask)
        
        # Application de Marching Cubes
        try:
            vertices, faces, normals, values = measure.marching_cubes(
                volume, 
                level=0.5,
                spacing=(1.0, 1.0, 1.0)
            )
        except ValueError as e:
            logger.error(f"Erreur Marching Cubes: {e}")
            # Fallback avec un volume simplifié
            volume_simplified = self._create_simplified_volume(depth_map, sketch_mask)
            vertices, faces, normals, values = measure.marching_cubes(
                volume_simplified,
                level=0.3,
                spacing=(1.0, 1.0, 1.0)
            )
        
        # Normalisation des vertices
        vertices = self._normalize_vertices(vertices)
        
        # Création du mesh Trimesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
        
        # Validation et nettoyage du maillage
        mesh = self._clean_mesh(mesh)
        
        return mesh
    
    def _create_3d_volume(self, depth_map: np.ndarray, sketch_mask: np.ndarray) -> np.ndarray:
        """
        Crée un volume 3D à partir d'une carte de profondeur
        """
        height, width = depth_map.shape
        depth_layers = 32  # Nombre de couches en profondeur
        
        # Initialisation du volume
        volume = np.zeros((height, width, depth_layers), dtype=np.float32)
        
        # Remplissage du volume basé sur la profondeur
        for i in range(height):
            for j in range(width):
                if sketch_mask[i, j] > 0.1:
                    # Profondeur normalisée entre 0 et 1
                    depth_val = depth_map[i, j]
                    # Conversion en index de couche
                    depth_layer = int(depth_val * (depth_layers - 1))
                    
                    # Remplissage des couches jusqu'à la profondeur
                    volume[i, j, :depth_layer+1] = 1.0
                    
                    # Dégradé pour un meilleur lissage
                    if depth_layer < depth_layers - 1:
                        fade_layers = min(3, depth_layers - depth_layer - 1)
                        for k in range(1, fade_layers + 1):
                            if depth_layer + k < depth_layers:
                                volume[i, j, depth_layer + k] = max(0, 1.0 - k * 0.3)
        
        return volume
    
    def _create_simplified_volume(self, depth_map: np.ndarray, sketch_mask: np.ndarray) -> np.ndarray:
        """
        Version simplifiée du volume pour fallback
        """
        # Réduction de résolution pour éviter les erreurs
        small_depth = depth_map[::2, ::2]
        small_mask = sketch_mask[::2, ::2]
        
        height, width = small_depth.shape
        depth_layers = 16
        
        volume = np.zeros((height, width, depth_layers), dtype=np.float32)
        
        # Version plus simple du remplissage
        mask_indices = np.where(small_mask > 0.1)
        for i, j in zip(mask_indices[0], mask_indices[1]):
            depth_val = small_depth[i, j]
            depth_layer = min(int(depth_val * depth_layers), depth_layers - 1)
            volume[i, j, :depth_layer+1] = 1.0
        
        return volume
    
    def _normalize_vertices(self, vertices: np.ndarray) -> np.ndarray:
        """
        Normalise les vertices du maillage
        """
        # Centrage
        center = np.mean(vertices, axis=0)
        vertices_centered = vertices - center
        
        # Normalisation de l'échelle
        max_extent = np.max(np.abs(vertices_centered))
        if max_extent > 0:
            vertices_normalized = vertices_centered / max_extent
        else:
            vertices_normalized = vertices_centered
        
        return vertices_normalized
    
    def _clean_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Nettoie et optimise le maillage
        """
        if len(mesh.faces) < self.min_mesh_faces:
            logger.warning(f"Maillage avec seulement {len(mesh.faces)} faces")
        
        # Suppression des composants déconnectés petits
        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            # Garder le plus grand composant
            largest_component = max(components, key=lambda x: len(x.faces))
            mesh = largest_component
            logger.info(f"Suppression de {len(components) - 1} petits composants")
        
        # Lissage du maillage
        if hasattr(mesh, 'smoothed'):
            try:
                mesh = mesh.smoothed()
            except:
                logger.warning("Échec du lissage du maillage")
        
        # Réparation basique
        mesh.remove_duplicate_faces()
        mesh.remove_degenerate_faces()
        mesh.remove_unreferenced_vertices()
        
        # Vérification de la validité
        if not mesh.is_valid:
            logger.warning("Maillage généré invalide, tentative de réparation")
            mesh.fix_normals()
        
        return mesh
    
    def generate_mesh_from_depth(self, depth_map: np.ndarray, sketch_mask: np.ndarray, 
                                class_info: Dict[str, Any]) -> trimesh.Trimesh:
        """
        Méthode principale pour générer un maillage à partir d'une carte de profondeur
        
        Args:
            depth_map: Carte de profondeur (H, W)
            sketch_mask: Masque du sketch (H, W)
            class_info: Informations sur la classe d'objet
            
        Returns:
            Maillage 3D
        """
        logger.info("Génération du maillage 3D...")
        
        try:
            # Méthode principale: Marching Cubes
            mesh = self.voxel_to_mesh_marching_cubes(depth_map, sketch_mask)
            
        except Exception as e:
            logger.warning(f"Marching Cubes échoué: {e}, tentative avec nuage de points")
            
            try:
                # Méthode de fallback: Poisson Surface Reconstruction
                pcd = self.depth_to_pointcloud(depth_map, sketch_mask)
                mesh = self._poisson_reconstruction(pcd)
                
            except Exception as e2:
                logger.error(f"Toutes les méthodes de génération ont échoué: {e2}")
                # Génération d'un mesh de base basé sur la classe
                mesh = self._generate_fallback_mesh(class_info)
        
        # Post-traitement spécifique à la classe
        mesh = self._apply_class_specific_processing(mesh, class_info)
        
        logger.info(f"Maillage généré avec {len(mesh.vertices)} vertices et {len(mesh.faces)} faces")
        
        return mesh
    
    def _poisson_reconstruction(self, pcd: o3d.geometry.PointCloud) -> trimesh.Trimesh:
        """
        Reconstruction de surface de Poisson
        """
        # Poisson surface reconstruction
        mesh_o3d, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, 
            depth=9,
            width=0,
            scale=1.1,
            linear_fit=False
        )
        
        # Conversion vers Trimesh
        vertices = np.asarray(mesh_o3d.vertices)
        faces = np.asarray(mesh_o3d.triangles)
        
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        return mesh
    
    def _generate_fallback_mesh(self, class_info: Dict[str, Any]) -> trimesh.Trimesh:
        """
        Génère un maillage de base selon la classe d'objet
        """
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
            # Mesh par défaut
            mesh = trimesh.creation.box(extents=[1.5, 1.5, 1.5])
        
        logger.info(f"Mesh de fallback généré pour la classe: {class_name}")
        
        return mesh
    
    def _apply_class_specific_processing(self, mesh: trimesh.Trimesh, 
                                       class_info: Dict[str, Any]) -> trimesh.Trimesh:
        """
        Applique des traitements spécifiques selon la classe d'objet
        """
        class_name = class_info.get('class_name', 'unknown')
        confidence = class_info.get('confidence', 0.5)
        
        # Ajustements basés sur la classe
        if class_name in ['sphere', 'circle'] and confidence > 0.7:
            # Lissage supplémentaire pour les objets sphériques
            if hasattr(mesh, 'smoothed'):
                try:
                    mesh = mesh.smoothed()
                except:
                    pass
        
        elif class_name in ['cube', 'box'] and confidence > 0.7:
            # Préservation des arêtes pour les objets cubiques
            pass  # Pas de lissage supplémentaire
        
        # Ajustement de l'échelle selon la confiance
        if confidence < 0.5:
            # Réduction légère pour les classifications incertaines
            mesh.apply_scale(0.9)
        
        return mesh

# src/utils/image_processing.py
import cv2
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def normalize_sketch(image: np.ndarray, target_size: int = 512) -> np.ndarray:
    """
    Normalise un sketch: redimensionnement, padding, normalisation des valeurs
    
    Args:
        image: Image d'entrée (H, W)
        target_size: Taille cible (carré)
        
    Returns:
        Image normalisée (target_size, target_size)
    """
    # Redimensionnement avec préservation du ratio
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
    
    # Normalisation des valeurs entre 0 et 1
    normalized = padded.astype(np.float32) / 255.0
    
    return normalized

def clean_sketch(image: np.ndarray) -> np.ndarray:
    """
    Nettoie un sketch: débruitage, amélioration du contraste
    """
    # Débruitage avec filtre bilatéral
    denoised = cv2.bilateralFilter(image, 9, 75, 75)
    
    # Amélioration du contraste avec CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    # Seuillage adaptatif pour binariser
    binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Inversion pour avoir les lignes en blanc sur fond noir
    inverted = cv2.bitwise_not(binary)
    
    # Morphologie pour nettoyer les artefacts
    kernel = np.ones((2,2), np.uint8)
    cleaned = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    return cleaned

def extract_sketch_contours(image: np.ndarray) -> Tuple[np.ndarray, list]:
    """
    Extrait les contours principaux d'un sketch
    
    Returns:
        Image des contours et liste des contours
    """
    # Détection des contours
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrage des contours trop petits
    min_area = 50
    filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    # Création de l'image des contours
    contour_image = np.zeros_like(image)
    cv2.drawContours(contour_image, filtered_contours, -1, (255), thickness=2)
    
    return contour_image, filtered_contours

# src/utils/validation.py
import numpy as np
from typing import Union, Dict, Any
import logging

logger = logging.getLogger(__name__)

def validate_sketch_input(image: np.ndarray) -> bool:
    """
    Valide qu'une image peut être utilisée comme sketch
    
    Args:
        image: Image à valider
        
    Returns:
        True si valide, False sinon
    """
    # Vérification de base
    if image is None or image.size == 0:
        logger.error("Image vide ou None")
        return False
    
    # Vérification des dimensions
    if len(image.shape) < 2 or len(image.shape) > 3:
        logger.error(f"Dimensions d'image invalides: {image.shape}")
        return False
    
    # Vérification de la taille minimale
    h, w = image.shape[:2]
    if h < 32 or w < 32:
        logger.error(f"Image trop petite: {h}x{w}")
        return False
    
    # Vérification de la taille maximale
    if h > 4096 or w > 4096:
        logger.error(f"Image trop grande: {h}x{w}")
        return False
    
    # Vérification du contenu (pas complètement vide)
    if len(image.shape) == 2:  # Niveaux de gris
        content_ratio = np.sum(image > 10) / image.size
    else:  # Couleur
        gray = np.mean(image, axis=2)
        content_ratio = np.sum(gray > 10) / gray.size
    
    if content_ratio < 0.01:  # Moins de 1% de contenu
        logger.error(f"Image avec trop peu de contenu: {content_ratio:.3f}")
        return False
    
    if content_ratio > 0.95:  # Plus de 95% de contenu (probablement pas un sketch)
        logger.warning(f"Image très dense, probablement pas un sketch: {content_ratio:.3f}")
    
    return True

def validate_mesh_output(mesh) -> Dict[str, Any]:
    """
    Valide un maillage 3D généré
    
    Returns:
        Dictionnaire avec les résultats de validation
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    try:
        # Statistiques de base
        num_vertices = len(mesh.vertices)
        num_faces = len(mesh.faces)
        
        validation_results['stats'] = {
            'num_vertices': num_vertices,
            'num_faces': num_faces,
            'is_watertight': mesh.is_watertight,
            'is_valid': mesh.is_valid,
            'volume': float(mesh.volume) if mesh.is_watertight else None
        }
        
        # Vérifications
        if num_vertices < 4:
            validation_results['errors'].append("Maillage avec trop peu de vertices")
            validation_results['is_valid'] = False
        
        if num_faces < 4:
            validation_results['errors'].append("Maillage avec trop peu de faces")
            validation_results['is_valid'] = False
        
        if num_vertices > 100000:
            validation_results['warnings'].append(f"Maillage très dense: {num_vertices} vertices")
        
        if not mesh.is_valid:
            validation_results['warnings'].append("Maillage marqué comme invalide par Trimesh")
        
        if not mesh.is_watertight:
            validation_results['warnings'].append("Maillage non étanche (non-watertight)")
        
        # Vérification des normales
        if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
            normal_magnitude = np.linalg.norm(mesh.vertex_normals, axis=1)
            zero_normals = np.sum(normal_magnitude < 0.1)
            if zero_normals > num_vertices * 0.1:
                validation_results['warnings'].append(f"Beaucoup de normales nulles: {zero_normals}")
        
    except Exception as e:
        validation_results['is_valid'] = False
        validation_results['errors'].append(f"Erreur lors de la validation: {str(e)}")
    
    return validation_results

# src/core/file_handler.py
import bpy
import bmesh
import numpy as np
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import trimesh

logger = logging.getLogger(__name__)

class BlenderExporter:
    """
    Gestionnaire d'export vers Blender (.blend)
    """
    
    def __init__(self):
        self.temp_dir = Path(tempfile.gettempdir()) / "sketch3d_exports"
        self.temp_dir.mkdir(exist_ok=True)
    
    def export_mesh_to_blend(self, mesh: trimesh.Trimesh, 
                           output_path: str, 
                           mesh_name: str = "Generated_Mesh") -> Dict[str, Any]:
        """
        Exporte un maillage Trimesh vers un fichier .blend
        
        Args:
            mesh: Maillage Trimesh à exporter
            output_path: Chemin de sortie du fichier .blend
            mesh_name: Nom de l'objet dans Blender
            
        Returns:
            Dictionnaire avec les informations d'export
        """
        try:
            # Sauvegarde temporaire en .obj pour import dans Blender
            temp_obj_path = self.temp_dir / "temp_mesh.obj"
            mesh.export(str(temp_obj_path))
            
            # Nettoyage de Blender
            bpy.ops.wm.read_factory_settings(use_empty=True)
            
            # Import du mesh
            bpy.ops.import_scene.obj(filepath=str(temp_obj_path))
            
            # Renommage de l'objet
            if bpy.context.selected_objects:
                obj = bpy.context.selected_objects[0]
                obj.name = mesh_name
            
            # Configuration de base de la scène
            self._setup_basic_scene()
            
            # Ajout de matériau de base
            self._add_basic_material(obj)
            
            # Sauvegarde du fichier .blend
            bpy.ops.wm.save_as_mainfile(filepath=output_path)
            
            # Nettoyage
            os.remove(temp_obj_path)
            
            export_info = {
                'success': True,
                'file_path': output_path,
                'mesh_name': mesh_name,
                'vertices_count': len(mesh.vertices),
                'faces_count': len(mesh.faces)
            }
            
            logger.info(f"Export .blend réussi: {output_path}")
            
            return export_info
            
        except Exception as e:
            logger.error(f"Erreur lors de l'export .blend: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _setup_basic_scene(self):
        """Configure une scène de base dans Blender"""
        # Ajout d'éclairage
        bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
        
        # Configuration de la caméra
        bpy.ops.object.camera_add(location=(7, 7, 7))
        camera = bpy.context.object
        
        # Pointage de la caméra vers l'origine
        constraint = camera.constraints.new(type='TRACK_TO')
        constraint.track_axis = 'TRACK_NEGATIVE_Z'
        constraint.up_axis = 'UP_Y'
    
    def _add_basic_material(self, obj):
        """Ajoute un matériau de base à l'objet"""
        # Création d'un nouveau matériau
        mat = bpy.data.materials.new(name="Generated_Material")
        mat.use_nodes = True
        
        # Configuration du matériau (couleur de base)
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs[0].default_value = (0.2, 0.5, 0.8, 1.0)  # Couleur bleue
        bsdf.inputs[7].default_value = 0.3  # Rugosité
        
        # Assignation du matériau
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)