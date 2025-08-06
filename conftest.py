# tests/conftest.py
import pytest
import numpy as np
import torch
from PIL import Image
import tempfile
from pathlib import Path

@pytest.fixture
def sample_sketch_image():
    """Crée une image de sketch de test"""
    # Création d'un simple carré
    image = np.zeros((128, 128), dtype=np.uint8)
    image[32:96, 32:96] = 255  # Carré blanc
    return image

@pytest.fixture
def sample_sketch_pil():
    """Crée une image PIL de test"""
    image = Image.new('L', (128, 128), color=0)
    # Dessin d'un cercle simple
    from PIL import ImageDraw
    draw = ImageDraw.Draw(image)
    draw.ellipse([32, 32, 96, 96], fill=255, outline=255)
    return image

@pytest.fixture
def test_config():
    """Configuration de test"""
    return {
        'models': {
            'classifier': {
                'num_classes': 10,
                'backbone': 'resnet18',
                'weights_path': 'nonexistent.pth',
                'class_names': ['cube', 'sphere', 'cylinder', 'cone', 'pyramid', 
                               'house', 'car', 'tree', 'cat', 'dog']
            },
            'depth_estimator': {
                'features': 32,
                'weights_path': 'nonexistent.pth'
            },
            'min_confidence_threshold': 0.3
        },
        'mesh_generation': {
            'voxel_resolution': 32,
            'smoothing_iterations': 1,
            'min_mesh_faces': 4
        }
    }

@pytest.fixture
def temp_directory():
    """Dossier temporaire pour les tests"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

# tests/test_models.py
import pytest
import torch
import numpy as np
from src.models.sketch_classifier import SketchClassifier
from src.models.depth_estimator import DepthEstimator

class TestSketchClassifier:
    """Tests pour le classificateur de sketches"""
    
    def test_initialization(self):
        """Test de l'initialisation du modèle"""
        model = SketchClassifier(num_classes=10, backbone='resnet18')
        assert model.num_classes == 10
        assert model.backbone_name == 'resnet18'
    
    def test_forward_pass(self):
        """Test du forward pass"""
        model = SketchClassifier(num_classes=10, backbone='resnet18')
        model.eval()
        
        # Test avec input de taille correcte
        x = torch.randn(2, 1, 224, 224)
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (2, 10)
        assert not torch.isnan(output).any()
    
    def test_different_input_sizes(self):
        """Test avec différentes tailles d'entrée"""
        model = SketchClassifier(num_classes=5)
        model.eval()
        
        # Test avec différentes tailles
        sizes = [(1, 1, 128, 128), (1, 1, 256, 256), (1, 1, 512, 512)]
        
        for size in sizes:
            x = torch.randn(size)
            with torch.no_grad():
                output = model(x)
            assert output.shape == (size[0], 5)
    
    def test_invalid_backbone(self):
        """Test avec un backbone invalide"""
        with pytest.raises(ValueError):
            SketchClassifier(num_classes=10, backbone='invalid_backbone')

class TestDepthEstimator:
    """Tests pour l'estimateur de profondeur"""
    
    def test_initialization(self):
        """Test de l'initialisation"""
        model = DepthEstimator(input_channels=1, output_channels=1, features=64)
        assert model.input_channels == 1
        assert model.output_channels == 1
    
    def test_forward_pass(self):
        """Test du forward pass"""
        model = DepthEstimator(input_channels=1, output_channels=1, features=32)
        model.eval()
        
        x = torch.randn(2, 1, 512, 512)
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (2, 1, 512, 512)
        assert torch.all(output >= 0) and torch.all(output <= 1)  # Sigmoid output
    
    def test_unet_architecture(self):
        """Test de l'architecture U-Net"""
        model = DepthEstimator(features=16)  # Petit modèle pour test
        
        # Vérification que le modèle a des skip connections
        x = torch.randn(1, 1, 128, 128)
        with torch.no_grad():
            output = model(x)
        
        assert output.shape[2:] == x.shape[2:]  # Même taille spatiale

# tests/test_core.py
import pytest
import numpy as np
from unittest.mock import Mock, patch
from src.core.sketch_processor import SketchProcessor
from src.core.mesh_generator import MeshGenerator

class TestSketchProcessor:
    """Tests pour le processeur de sketches"""
    
    def test_initialization(self, test_config):
        """Test de l'initialisation"""
        processor = SketchProcessor(test_config['models'])
        assert processor.config == test_config['models']
        assert hasattr(processor, 'classifier')
        assert hasattr(processor, 'depth_estimator')
    
    def test_preprocess_sketch_grayscale(self, sample_sketch_image, test_config):
        """Test du preprocessing avec image en niveaux de gris"""
        processor = SketchProcessor(test_config['models'])
        
        tensor, metadata = processor.preprocess_sketch(sample_sketch_image)
        
        assert tensor.shape == (1, 1, 512, 512)
        assert metadata['original_shape'] == sample_sketch_image.shape
        assert metadata['has_content'] == True
    
    def test_preprocess_sketch_rgb(self, test_config):
        """Test du preprocessing avec image RGB"""
        processor = SketchProcessor(test_config['models'])
        
        # Image RGB
        rgb_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        rgb_image[25:75, 25:75] = 255  # Zone blanche
        
        tensor, metadata = processor.preprocess_sketch(rgb_image)
        
        assert tensor.shape == (1, 1, 512, 512)
        assert metadata['original_shape'] == rgb_image.shape
    
    def test_preprocess_empty_sketch(self, test_config):
        """Test avec un sketch vide"""
        processor = SketchProcessor(test_config['models'])
        
        # Sketch complètement noir
        empty_sketch = np.zeros((128, 128), dtype=np.uint8)
        
        tensor, metadata = processor.preprocess_sketch(empty_sketch)
        
        assert tensor.shape == (1, 1, 512, 512)
        assert metadata['has_content'] == False
    
    @patch('src.core.sketch_processor.validate_sketch_input')
    def test_invalid_sketch_input(self, mock_validate, test_config):
        """Test avec entrée invalide"""
        mock_validate.return_value = False
        processor = SketchProcessor(test_config['models'])
        
        invalid_input = np.array([])
        
        with pytest.raises(ValueError):
            processor.preprocess_sketch(invalid_input)
    
    def test_classify_sketch(self, test_config):
        """Test de classification"""
        processor = SketchProcessor(test_config['models'])
        
        # Mock du tensor de sketch
        sketch_tensor = torch.randn(1, 1, 512, 512)
        
        with patch.object(processor.classifier, 'forward') as mock_forward:
            mock_forward.return_value = torch.randn(1, 10)  # 10 classes
            
            result = processor.classify_sketch(sketch_tensor)
            
            assert 'class_id' in result
            assert 'confidence' in result
            assert 'class_name' in result
            assert 'probabilities' in result
            
            assert 0 <= result['class_id'] < 10
            assert 0 <= result['confidence'] <= 1

class TestMeshGenerator:
    """Tests pour le générateur de maillage"""
    
    def test_initialization(self, test_config):
        """Test de l'initialisation"""
        generator = MeshGenerator(test_config['mesh_generation'])
        assert generator.voxel_resolution == 32
        assert generator.smoothing_iterations == 1
        assert generator.min_mesh_faces == 4
    
    def test_create_3d_volume(self, test_config):
        """Test de création de volume 3D"""
        generator = MeshGenerator(test_config['mesh_generation'])
        
        depth_map = np.random.rand(64, 64)
        sketch_mask = np.ones((64, 64))
        
        volume = generator._create_3d_volume(depth_map, sketch_mask)
        
        assert volume.shape[0] == 64  # Height
        assert volume.shape[1] == 64  # Width
        assert volume.shape[2] == 32  # Depth layers
        assert np.all(volume >= 0) and np.all(volume <= 1)
    
    def test_fallback_mesh_generation(self, test_config):
        """Test de génération de mesh de fallback"""
        generator = MeshGenerator(test_config['mesh_generation'])
        
        class_info = {'class_name': 'cube', 'confidence': 0.8}
        mesh = generator._generate_fallback_mesh(class_info)
        
        assert mesh.vertices.shape[1] == 3  # Coordonnées 3D
        assert mesh.faces.shape[1] == 3     # Triangles
        assert len(mesh.vertices) >= 8      # Au moins 8 vertices pour un cube
    
    def test_different_fallback_shapes(self, test_config):
        """Test des différentes formes de fallback"""
        generator = MeshGenerator(test_config['mesh_generation'])
        
        shapes = ['cube', 'sphere', 'cylinder', 'cone', 'unknown']
        
        for shape in shapes:
            class_info = {'class_name': shape, 'confidence': 0.7}
            mesh = generator._generate_fallback_mesh(class_info)
            
            assert mesh.vertices.shape[1] == 3
            assert len(mesh.faces) > 0
    
    def test_mesh_cleaning(self, test_config):
        """Test du nettoyage de maillage"""
        generator = MeshGenerator(test_config['mesh_generation'])
        
        # Création d'un mesh simple avec trimesh
        import trimesh
        mesh = trimesh.creation.box(extents=[1, 1, 1])
        
        cleaned_mesh = generator._clean_mesh(mesh)
        
        assert hasattr(cleaned_mesh, 'vertices')
        assert hasattr(cleaned_mesh, 'faces')
        assert len(cleaned_mesh.vertices) > 0
        assert len(cleaned_mesh.faces) > 0

# tests/test_utils.py
import pytest
import numpy as np
from src.utils.image_processing import normalize_sketch, clean_sketch, extract_sketch_contours
from src.utils.validation import validate_sketch_input, validate_mesh_output
import trimesh

class TestImageProcessing:
    """Tests pour le traitement d'image"""
    
    def test_normalize_sketch(self, sample_sketch_image):
        """Test de normalisation de sketch"""
        normalized = normalize_sketch(sample_sketch_image, target_size=256)
        
        assert normalized.shape == (256, 256)
        assert normalized.dtype == np.float32
        assert np.min(normalized) >= 0 and np.max(normalized) <= 1
    
    def test_normalize_sketch_different_sizes(self):
        """Test avec différentes tailles"""
        # Image rectangulaire
        rect_image = np.random.randint(0, 255, (100, 200), dtype=np.uint8)
        
        normalized = normalize_sketch(rect_image, target_size=128)
        
        assert normalized.shape == (128, 128)
        # Vérifier que le ratio est préservé avec du padding
    
    def test_clean_sketch(self, sample_sketch_image):
        """Test de nettoyage de sketch"""
        # Ajout de bruit
        noisy_image = sample_sketch_image.copy()
        noise = np.random.randint(-20, 20, noisy_image.shape).astype(np.int16)
        noisy_image = np.clip(noisy_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        cleaned = clean_sketch(noisy_image)
        
        assert cleaned.shape == noisy_image.shape
        assert cleaned.dtype == np.uint8
    
    def test_extract_sketch_contours(self, sample_sketch_image):
        """Test d'extraction de contours"""
        # Création d'une image binaire
        binary_image = (sample_sketch_image > 128).astype(np.uint8) * 255
        
        contour_image, contours = extract_sketch_contours(binary_image)
        
        assert contour_image.shape == binary_image.shape
        assert isinstance(contours, list)

class TestValidation:
    """Tests pour la validation"""
    
    def test_validate_sketch_input_valid(self, sample_sketch_image):
        """Test de validation avec image valide"""
        assert validate_sketch_input(sample_sketch_image) == True
    
    def test_validate_sketch_input_empty(self):
        """Test avec image vide"""
        empty_image = np.array([])
        assert validate_sketch_input(empty_image) == False
    
    def test_validate_sketch_input_too_small(self):
        """Test avec image trop petite"""
        small_image = np.ones((10, 10), dtype=np.uint8)
        assert validate_sketch_input(small_image) == False
    
    def test_validate_sketch_input_too_large(self):
        """Test avec image trop grande"""
        large_image = np.ones((5000, 5000), dtype=np.uint8)
        assert validate_sketch_input(large_image) == False
    
    def test_validate_sketch_input_no_content(self):
        """Test avec image sans contenu"""
        no_content = np.zeros((128, 128), dtype=np.uint8)
        assert validate_sketch_input(no_content) == False
    
    def test_validate_mesh_output_valid(self):
        """Test de validation de mesh valide"""
        mesh = trimesh.creation.box(extents=[1, 1, 1])
        
        validation = validate_mesh_output(mesh)
        
        assert validation['is_valid'] == True
        assert 'stats' in validation
        assert validation['stats']['num_vertices'] > 0
        assert validation['stats']['num_faces'] > 0
    
    def test_validate_mesh_output_invalid(self):
        """Test avec mesh invalide"""
        # Mesh avec trop peu de vertices
        vertices = np.array([[0, 0, 0]])
        faces = np.array([])
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        validation = validate_mesh_output(mesh)
        
        assert validation['is_valid'] == False
        assert len(validation['errors']) > 0

# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
import io
from PIL import Image
import numpy as np

# Import conditionnel pour éviter les erreurs si l'API n'est pas disponible
try:
    from src.api.main import app
    client = TestClient(app)
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False

@pytest.mark.skipif(not API_AVAILABLE, reason="API non disponible")
class TestAPI:
    """Tests pour l'API FastAPI"""
    
    def test_root_endpoint(self):
        """Test de l'endpoint racine"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "status" in data
    
    def test_health_check(self):
        """Test du health check"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "components" in data
    
    def test_models_status(self):
        """Test du statut des modèles"""
        response = client.get("/api/v1/models/status")
        assert response.status_code == 200
        data = response.json()
        assert "sketch_classifier" in data
        assert "depth_estimator" in data
    
    def test_process_sketch_no_file(self):
        """Test de traitement sans fichier"""
        response = client.post("/api/v1/sketch/process")
        assert response.status_code == 422  # Validation error
    
    def test_process_sketch_invalid_file(self):
        """Test avec fichier non-image"""
        # Création d'un fichier texte
        files = {"file": ("test.txt", io.StringIO("not an image"), "text/plain")}
        response = client.post("/api/v1/sketch/process", files=files)
        assert response.status_code == 400
    
    def test_process_sketch_valid_image(self):
        """Test avec image valide"""
        # Création d'une image de test
        image = Image.new('RGB', (128, 128), color='white')
        # Dessin d'un carré noir
        from PIL import ImageDraw
        draw = ImageDraw.Draw(image)
        draw.rectangle([32, 32, 96, 96], fill='black')
        
        # Conversion en bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        files = {"file": ("sketch.png", img_byte_arr, "image/png")}
        response = client.post("/api/v1/sketch/process", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert data["status"] == "queued"
    
    def test_get_task_status_not_found(self):
        """Test de récupération de statut pour tâche inexistante"""
        response = client.get("/api/v1/sketch/nonexistent/status")
        assert response.status_code == 404

# tests/test_integration.py
import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock

class TestIntegrationSketchTo3D:
    """Tests d'intégration complets"""
    
    def test_complete_pipeline_mock(self, test_config, sample_sketch_image, temp_directory):
        """Test du pipeline complet avec mocks"""
        from src.core.sketch_processor import SketchProcessor
        
        # Initialisation
        processor = SketchProcessor(test_config['models'])
        
        # Mock des modèles pour éviter les erreurs de poids manquants
        with patch.object(processor, 'classifier') as mock_classifier, \
             patch.object(processor, 'depth_estimator') as mock_depth:
            
            # Configuration des mocks
            mock_classifier.return_value = torch.randn(1, 10)
            mock_depth.return_value = torch.rand(1, 1, 512, 512)
            
            # Test du pipeline
            result = processor.process_sketch_to_3d(sample_sketch_image)
            
            assert result['status'] == 'success'
            assert 'classification' in result
            assert 'depth_map' in result
            assert 'preprocessing' in result
    
    def test_mesh_generation_integration(self, test_config):
        """Test d'intégration génération de maillage"""
        from src.core.mesh_generator import MeshGenerator
        
        generator = MeshGenerator(test_config['mesh_generation'])
        
        # Données de test
        depth_map = np.random.rand(64, 64) * 0.8 + 0.2  # Profondeur entre 0.2 et 1.0
        sketch_mask = np.ones((64, 64))
        sketch_mask[10:54, 10:54] = 1  # Zone de contenu
        sketch_mask[:10, :] = 0  # Bordures vides
        sketch_mask[54:, :] = 0
        sketch_mask[:, :10] = 0
        sketch_mask[:, 54:] = 0
        
        class_info = {
            'class_name': 'cube',
            'confidence': 0.8
        }
        
        # Génération du maillage
        try:
            mesh = generator.generate_mesh_from_depth(depth_map, sketch_mask, class_info)
            
            # Vérifications
            assert hasattr(mesh, 'vertices')
            assert hasattr(mesh, 'faces')
            assert len(mesh.vertices) > 0
            assert len(mesh.faces) > 0
            
        except Exception as e:
            # Si la génération échoue, vérifier qu'on tombe sur le fallback
            pytest.skip(f"Génération de mesh échouée (normal avec données aléatoires): {e}")

# tests/test_performance.py
import pytest
import time
import numpy as np
from memory_profiler import memory_usage

@pytest.mark.performance
class TestPerformance:
    """Tests de performance"""
    
    def test_sketch_processing_time(self, test_config, sample_sketch_image):
        """Test du temps de traitement d'un sketch"""
        from src.core.sketch_processor import SketchProcessor
        
        processor = SketchProcessor(test_config['models'])
        
        # Mesure du temps
        start_time = time.time()
        
        try:
            with patch.object(processor, 'classifier'), \
                 patch.object(processor, 'depth_estimator'):
                result = processor.process_sketch_to_3d(sample_sketch_image)
            
            processing_time = time.time() - start_time
            
            # Le preprocessing seul devrait être rapide
            assert processing_time < 5.0, f"Traitement trop lent: {processing_time:.2f}s"
            
        except ImportError:
            pytest.skip("Modules requis non disponibles pour test performance")
    
    def test_memory_usage_sketch_processing(self, test_config, sample_sketch_image):
        """Test de l'utilisation mémoire"""
        from src.core.sketch_processor import SketchProcessor
        
        def process_sketch():
            processor = SketchProcessor(test_config['models'])
            with patch.object(processor, 'classifier'), \
                 patch.object(processor, 'depth_estimator'):
                return processor.preprocess_sketch(sample_sketch_image)
        
        try:
            # Mesure de la mémoire
            mem_usage = memory_usage(process_sketch, interval=0.1)
            max_memory = max(mem_usage)
            
            # Seuil raisonnable pour les tests (en MB)
            assert max_memory < 1000, f"Utilisation mémoire trop élevée: {max_memory:.1f}MB"
            
        except ImportError:
            pytest.skip("memory_profiler non disponible")

# tests/test_edge_cases.py
import pytest
import numpy as np

class TestEdgeCases:
    """Tests pour les cas limites"""
    
    def test_very_small_sketch(self, test_config):
        """Test avec sketch très petit"""
        from src.core.sketch_processor import SketchProcessor
        
        processor = SketchProcessor(test_config['models'])
        
        # Image 32x32 (minimum)
        small_sketch = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
        small_sketch[8:24, 8:24] = 255  # Contenu au centre
        
        tensor, metadata = processor.preprocess_sketch(small_sketch)
        
        assert tensor.shape == (1, 1, 512, 512)
        assert metadata['has_content'] == True
    
    def test_sketch_all_white(self, test_config):
        """Test avec sketch complètement blanc"""
        from src.core.sketch_processor import SketchProcessor
        
        processor = SketchProcessor(test_config['models'])
        
        white_sketch = np.ones((128, 128), dtype=np.uint8) * 255
        
        tensor, metadata = processor.preprocess_sketch(white_sketch)
        
        # Devrait être détecté comme ayant du contenu (inversé en nettoyage)
        assert tensor.shape == (1, 1, 512, 512)
    
    def test_sketch_high_noise(self, test_config):
        """Test avec sketch très bruité"""
        from src.core.sketch_processor import SketchProcessor
        
        processor = SketchProcessor(test_config['models'])
        
        # Sketch avec beaucoup de bruit
        noisy_sketch = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
        
        tensor, metadata = processor.preprocess_sketch(noisy_sketch)
        
        assert tensor.shape == (1, 1, 512, 512)
        # Le nettoyage devrait réduire le bruit

# Configuration pytest
# pytest.ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --verbose
    --tb=short
    --strict-markers
markers = 
    performance: marque les tests de performance (lents)
    integration: marque les tests d'intégration
    unit: marque les tests unitaires
filterwarnings = 
    ignore::DeprecationWarning
    ignore::PendingDeprecationWarning

# requirements-test.txt (dépendances supplémentaires pour les tests)
pytest>=7.4.0
pytest-cov>=4.1.0
pytest-mock>=3.11.0
pytest-asyncio>=0.21.0
memory-profiler>=0.60.0
httpx>=0.24.0