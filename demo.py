# demo.py
"""Demo for DrawGen"""

import sys
import subprocess
import time
import requests
import numpy as np
from pathlib import Path
import logging
from PIL import Image, ImageDraw
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check Dependencies"""
    logger.info("🔍 Dependencies check...")
    
    required_packages = [
        ('torch', 'torch'), ('torchvision', 'torchvision'), 
        ('opencv-python', 'cv2'), ('fastapi', 'fastapi'), ('uvicorn', 'uvicorn'),
        ('numpy', 'numpy'), ('pillow', 'PIL'), ('open3d', 'open3d'),
        ('trimesh', 'trimesh'), ('scikit-image', 'skimage')
    ]
    
    missing_packages = []
    
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            logger.info(f"  ✓ {package_name}")
        except ImportError:
            missing_packages.append(package_name)
            logger.error(f"  ✗ {package_name}")
    
    if missing_packages:
        if missing_packages == ['open3d']:
            logger.warning(f"Package optionnel manquant: {missing_packages}")
            logger.info("Open3D n'est pas critique, le système peut fonctionner avec Trimesh")
            return True
        else:
            logger.error(f"Packages essentiels manquants: {missing_packages}")
            logger.info("Installez avec: pip install " + " ".join(missing_packages))
            return False
    
    logger.info("✅ All dependencies found")
    return True

def test_models():
    """Models testing"""
    logger.info("🧠 Models testing...")
    
    try:
        from models import SketchClassifier, DepthEstimator
        import torch
        
        classifier = SketchClassifier(num_classes=10)
        test_input = torch.randn(1, 1, 224, 224)
        
        with torch.no_grad():
            output = classifier(test_input)
            assert output.shape == (1, 10)
        
        logger.info("  ✓ SketchClassifier")
        
        depth_model = DepthEstimator()
        test_input = torch.randn(1, 1, 512, 512)
        
        with torch.no_grad():
            output = depth_model(test_input)
            assert output.shape == (1, 1, 512, 512)
        
        logger.info("  ✓ DepthEstimator")
        logger.info("✅ Working Models")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error Models: {e}")
        return False

def test_pipeline():
    """Main pipeline test"""
    logger.info("⚙️ Pipeline test...")
    
    try:
        from utils import validate_sketch_input, normalize_sketch, clean_sketch
        
        test_sketch = np.zeros((128, 128), dtype=np.uint8)
        test_sketch[32:96, 32:96] = 255
        
        assert validate_sketch_input(test_sketch) == True
        logger.info("  ✓ Validation")
        
        normalized = normalize_sketch(test_sketch)
        assert normalized.shape == (512, 512)
        logger.info("  ✓ Standardization")
        
        cleaned = clean_sketch(test_sketch)
        assert cleaned.shape == test_sketch.shape
        logger.info("  ✓ Cleaning")
        
        logger.info("✅ Working pipeline")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error pipeline: {e}")
        return False

def test_mesh_generation():
    """Mesh generation test"""
    logger.info("🔺 Test mesh...")
    
    try:
        from mesh_generator import MeshGenerator
        import trimesh
        
        config = {
            'voxel_resolution': 32,
            'smoothing_iterations': 1,
            'min_mesh_faces': 4
        }
        
        generator = MeshGenerator(config)
        
        class_info = {'class_name': 'cube', 'confidence': 0.8}
        mesh = generator._generate_fallback_mesh(class_info)
        
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0
        
        logger.info("  ✓ Fallback generation")
        logger.info("✅ Mesh generation OK")
        return True
        
    except Exception as e:
        logger.error(f"❌ Mesh error: {e}")
        return False

def create_test_sketch():
    """Skecth testing"""
    logger.info("🎨 Sketch testing creation...")
    
    img = Image.new('RGB', (256, 256), color='white')
    draw = ImageDraw.Draw(img)
    
    draw.rectangle([50, 150, 200, 230], outline='black', width=3)
    draw.polygon([40, 150, 125, 80, 210, 150], outline='black', width=3)
    draw.rectangle([100, 180, 140, 230], outline='black', width=2)
    draw.rectangle([70, 170, 90, 190], outline='black', width=2)
    draw.rectangle([160, 170, 180, 190], outline='black', width=2)
    
    test_dir = Path("temp")
    test_dir.mkdir(exist_ok=True)
    
    sketch_path = test_dir / "test_house.png"
    img.save(sketch_path)
    
    logger.info(f"  ✓ Sketch sauvé: {sketch_path}")
    return sketch_path

def start_api_server():
    """Background API Start"""
    logger.info("🚀 API Starting...")
    
    try:
        import uvicorn
        import threading
        from api import app
        
        def run_server():
            uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        time.sleep(3)
        
        response = requests.get("http://127.0.0.1:8000/health", timeout=5)
        if response.status_code == 200:
            logger.info("✅ API started")
            return True
        else:
            logger.error(f"❌ API error {response.status_code}")
            return False
        
    except Exception as e:
        logger.error(f"❌ Error API: {e}")
        return False

def test_api_with_sketch():
    """API test with a sketch"""
    logger.info("🧪 Test API...")
    
    try:
        sketch_path = create_test_sketch()
        
        with open(sketch_path, 'rb') as f:
            files = {'file': ('test_house.png', f, 'image/png')}
            response = requests.post(
                "http://127.0.0.1:8000/api/v1/sketch/process",
                files=files,
                timeout=10
            )
        
        if response.status_code == 200:
            data = response.json()
            task_id = data.get('task_id')
            logger.info(f"  ✓ Task created: {task_id}")
            
            status_response = requests.get(
                f"http://127.0.0.1:8000/api/v1/sketch/{task_id}/status",
                timeout=5
            )
            
            if status_response.status_code == 200:
                logger.info("✅ Test API successful")
                return True
            
        logger.error("❌ Test API failed")
        return False
        
    except Exception as e:
        logger.error(f"❌ Error test API: {e}")
        return False

def main():
    """Main function"""
    print("\n" + "="*50)
    print("🎯 DrawGen - DEMONSTRATION")
    print("="*50)
    
    success = True
    success &= check_dependencies()
    success &= test_models()
    success &= test_pipeline()
    success &= test_mesh_generation()
    
    if not success:
        logger.error("\n❌ Some tests failed")
        logger.info("💡 Try: pip install -r requirements.txt")
        return 1
    
    logger.info("\n🎉 Tous les tests passent!")
    
    if start_api_server():
        test_api_with_sketch()
        
        print("\n" + "="*50)
        print("✅ DrawGen Ready")
        print("="*50)
        print("\n🌐 API: http://127.0.0.1:8000")
        print("📚 Docs: http://127.0.0.1:8000/docs")
        print("\n💡 Usage:")
        print("  1. Open http://127.0.0.1:8000/docs")
        print("  2. Try /api/v1/sketch/process")
        print("  3. Upload your sketchs!")
        print("\n⌨️  Ctrl+C to stop")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n👋 Bye!")
            return 0
    else:
        logger.error("❌ Impossible to start the API")
        return 1

if __name__ == "__main__":

    exit(main())
