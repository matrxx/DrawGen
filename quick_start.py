# scripts/quick_start.py
#!/usr/bin/env python3

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
    logger.info("🔍 Dependences check...")
    
    required_packages = [
        'torch', 'torchvision', 'opencv-python', 'fastapi', 'uvicorn',
        'numpy', 'pillow', 'open3d', 'trimesh', 'scikit-image'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
            elif package == 'scikit-image':
                import skimage
            else:
                __import__(package.replace('-', '_'))
            logger.info(f"  ✓ {package}")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"  ✗ {package}")
    
    if missing_packages:
        logger.error(f"Packages missing: {missing_packages}")
        logger.info("Installe with: pip install " + " ".join(missing_packages))
        return False
    
    logger.info("✅ Everything is ok")
    return True

def test_models():
    """Initializing models"""
    logger.info("🧠 Testing models...")
    
    try:
        from src.models.sketch_classifier import SketchClassifier
        classifier = SketchClassifier(num_classes=10)
        logger.info("  ✓ SketchClassifier")
        
        from src.models.depth_estimator import DepthEstimator
        depth_model = DepthEstimator()
        logger.info("  ✓ DepthEstimator")
        
        import torch
        test_input = torch.randn(1, 1, 224, 224)
        
        with torch.no_grad():
            output = classifier(test_input)
            assert output.shape == (1, 10)
        
        test_input = torch.randn(1, 1, 512, 512)
        with torch.no_grad():
            output = depth_model(test_input)
            assert output.shape == (1, 1, 512, 512)
        
        logger.info("✅ Models working")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error models: {e}")
        return False

def test_core_pipeline():
    logger.info("⚙️ Major pipeline test...")
    
    try:
        from src.utils.validation import validate_sketch_input
        from src.utils.image_processing import normalize_sketch, clean_sketch
        
        test_sketch = np.zeros((128, 128), dtype=np.uint8)
        test_sketch[32:96, 32:96] = 255
        
        assert validate_sketch_input(test_sketch) == True
        logger.info("  ✓ Validation")
        
        normalized = normalize_sketch(test_sketch)
        assert normalized.shape == (512, 512)
        logger.info("  ✓ Normalisation")
        
        cleaned = clean_sketch(test_sketch)
        assert cleaned.shape == test_sketch.shape
        logger.info("  ✓ Nettoyage")
        
        logger.info("✅ Pipeline working")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error pipeline: {e}")
        return False

def test_mesh_generation():
    logger.info("🔺 Mesh generation test...")
    
    try:
        from src.core.mesh_generator import MeshGenerator
        import trimesh
        
        config = {
            'voxel_resolution': 32,
            'smoothing_iterations': 1,
            'min_mesh_faces': 4
        }
        
        generator = MeshGenerator(config)
        
        depth_map = np.random.rand(64, 64) * 0.8 + 0.2
        sketch_mask = np.ones((64, 64))
        class_info = {'class_name': 'cube', 'confidence': 0.8}
        
        mesh = generator._generate_fallback_mesh(class_info)
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0
        
        logger.info("  ✓ Mesh fallback generation")
        logger.info("✅ Mesh generation working")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error mesh generation: {e}")
        logger.warning("Could be normal if Open3D/Trimesh aren't working")
        return False

def create_test_sketch():
    logger.info("🎨 Sketch test creation...")
    
    img = Image.new('RGB', (256, 256), color='white')
    draw = ImageDraw.Draw(img)
    
    draw.rectangle([50, 150, 200, 230], outline='black', width=3)
    
    draw.polygon([40, 150, 125, 80, 210, 150], outline='black', width=3)
    
    draw.rectangle([100, 180, 140, 230], outline='black', width=2)
    
    draw.rectangle([70, 170, 90, 190], outline='black', width=2)
    draw.rectangle([160, 170, 180, 190], outline='black', width=2)
    
    test_dir = Path("data/temp")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    sketch_path = test_dir / "test_house.png"
    img.save(sketch_path)
    
    logger.info(f"  ✓ Sketch sauvé: {sketch_path}")
    return sketch_path

def start_api_server():
    logger.info("🚀 API server starting...")
    
    try:
        from src.api.main import app
        import uvicorn
        
        import threading
        
        def run_server():
            uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        time.sleep(3)
        
        try:
            response = requests.get("http://127.0.0.1:8000/health", timeout=5)
            if response.status_code == 200:
                logger.info("✅ API successfully started")
                logger.info("📡 API reachable at: http://127.0.0.1:8000")
                logger.info("📚 Documentation: http://127.0.0.1:8000/docs")
                return True
            else:
                logger.error(f"❌ API answer with code {response.status_code}")
                return False
        
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ API not reachable: {e}")
            return False
    
    except Exception as e:
        logger.error(f"❌ Error API not starting: {e}")
        return False

def test_api_with_sketch():
    logger.info("🧪 API test with a sketch...")
    
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
                status_data = status_response.json()
                logger.info(f"  ✓ Statut: {status_data['status']}")
                logger.info("✅ Test API successful")
                return True
            
        logger.error(f"❌ Test API error: {response.status_code}")
        return False
        
    except Exception as e:
        logger.error(f"❌ Error test API: {e}")
        return False

def show_usage_examples():
    logger.info("\n" + "="*50)
    logger.info("📖 Usage Examples")
    logger.info("="*50)
    
    print("""
🔧 Command Line:
    
    # Models training
    python scripts/train_models.py --model classifier
    python scripts/train_models.py --model all
    
    # Tests
    python -m pytest tests/
    pytest tests/ -v --cov=src
    
    # Start API
    python -m src.api.main
    make run-dev

🌐 API REST:
    
    # Health check
    GET http://localhost:8000/health
    
    # Sketch processing
    POST http://localhost:8000/api/v1/sketch/process
    Content-Type: multipart/form-data
    file: [votre_sketch.png]
    
    # Status processing
    GET http://localhost:8000/api/v1/sketch/{task_id}/status
    
    # Downloading the .blend file
    GET http://localhost:8000/api/v1/sketch/{task_id}/download

🐍 PYTHON:
    
    from src.core.sketch_processor import SketchProcessor
    from src.core.mesh_generator import MeshGenerator
    
    # Configuration
    config = {...}  # Voir config/model_config.yaml
    
    # Processing
    processor = SketchProcessor(config['models'])
    result = processor.process_sketch_to_3d(image_array)
    
    # 3D Generation
    generator = MeshGenerator(config['mesh_generation'])
    mesh = generator.generate_mesh_from_depth(
        depth_map, sketch_mask, class_info
    )

🐳 DOCKER:
    
    # Building
    docker build -f docker/Dockerfile -t drawgen .
    
    # Starting
    docker-compose -f docker/docker-compose.yml up -d
    
    # Logs
    docker-compose logs -f sketch3d-api
""")

def main():
    print("\n" + "="*60)
    print("🎯 SKETCH-TO-3D - DÉMARRAGE RAPIDE")
    print("="*60)
    
    success = True
    
    success &= check_dependencies()
    success &= test_models()
    success &= test_core_pipeline()
    success &= test_mesh_generation()
    
    if not success:
        logger.error("\n❌ Some tests failed. Check the logs for more details.")
        logger.info("💡 Try: pip install -r requirements.txt")
        return 1
    
    logger.info("\n🎉 All tests working!")
    
    print("\n" + "="*60)
    print("🚀 Starting API")
    print("="*60)
    
    if start_api_server():
        test_api_with_sketch()
        
        show_usage_examples()
        
        print("\n" + "="*60)
        print("✅ DrawGen ready to use")
        print("="*60)
        print("\n🌐 API reachable at: http://127.0.0.1:8000")
        print("📚 Documentation: http://127.0.0.1:8000/docs")
        print("🧪 Interface test: http://127.0.0.1:8000/docs#/default/process_sketch_api_v1_sketch_process_post")
        print("\n⌨️  Press Ctrl+C to stop")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n👋 Server stopping...")
            return 0
    else:
        logger.error("❌ Impossible to start the API")
        return 1

if __name__ == "__main__":
    exit(main())

# scripts/demo.py
#!/usr/bin/env python3
"""
Demo Script for DrawGen
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import requests
import time
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InteractiveDemo:
    
    def __init__(self, api_url="http://localhost:8000"):
        self.api_url = api_url
        self.demo_dir = Path("data/demo")
        self.demo_dir.mkdir(parents=True, exist_ok=True)
    
    def create_sample_sketches(self):
        logger.info("🎨 Sketches example creation...")
        
        sketches = []
        
        house = self._draw_house()
        house_path = self.demo_dir / "house.png"
        house.save(house_path)
        sketches.append(("House", house_path))
        
        car = self._draw_car()
        car_path = self.demo_dir / "car.png"
        car.save(car_path)
        sketches.append(("Car", car_path))
        
        tree = self._draw_tree()
        tree_path = self.demo_dir / "tree.png"
        tree.save(tree_path)
        sketches.append(("Tree", tree_path))
        
        cube = self._draw_cube()
        cube_path = self.demo_dir / "cube.png"
        cube.save(cube_path)
        sketches.append(("Cube", cube_path))
        
        logger.info(f"✅ {len(sketches)} sketches created")
        return sketches
    
    def _draw_house(self):
        img = Image.new('RGB', (256, 256), color='white')
        draw = ImageDraw.Draw(img)
        
        draw.rectangle([60, 140, 196, 220], outline='black', width=3)
        
        draw.polygon([50, 140, 128, 70, 206, 140], outline='black', width=3)
        
        draw.rectangle([110, 180, 146, 220], outline='black', width=2)
        draw.circle([140, 200], 2, fill='black')

        draw.rectangle([75, 155, 95, 175], outline='black', width=2)
        draw.line([85, 155, 85, 175], fill='black', width=1)
        draw.line([75, 165, 95, 165], fill='black', width=1)
        
        draw.rectangle([160, 155, 180, 175], outline='black', width=2)
        draw.line([170, 155, 170, 175], fill='black', width=1)
        draw.line([160, 165, 180, 165], fill='black', width=1)
        
        draw.rectangle([150, 80, 170, 110], outline='black', width=2)
        
        return img
    
    def _draw_car(self):
        img = Image.new('RGB', (256, 256), color='white')
        draw = ImageDraw.Draw(img)
        
        draw.rounded_rectangle([40, 140, 216, 180], radius=10, outline='black', width=3)
        
        draw.rounded_rectangle([80, 110, 176, 140], radius=8, outline='black', width=3)
        
        draw.circle([70, 190], 20, outline='black', width=3)
        draw.circle([186, 190], 20, outline='black', width=3)
        
        draw.circle([70, 190], 8, outline='black', width=2)
        draw.circle([186, 190], 8, outline='black', width=2)
        
        draw.rectangle([85, 115, 125, 135], outline='black', width=2)
        draw.rectangle([131, 115, 171, 135], outline='black', width=2)
        
        draw.circle([216, 155], 8, outline='black', width=2)
        draw.circle([40, 155], 8, outline='black', width=2)
        
        return img
    
    def _draw_tree(self):
        img = Image.new('RGB', (256, 256), color='white')
        draw = ImageDraw.Draw(img)
        
        draw.rectangle([118, 160, 138, 230], outline='black', width=3, fill='white')
        
        draw.circle([128, 120], 40, outline='black', width=3)
        draw.circle([108, 100], 30, outline='black', width=2)
        draw.circle([148, 100], 30, outline='black', width=2)
        draw.circle([128, 80], 25, outline='black', width=2)
        
        draw.line([125, 170, 125, 220], fill='black', width=1)
        draw.line([131, 175, 131, 225], fill='black', width=1)
        
        return img
    
    def _draw_cube(self):
        img = Image.new('RGB', (256, 256), color='white')
        draw = ImageDraw.Draw(img)
        
        draw.rectangle([80, 120, 160, 200], outline='black', width=3)
        
        draw.polygon([160, 120, 200, 80, 200, 160, 160, 200], outline='black', width=3)
        
        draw.polygon([80, 120, 120, 80, 200, 80, 160, 120], outline='black', width=3)
        
        draw.line([80, 200, 120, 160], fill='gray', width=1)
        draw.line([120, 160, 200, 160], fill='gray', width=1)
        draw.line([120, 160, 120, 80], fill='gray', width=1)
        
        return img
    
    def test_api_connection(self):
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info("✅ API connected")
                return True
            else:
                logger.error(f"❌ API answer with the following code{response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Impossible to reach the API: {e}")
            return False
    
    def process_sketch(self, sketch_path, sketch_name):
        logger.info(f"🔄 Treatment of {sketch_name}...")
        
        try:
            with open(sketch_path, 'rb') as f:
                files = {'file': (sketch_path.name, f, 'image/png')}
                data = {'mesh_name': f"Generated_{sketch_name}"}
                
                response = requests.post(
                    f"{self.api_url}/api/v1/sketch/process",
                    files=files,
                    data=data,
                    timeout=10
                )
            
            if response.status_code != 200:
                logger.error(f"❌ Error upload: {response.status_code}")
                return None
            
            task_data = response.json()
            task_id = task_data['task_id']
            logger.info(f"📋 Tâche créée: {task_id}")
            
            max_attempts = 30
            for attempt in range(max_attempts):
                time.sleep(2)
                
                status_response = requests.get(
                    f"{self.api_url}/api/v1/sketch/{task_id}/status",
                    timeout=5
                )
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    status = status_data['status']
                    progress = status_data.get('progress', 0)
                    
                    print(f"  📊 {status} - {progress}%")
                    
                    if status == 'completed':
                        logger.info(f"✅ {sketch_name} success!")
                        return task_id
                    elif status == 'failed':
                        error = status_data.get('error', 'Error')
                        logger.error(f"❌ Treatment failure: {error}")
                        return None
                else:
                    logger.warning(f"⚠️ Status error: {status_response.status_code}")
            
            logger.error(f"⏰ Timeout for {sketch_name}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Treatment failure {sketch_name}: {e}")
            return None
    
    def download_blend_file(self, task_id, sketch_name):
        try:
            download_response = requests.get(
                f"{self.api_url}/api/v1/sketch/{task_id}/download",
                timeout=30
            )
            
            if download_response.status_code == 200:
                blend_path = self.demo_dir / f"{sketch_name.lower()}_model.blend"
                
                with open(blend_path, 'wb') as f:
                    f.write(download_response.content)
                
                logger.info(f"💾 File .blend saved: {blend_path}")
                return blend_path
            else:
                logger.error(f"❌ Download error: {download_response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Download error: {e}")
            return None
    
    def show_results_summary(self, results):
        print("\n" + "="*60)
        print("📊 SUMMARY OF THE DEMONSTRATION")
        print("="*60)
        
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        print(f"\n✅ Succeed: {len(successful)}/{len(results)}")
        for result in successful:
            print(f"  • {result['name']}: {result['blend_file']}")
        
        if failed:
            print(f"\n❌ Failed: {len(failed)}")
            for result in failed:
                print(f"  • {result['name']}: {result['error']}")
        
        print(f"\n📁 All files are in: {self.demo_dir}")
        
        if successful:
            print("\n💡 NEXT STEPS:")
            print("  1. Open Blender")
            print("  2. File > Import > Blender (.blend)")
            print("  3. Choose the .blend file you generated")
            print("  4. Have fun!")
    
    def run_full_demo(self):
        print("\n" + "="*60)
        print("🎯 DrawGen Demo")
        print("="*60)
        
        if not self.test_api_connection():
            logger.error("❌ API not accessible. Start the API with: python -m src.api.main")
            return
        
        sketches = self.create_sample_sketches()
        
        results = []
        
        for sketch_name, sketch_path in sketches:
            print(f"\n{'='*40}")
            print(f"🎨 Treatment: {sketch_name}")
            print(f"{'='*40}")
            
            task_id = self.process_sketch(sketch_path, sketch_name)
            
            if task_id:
                blend_file = self.download_blend_file(task_id, sketch_name)
                
                results.append({
                    'name': sketch_name,
                    'success': blend_file is not None,
                    'blend_file': blend_file,
                    'task_id': task_id,
                    'error': None
                })
            else:
                results.append({
                    'name': sketch_name,
                    'success': False,
                    'blend_file': None,
                    'task_id': None,
                    'error': "Treatment failed"
                })
        
        self.show_results_summary(results)
        
        return results

def main():
    demo = InteractiveDemo()
    
    print("🚀 Starting the demonstration...")
    print("⚠️  Make sure the API is started (python -m src.api.main)")
    
    input("Press Enter to continue...")
    
    results = demo.run_full_demo()
    
    print("\n🎉 Demonstration over!")
    
    return 0 if any(r['success'] for r in results) else 1

if __name__ == "__main__":
    exit(main())

# GETTING_STARTED.md
# DrawGen Quick Start

## 🚀  Quick Installation

### Prerequisites
- Python 3.9+
- CUDA 11.8+ (optionnel, pour GPU)
- 8GB RAM minimum
- 2GB espace disque

### Automatic Installation

```bash
# Clone the project
git clone https://github.com/votre-repo/sketch3d-backend
cd sketch3d-backend

# Complete installation
make setup-dev
```

### Manual Installation

```bash
# Virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Dependencies
pip install -r requirements.txt

# Folder structure
mkdir -p data/{datasets,models,temp} logs checkpoints
```

## 🏁 Quick start

### Option 1: Automatic Script
```bash
python scripts/quick_start.py
```

### Option 2: Manual
```bash
# Components check
python -c "from src.models.sketch_classifier import SketchClassifier; print('✅ OK')"

# API Start
python -m src.api.main
```

## 🧪 First Use

### 1. Via Web Interface
1. Open http://localhost:8000/docs
2. Try the endpoint `/api/v1/sketch/process`
3. Upload a sketch image
4. Download the .blend file

```python
import requests

with open('mon_sketch.png', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/api/v1/sketch/process', files=files)

task_id = response.json()['task_id']

status = requests.get(f'http://localhost:8000/api/v1/sketch/{task_id}/status')
print(status.json())
```

```bash
python scripts/demo.py
```


```bash
make test

pytest tests/test_models.py -v

pytest tests/test_performance.py -m performance
```


**Models:**
```python
from src.models.sketch_classifier import SketchClassifier
from src.models.depth_estimator import DepthEstimator
import torch

# Test classifier
model = SketchClassifier(num_classes=10)
x = torch.randn(1, 1, 224, 224)
output = model(x)

print("✅ Models OK")
```

**Pipeline:**
```python
from src.core.sketch_processor import SketchProcessor
import numpy as np

config = {'models': {...}}  # Check config/model_config.yaml
processor = SketchProcessor(config['models'])

test_image = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
result = processor.preprocess_sketch(test_image)

print("✅ Pipeline OK")
```

## 🔧 Configuration

### Configuration Files
- `config/model_config.yaml` - Model settings
- `requirements.txt` - Python Dependencies
- `docker-compose.yml` - Docker Configuration

### Environment Variables
```bash
export CUDA_VISIBLE_DEVICES=0
export MODEL_PATH=/path/to/models
export TEMP_PATH=/tmp/drawgen
```

## 🐳 Docker (Alternative)

### Start with Docker
```bash
# Building
docker build -f docker/Dockerfile -t drawgen .

# Start
docker-compose up -d

# Check
curl http://localhost:8000/health
```

## 🚨 Common Problems

### Error CUDA
```bash
# Check CUDA
nvidia-smi

# Install PyTorch CPU if no GPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Errors Open3D/Trimesh
```bash
# Update
pip install --upgrade open3d trimesh

# Alternative conda
conda install -c open3d-admin open3d
```

### Error Blender Export
```bash
# Install Blender Python API
pip install bpy

# Where Blender system
export BLENDER_PATH=/path/to/blender
```

### Port 8000 Used
```bash
# Change the port
uvicorn src.api.main:app --port 8001

# Kill process
lsof -i :8000
kill -9 <PID>
```

## 📚 Additional Ressources

### Documentation
- [Architecture](docs/architecture.md)
- [API Reference](docs/api.md)
- [Training Guide](docs/training.md)

### Datasets
```bash
# Download demo datasets
python scripts/download_datasets.py --all

# QuickDraw full (optional)
python scripts/download_datasets.py --quickdraw
```

### Models training
```bash
# Training classifier
python scripts/train_models.py --model classifier

# Train models
python scripts/train_models.py --model all
```

## 🆘 Support

### Logs and Debugging
```bash
# Detailled logs
export PYTHONPATH=/path/to/sketch3d-backend
python -m src.api.main --log-level debug

# Tests with coverage
pytest --cov=src --cov-report=html
```

### Useful Commands
```bash
# System Status
make health-check

# Cleaning
make clean

# Update dependencies
pip install -r requirements.txt --upgrade
```

**If the problem persists:**
1. Check the logs in `logs/`
2. Test the components individually
3. Consult the API documentation
4. Open an issue with the error logs

---

🎉 **Congratulations!** Drawgen is now operational.

Check http://localhost:8000/docs for the interactive interface.
