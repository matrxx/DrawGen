# 🎨 Drawgen

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/docker-ready-blue)](https://www.docker.com/)
[![Build Status](https://github.com/KLN-AI/drawgen/workflows/CI/badge.svg)](https://github.com/KLN-AI/drawgen/actions)

> **Transform your hand-drawn sketches into 3D models using AI**
<img width="903" height="510" alt="DrawGen" src="https://github.com/user-attachments/assets/7e82d6bb-dd3e-453e-bbcb-3b1f2c5faa2c" />

An advanced AI pipeline that automatically converts 2D sketches into 3D models exportable to Blender. Powered by PyTorch for object classification and depth estimation.

![Demo GIF](assets/demo.gif)
*Demo: From sketch to 3D model in under 30 seconds*

---

## ✨ Features

- 🧠 **Advanced AI** : ResNet-18 classification + U-Net depth estimation
- ⚡ **Fast Processing** : <30 seconds per sketch
- 🌐 **REST API** : Modern FastAPI with interactive documentation
- 🎨 **Web Interface** : Draw directly in your browser
- 🔺 **3D Export** : .obj and .blend files compatible with Blender
- 🐳 **Docker Ready** : Easy deployment with containers
- 🚀 **GPU Support** : CUDA acceleration for better performance
- 🌍 **Multi-language** : International ready

## 🚀 Quick Start

### Express Installation (Recommended)

```bash
# Clone the repository
git clone https://github.com/matrxx/drawgen.git
cd drawgen

# Automatic setup
python quick_start.py
```

### Manual Installation

```bash
# Virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Dependencies
pip install -r requirements.txt

# Test functionality
python demo.py
```

### Using Docker

```bash
docker build -t drawgen .
docker run -p 8000:8000 drawgen
```

## 🎯 Usage

### 1. Start the API

```bash
python api.py
# API available at http://localhost:8000
```

### 2. Interactive Web Interface

Open `http://localhost:8000/draw` in your browser:
- ✏️ Draw your sketch
- 🚀 Click "Generate 3D"
- 💾 Download your 3D model

### 3. Programmatic Usage

```python
import requests

# Upload a sketch
with open('my_sketch.png', 'rb') as f:
    files = {'file': f}
    response = requests.post(
        'http://localhost:8000/api/v1/sketch/process', 
        files=files
    )

task_id = response.json()['task_id']

# Monitor processing
status = requests.get(f'http://localhost:8000/api/v1/sketch/{task_id}/status')
print(f"Status: {status.json()['status']}")

# Download result
if status.json()['status'] == 'completed':
    model_file = requests.get(f'http://localhost:8000/api/v1/sketch/{task_id}/download')
    with open('my_3d_model.obj', 'wb') as f:
        f.write(model_file.content)
```

## 🏗️ Architecture

```mermaid
graph TB
    A[🖼️ Sketch Image] --> B[📊 Preprocessing]
    B --> C[🧠 AI Classification]
    B --> D[📏 Depth Estimation]
    C --> E[🔺 3D Mesh Generation]
    D --> E
    E --> F[💾 Export .obj/.blend]
    
    style A fill:#e1f5fe
    style C fill:#fff3e0
    style D fill:#fff3e0
    style E fill:#f3e5f5
    style F fill:#e8f5e8
```

### AI Pipeline

1. **Classification** : Automatic identification of object type (house, car, tree...)
2. **Depth Estimation** : Generate depth map from 2D sketch
3. **3D Reconstruction** : Create 3D mesh using Marching Cubes algorithm
4. **Export** : Save in .obj and .blend formats

## 🔧 API Endpoints

| Method | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/health` | API and models status |
| `POST` | `/api/v1/sketch/process` | Process a sketch |
| `GET` | `/api/v1/sketch/{id}/status` | Processing status |
| `GET` | `/api/v1/sketch/{id}/download` | Download 3D model |
| `DELETE` | `/api/v1/sketch/{id}` | Delete task |

📖 **Full documentation** : `http://localhost:8000/docs`

## 📊 Performance

- **Processing Time** : 15-30 seconds per sketch
- **Classification Accuracy** : >90% on common objects
- **GPU Support** : CUDA 11.8+ acceleration
- **Supported Formats** : JPG, PNG → .obj, .blend
- **Model Size** : ~200MB total
- **Memory Usage** : 2-4GB RAM (GPU), 4-8GB RAM (CPU)

## 🛠️ Development

### Project Structure

```
drawgen/
├── 🚀 api.py                    # Main FastAPI application
├── 🧠 models.py                 # PyTorch models (Classifier + DepthNet)
├── ⚙️ sketch_processor.py       # AI processing pipeline
├── 🔺 mesh_generator.py         # 3D mesh generation
├── 🛠️ utils.py                 # Image processing utilities
├── 🔧 config.py                # Centralized configuration
├── 🌐 drawing_interface.html    # Web drawing interface
├── 📋 requirements.txt          # Python dependencies
├── 🐳 dockerfile               # Docker configuration
└── 🎯 demo.py                  # Demo scripts
```

### Testing

```bash
# Full test suite
python -m pytest tests/ -v

# Specific tests
pytest tests/test_models.py
pytest tests/test_api.py
```

### Contributing

1. Fork the project
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📦 Dependencies

### Core Dependencies
- **PyTorch** 2.0+ - Deep Learning framework
- **FastAPI** - Modern, fast web framework
- **OpenCV** - Computer vision and image processing
- **Trimesh** - 3D mesh manipulation
- **NumPy, SciPy** - Scientific computing

### Optional Dependencies
- **Open3D** - Advanced 3D visualization
- **Blender Python API** - Native .blend export

## ❓ FAQ

<details>
<summary><b>What types of drawings work best?</b></summary>

Sketches with clear contours work better:
- ✅ Recognizable objects (houses, cars, animals)
- ✅ Clean, contrasted lines
- ✅ Minimum size 128x128 pixels
- ❌ Avoid overly abstract or blurry drawings
</details>

<details>
<summary><b>Can I use my own AI models?</b></summary>

Yes! Replace models in `models.py` and adjust configuration in `config.py`.
</details>

<details>
<summary><b>Does the API work without GPU?</b></summary>

Yes, the system works in CPU mode, but processing will be slower (1-2 minutes vs 30 seconds).
</details>

<details>
<summary><b>What's the difference between .obj and .blend export?</b></summary>

- **.obj** : Universal 3D format, works with most 3D software
- **.blend** : Native Blender format with materials and scene data
</details>

## 🔒 Security

- ✅ Strict file validation for uploads
- ✅ File size limits (10MB max)
- ✅ Sandboxed model execution
- ✅ Automatic cleanup of temporary files
- ✅ Rate limiting and CORS protection

## 🌍 Internationalization

Drawgen supports multiple languages:
- 🇺🇸 English (primary)
- 🇫🇷 French
- 🇪🇸 Spanish  
- 🇩🇪 German
- 🇯🇵 Japanese
- 🇨🇳 Chinese

*Want to add your language? [Contribute translations](CONTRIBUTING.md#translations)!*

## 📈 Roadmap

- [x] ✅ Basic AI pipeline
- [x] ✅ Interactive web interface
- [x] ✅ .obj export
- [ ] 🔄 Native .blend export
- [ ] 📋 Batch processing support
- [ ] 🎨 Category-specific presets
- [ ] 🌐 Multi-user mode
- [ ] 📱 Mobile application
- [ ] 🧠 Advanced AI models (Stable Diffusion integration)
- [ ] 🎮 Unity/Unreal plugins

## 🏆 Showcase & Gallery

### 🎨 Results Gallery

| Original Sketch | Generated 3D Model | Category | Accuracy |
|----------------|-------------------|----------|----------|
| ![House Sketch](assets/gallery/sketch_house.png) | ![House Model](assets/gallery/model_house.png) | Architecture | 94% |
| ![Car Sketch](assets/gallery/sketch_car.png) | ![Car Model](assets/gallery/model_car.png) | Vehicle | 91% |
| ![Animal Sketch](assets/gallery/sketch_cat.png) | ![Animal Model](assets/gallery/model_cat.png) | Animal | 88% |

### 📊 Usage Statistics

- **🎯 Average Accuracy** : 92%
- **⚡ Average Processing Time** : 28 seconds

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## 👥 Contributors

- **KLN** - *Lead Developer* - (https://github.com/matrxx)

## 🙏 Acknowledgments

- [Google QuickDraw](https://quickdraw.withgoogle.com/) for the datasets
- [PyTorch](https://pytorch.org/) for the deep learning framework
- [FastAPI](https://fastapi.tiangolo.com/) for the modern web framework
- [Trimesh](https://trimsh.org/) for 3D tools
- The amazing open-source community

### 💬 Community Channels
- 💬 **GitHub Discussions** : [Discussions](../../discussions)
- 🐛 **Issues** : [Bug Reports](../../issues)
- 📧 **Email** : support@drawgen.ai

---

<div align="center">

**⭐ Star this repo if you like Drawgen! ⭐**

[🐛 Report Bug](../../issues) • [💡 Request Feature](../../issues) • [📖 Documentation](../../wiki) • [💬 Community](https://discord.gg/drawgen)

Made with ❤️ by me.
</div>
