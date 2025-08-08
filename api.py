# api.py
"""API FastAPI for DrawGen"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from PIL import Image
import io
import uuid
import tempfile
from pathlib import Path
import logging
from typing import Dict, Any

from sketch_processor import SketchProcessor
from mesh_generator import MeshGenerator
from utils import validate_sketch_input, validate_mesh_output
from config import get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = get_config()

app = FastAPI(
    title="Sketch-to-3D API",
    description="API pour convertir des sketches en modèles 3D",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sketch_processor = SketchProcessor(config['models'])
mesh_generator = MeshGenerator(config['mesh_generation'])

processing_tasks = {}

class ProcessingStatus:
    def __init__(self):
        self.status = "queued"
        self.progress = 0
        self.message = ""
        self.result = None
        self.error = None

@app.get("/draw")
async def drawing_interface():
    """Drawing Interface"""
    return FileResponse("drawing_interface.html", media_type="text/html")

@app.get("/")
async def root():
    """Base endpoint"""
    return {"message": "DrawGen API", "status": "running", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Check API Health"""
    return {
        "status": "healthy",
        "components": {
            "sketch_processor": "loaded",
            "mesh_generator": "loaded"
        },
        "device": str(sketch_processor.device)
    }

@app.post("/api/v1/sketch/process")
async def process_sketch(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    mesh_name: str = "Generated_Mesh"
):
    """Principal Endpoint For Drawing Processing"""
    task_id = str(uuid.uuid4())
    
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="The file has to be an image")
        
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_array = np.array(image)
        
        if not validate_sketch_input(image_array):
            raise HTTPException(status_code=400, detail="Sketch image invalid")
        
        processing_tasks[task_id] = ProcessingStatus()
        
        background_tasks.add_task(
            process_sketch_background,
            task_id,
            image_array,
            mesh_name
        )
        
        return {
            "task_id": task_id,
            "status": "queued",
            "message": "Treatment started"
        }
        
    except Exception as e:
        logger.error(f"Initialization Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_sketch_background(task_id: str, image_array: np.ndarray, mesh_name: str):
    """Background Sketch Treatment"""
    task_status = processing_tasks[task_id]
    
    try:
        task_status.status = "processing"
        task_status.progress = 10
        task_status.message = "Analyse du sketch..."
        
        sketch_result = sketch_processor.process_sketch_to_3d(image_array)
        
        if sketch_result['status'] != 'success':
            raise Exception(f"Treatment Error: {sketch_result.get('error_message')}")
        
        task_status.progress = 50
        task_status.message = "3D Mesh Processing..."
        
        depth_map = sketch_result['depth_map'].squeeze()
        sketch_mask = (sketch_result['sketch_tensor'].squeeze() > 0.1).astype(np.float32)
        
        mesh = mesh_generator.generate_mesh_from_depth(
            depth_map,
            sketch_mask,
            sketch_result['classification']
        )
        
        mesh_validation = validate_mesh_output(mesh)
        
        stats = mesh_validation.get('stats', {})
        if stats:
            for key, value in stats.items():
                if hasattr(value, 'item'):  # if numpy scalar
                    stats[key] = value.item() if value is not None else None
                elif isinstance(value, (np.ndarray, np.generic)):
                    stats[key] = value.tolist() if value is not None else None
        
        task_status.progress = 80
        task_status.message = "Model Export..."
        
        temp_dir = Path(tempfile.gettempdir()) / "drawgen_results"
        temp_dir.mkdir(exist_ok=True)
        
        model_path = temp_dir / f"{task_id}.obj"
        mesh.export(str(model_path))
        
        task_status.status = "completed"
        task_status.progress = 100
        task_status.message = "Treatment ended"
        task_status.result = {
            "model_file_path": str(model_path),
            "mesh_stats": stats,
            "classification": {
                "class_id": int(sketch_result['classification']['class_id']),
                "confidence": float(sketch_result['classification']['confidence']),
                "class_name": str(sketch_result['classification']['class_name'])
            }
        }
        
        logger.info(f"Treatment ended for {task_id}")
        
    except Exception as e:
        task_status.status = "failed"
        task_status.error = str(e)
        task_status.message = f"Error: {str(e)}"
        logger.error(f"Treatment error {task_id}: {str(e)}")

@app.get("/api/v1/sketch/{task_id}/status")
async def get_task_status(task_id: str):
    """Fetch task status"""
    if task_id not in processing_tasks:
        raise HTTPException(status_code=404, detail="Task not founde")
    
    task_status = processing_tasks[task_id]
    
    response = {
        "task_id": task_id,
        "status": task_status.status,
        "progress": task_status.progress,
        "message": task_status.message
    }
    
    if task_status.status == "completed":
        response["result"] = task_status.result
    elif task_status.status == "failed":
        response["error"] = task_status.error
    
    return response

@app.get("/api/v1/sketch/{task_id}/download")
async def download_model_file(task_id: str):
    """Download the 3D generated file"""
    if task_id not in processing_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_status = processing_tasks[task_id]
    
    if task_status.status != "completed":
        raise HTTPException(status_code=400, detail="Treatment not ended")
    
    model_path = task_status.result["model_file_path"]
    
    if not Path(model_path).exists():
        raise HTTPException(status_code=404, detail="Model file not found")
    
    return FileResponse(
        path=model_path,
        filename=f"model_{task_id}.obj",
        media_type="application/octet-stream"
    )

@app.delete("/api/v1/sketch/{task_id}")
async def delete_task(task_id: str):
    """Delete a task"""
    if task_id not in processing_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_status = processing_tasks[task_id]
    
    if task_status.result and "model_file_path" in task_status.result:
        model_path = Path(task_status.result["model_file_path"])
        if model_path.exists():
            model_path.unlink()
    
    del processing_tasks[task_id]
    
    return {"message": f"Task {task_id} deleted"}

@app.get("/api/v1/models/status")
async def get_models_status():
    """AI Models status"""
    return {
        "sketch_classifier": {
            "loaded": True,
            "device": str(sketch_processor.device),
            "num_classes": config['models']['classifier']['num_classes']
        },
        "depth_estimator": {
            "loaded": True,
            "device": str(sketch_processor.device),
            "input_size": "512x512"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host=config['api']['host'],
        port=config['api']['port'],
        reload=True,
        log_level="info"

    )
