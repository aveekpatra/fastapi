"""
FastAPI + RAG Pipeline for Railway
Runs vectorization as a background task with status endpoint.
"""

import asyncio
import os
import threading
import time
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel

# Import the vectorization pipeline
# We'll import functions from vectorize_new_fast.py

app = FastAPI(
    title="Czech Court Decisions RAG Pipeline",
    description="API for vectorizing Czech court decisions from justice.cz",
    version="1.0.0"
)

# Global state for pipeline status
pipeline_state = {
    "status": "idle",
    "started_at": None,
    "last_update": None,
    "stats": {},
    "error": None
}

# Lock for thread safety
state_lock = threading.Lock()


def get_volume_path():
    """Get volume path at runtime (after Railway mounts it)."""
    env_path = os.getenv("RAILWAY_VOLUME_MOUNT_PATH")
    if env_path and os.path.exists(env_path):
        return env_path
    if os.path.exists("/fastapi-volume"):
        return "/fastapi-volume"
    return "."


class PipelineStatus(BaseModel):
    status: str
    started_at: Optional[str]
    last_update: Optional[str]
    stats: dict
    error: Optional[str]


class StartResponse(BaseModel):
    message: str
    status: str


@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "service": "Czech Court Decisions RAG Pipeline",
        "status": "running",
        "endpoints": {
            "/status": "Get pipeline status",
            "/start": "Start vectorization (POST)",
            "/stop": "Stop vectorization (POST)",
            "/logs": "Get log file (GET, ?lines=100&log_type=pipeline|errors)",
            "/checkpoint": "View checkpoint status (GET)",
            "/clear-checkpoint": "Clear checkpoint to restart (POST)",
            "/volume-info": "Check volume mount status (GET)",
            "/health": "Health check"
        }
    }


@app.get("/health")
def health():
    """Health check for Railway."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.get("/volume-info")
def volume_info():
    """Check volume mount status for debugging."""
    env_path = os.getenv("RAILWAY_VOLUME_MOUNT_PATH")
    detected_path = get_volume_path()
    
    return {
        "env_RAILWAY_VOLUME_MOUNT_PATH": env_path,
        "detected_volume_path": detected_path,
        "fastapi_volume_exists": os.path.exists("/fastapi-volume"),
        "env_path_exists": os.path.exists(env_path) if env_path else False,
        "files_in_volume": os.listdir(detected_path) if os.path.exists(detected_path) else []
    }


@app.get("/status", response_model=PipelineStatus)
def get_status():
    """Get current pipeline status."""
    with state_lock:
        return PipelineStatus(**pipeline_state)


@app.post("/start", response_model=StartResponse)
def start_pipeline(background_tasks: BackgroundTasks):
    """Start the vectorization pipeline."""
    with state_lock:
        if pipeline_state["status"] == "running":
            raise HTTPException(status_code=400, detail="Pipeline already running")
        
        pipeline_state["status"] = "starting"
        pipeline_state["started_at"] = datetime.utcnow().isoformat()
        pipeline_state["error"] = None
    
    background_tasks.add_task(run_pipeline)
    
    return StartResponse(
        message="Pipeline starting in background",
        status="starting"
    )


@app.post("/stop")
def stop_pipeline():
    """Request pipeline stop."""
    with state_lock:
        if pipeline_state["status"] != "running":
            raise HTTPException(status_code=400, detail="Pipeline not running")
        pipeline_state["status"] = "stopping"
    
    return {"message": "Stop requested", "status": "stopping"}


@app.get("/logs")
def get_logs(lines: int = 100, log_type: str = "pipeline"):
    """Get recent log lines from volume storage."""
    volume_path = get_volume_path()
    log_file = "rag_pipeline.log" if log_type == "pipeline" else "rag_errors.log"
    log_path = os.path.join(volume_path, log_file)
    
    try:
        if not os.path.exists(log_path):
            return {"error": f"Log file not found at {log_path}", "volume_path": VOLUME_PATH}
        
        with open(log_path, "r", encoding="utf-8") as f:
            content = f.readlines()
            return {
                "file": log_path,
                "total_lines": len(content),
                "showing_last": min(lines, len(content)),
                "logs": content[-lines:]
            }
    except Exception as e:
        return {"error": str(e)}


@app.get("/checkpoint")
def get_checkpoint():
    """Get current checkpoint data (processed days)."""
    volume_path = get_volume_path()
    checkpoint_path = os.path.join(volume_path, "rag_checkpoint.json")
    
    try:
        if not os.path.exists(checkpoint_path):
            return {
                "exists": False,
                "path": checkpoint_path,
                "message": "No checkpoint file - pipeline will start from beginning"
            }
        
        import json
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        processed_days = data.get("processed_days", [])
        return {
            "exists": True,
            "path": checkpoint_path,
            "total_days_processed": len(processed_days),
            "last_processed": max(processed_days) if processed_days else None,
            "first_processed": min(processed_days) if processed_days else None,
            "total_chunks": data.get("total_chunks", 0),
            "sample_days": processed_days[-10:] if len(processed_days) > 10 else processed_days
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/clear-checkpoint")
def clear_checkpoint():
    """Clear checkpoint to restart from beginning. USE WITH CAUTION."""
    volume_path = get_volume_path()
    checkpoint_path = os.path.join(volume_path, "rag_checkpoint.json")
    
    try:
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            return {"message": "Checkpoint cleared - pipeline will start from beginning"}
        else:
            return {"message": "No checkpoint file exists"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def run_pipeline():
    """Run the vectorization pipeline."""
    global pipeline_state
    
    try:
        with state_lock:
            pipeline_state["status"] = "running"
        
        # Import here to avoid loading model at startup
        from vectorize_new_fast import RAGPipeline
        
        # Create and run pipeline
        pipeline = RAGPipeline()
        
        # Hook into stats updates
        original_process = pipeline.process_decisions_batch
        
        def wrapped_process(decisions):
            result = original_process(decisions)
            with state_lock:
                pipeline_state["stats"] = pipeline.stats.copy()
                pipeline_state["last_update"] = datetime.utcnow().isoformat()
                
                # Check for stop request
                if pipeline_state["status"] == "stopping":
                    raise KeyboardInterrupt("Stop requested")
            return result
        
        pipeline.process_decisions_batch = wrapped_process
        
        # Run
        pipeline.run()
        
        with state_lock:
            pipeline_state["status"] = "completed"
            pipeline_state["stats"] = pipeline.stats.copy()
            
    except KeyboardInterrupt:
        with state_lock:
            pipeline_state["status"] = "stopped"
            
    except Exception as e:
        with state_lock:
            pipeline_state["status"] = "error"
            pipeline_state["error"] = str(e)


# Auto-start pipeline on Railway (optional)
@app.on_event("startup")
async def startup_event():
    """Optionally auto-start pipeline on deployment."""
    auto_start = os.getenv("AUTO_START_PIPELINE", "false").lower() == "true"
    
    if auto_start:
        print("Auto-starting pipeline...")
        asyncio.create_task(asyncio.to_thread(run_pipeline))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
