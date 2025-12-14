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
            "/health": "Health check"
        }
    }


@app.get("/health")
def health():
    """Health check for Railway."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


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
