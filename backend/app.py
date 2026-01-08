"""
SAM3 FastAPI Backend

This file:
- Fixes Python import paths so `sam3/` is discoverable
- Loads the SAM3 model once at startup using lifespan
- Exposes /health and /segment endpoints
- Works on Apple Silicon (MPS), CUDA GPUs, and CPU
- Preserves your original segmentation logic
"""

# ============================================================
# 1. PYTHON PATH FIX (CRITICAL)
# ============================================================
# When running `python backend/app.py`, Python's working directory
# is `backend/`, so it cannot see the repo root by default.
# This explicitly adds the project root so `sam3` and backend imports work.

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))   # backend/
project_root = os.path.dirname(current_dir)                # repo root

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ============================================================
# 2. STANDARD IMPORTS
# ============================================================

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Optional
import io
import numpy as np
from PIL import Image

# ============================================================
# 3. LOCAL PROJECT IMPORTS
# ============================================================

from backend.load_model import load_sam3_model
from backend.utils import mask_to_polygons

# ============================================================
# 4. FASTAPI LIFESPAN (MODEL LOAD / UNLOAD)
# ============================================================
# This ensures the SAM3 model is loaded ONCE at startup
# and reused across all requests.

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Initializing SAM 3 Model...")
    app.state.processor = load_sam3_model()
    yield
    print("🛑 Shutting down server...")
    del app.state.processor

# ============================================================
# 5. FASTAPI APP INITIALIZATION
# ============================================================

app = FastAPI(
    title="SAM 3 Local API",
    lifespan=lifespan
)

# ============================================================
# 6. CORS CONFIGURATION
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 7. HEALTH CHECK ENDPOINT
# ============================================================
# Used to verify:
# - Server is running
# - Model is loaded
# - Which device the model is on

@app.get("/health")
def health_check():
    processor = getattr(app.state, "processor", None)
    if processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "healthy",
        "device": str(processor.device),
    }

# ============================================================
# 8. SEGMENT ENDPOINT (YOUR ORIGINAL LOGIC, FIXED)
# ============================================================

@app.post("/segment")
async def segment(
    file: UploadFile = File(...),
    prompt_text: Optional[str] = Form(None),
    prompt_points: Optional[str] = Form(None),  # kept for future use
    prompt_box: Optional[str] = Form(None)      # kept for future use
):
    # --------------------------------------------------------
    # Validate model readiness
    # --------------------------------------------------------
    processor = getattr(app.state, "processor", None)
    if processor is None:
        raise HTTPException(status_code=503, detail="Model is still loading")

    if not any([prompt_text, prompt_points, prompt_box]):
        raise HTTPException(status_code=400, detail="At least one prompt is required")

    try:
        # ----------------------------------------------------
        # Load image from request
        # ----------------------------------------------------
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # ----------------------------------------------------
        # Run SAM3 inference
        # ----------------------------------------------------
        state = {}
        processor.set_image(image, state)

        if prompt_text:
            processor.set_text_prompt(prompt_text, state)

        masks = state.get("masks")
        scores = state.get("scores")

        if masks is None or len(masks) == 0:
            return {"status": "success", "results": []}

        # ----------------------------------------------------
        # SAFE tensor → CPU conversion (GPU/MPS compatible)
        # ----------------------------------------------------
        masks = masks.detach().cpu().numpy().astype(np.uint8)
        scores = (
            scores.detach().cpu().tolist()
            if scores is not None
            else [1.0] * len(masks)
        )

        label = prompt_text or "object"

        # ----------------------------------------------------
        # Convert masks to polygons
        # ----------------------------------------------------
        formatted_results = []
        for i, mask in enumerate(masks):
            polys = mask_to_polygons(mask)
            for poly in polys:
                formatted_results.append({
                    "label": label,
                    "confidence": float(scores[i]),
                    "points": poly
                })

        return {"status": "success", "results": formatted_results}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# 9. LOCAL DEV ENTRYPOINT
# ============================================================
# Allows running directly with:
#   python backend/app.py

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
