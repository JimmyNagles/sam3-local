# SAM 3 Local Workstation 🚀

A local, privacy-first workstation for the Segment Anything Model 3 (SAM 3).
Designed for **MacBook Air M4** (Apple Silicon) with CPU-optimized inference.

## 🌟 Features
- **Local Inference**: Run SAM 3 entirely offline.
- **Apple Silicon Optimized**: Tuned for stability on M-series chips (CPU mode enforced for reliability).
- **FastAPI Backend**: Robust API for segmentation requests.
- **Modern Frontend**: React/Next.js interface for easy labeling (Coming Soon).

## 🛠️ Prerequisites
- **Python 3.10+**
- **Node.js 18+** (for frontend)
- **30GB+ Free Disk Space**

## 📂 Project Structure
```bash
sam3-local/
├── backend/          # Python FastAPI Server
│   ├── app.py        # API Endpoints
│   ├── load_model.py # Model Logic & Monkey-patches
│   └── weights/      # Model Checkpoints
├── frontend/         # Next.js Application
└── data/             # Your local datasets
```

## 🚀 Getting Started

### 1. Quick Setup (Recommended)

Run the included setup script to automatically create the virtual environment and install all dependencies:

```bash
chmod +x setup_dev.sh
./setup_dev.sh
```

### 2. Manual Setup
If you prefer manual setup:
```bash
# Activate existing venv (if in parent dir) or create new one
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision fastapi uvicorn python-multipart opencv-python pycocotools psutil huggingface_hub pil
```

### 2. Running the Server

Start the inference backend:

```bash
cd sam3-local/backend
python app.py
```
> The server will start on `http://localhost:8000`.
> *Note: On the first run, the model weights (~3.2GB) will be automatically downloaded from HuggingFace. This may take a few minutes.*

### 3. Model Weights (Optional Manual Setup)
If you want to skip the download (e.g., copying from a USB drive):
1.  Create the folder `backend/weights/`.
2.  Place the `sam3.pt` file inside it.
3.  The path should be: `backend/weights/sam3.pt`.

### 3. Usage API

**Endpoint**: `POST /segment`

**Example `curl` command**:
```bash
curl -X POST "http://localhost:8000/segment" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/image.jpg" \
  -F "prompt_text=cat"
```

---
**Status**: 🚧 Under Construction (Backend Active, Frontend In-Progress)
