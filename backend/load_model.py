
import torch
import os
import sys

# --- Path Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

def load_sam3_model():
    """
    Detects GPU and loads SAM 3 to the correct device.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Initializing SAM 3 on: {device}")
    
    weights_dir = os.path.join(current_dir, "weights")
    ckpt_path = os.path.join(weights_dir, "sam3.pt")
    
    if not os.path.exists(ckpt_path):
        print(f"Local weights not found at {ckpt_path}. Model will download from HF...")
        ckpt_path = None 
    else:
        print(f"Found local weights at {ckpt_path}")

    try:
        model = build_sam3_image_model(
            checkpoint_path=ckpt_path,
            load_from_HF=(ckpt_path is None),
            device=device
        )
        
        model = model.to(device)
        model.eval()
        
        processor = Sam3Processor(model, device=device)
        print(f"✅ SAM 3 model loaded successfully on {device}")
        return processor

    except Exception as e:
        import traceback
        print("❌ Error loading the model:")
        traceback.print_exc()
        raise RuntimeError(f"Failed to load SAM 3 model: {e}")

