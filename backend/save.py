import json
import os
import time

def save_masks(image_name, masks, scores, boxes, output_dir="../data/labels"):
    """
    Saves the segmentation results to a JSON file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Create valid filename
    base_name = os.path.splitext(os.path.basename(image_name))[0]
    filename = f"{base_name}_labels.json"
    filepath = os.path.join(output_dir, filename)
    
    data = {
        "image": image_name,
        "timestamp": time.time(),
        "annotations": []
    }
    
    # We assume masks, scores, boxes are lists of same length
    # Note: masks here might be RLE or ignored if not passed back fully from API yet
    # For now we save boxes and scores.
    
    for i in range(len(scores)):
        ann = {
            "id": i,
            "score": scores[i],
            "box": boxes[i] if i < len(boxes) else None,
            # "mask_rle": ... # To be implemented
        }
        data["annotations"].append(ann)
        
    try:
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved labels to {filepath}")
        return filepath
    except Exception as e:
        print(f"Error saving labels: {e}")
        return None
