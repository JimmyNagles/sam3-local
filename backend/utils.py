import cv2
import numpy as np

def mask_to_polygons(mask):
    """
    Converts a binary mask to a list of polygons.
    Ported from sam3.agent.helpers.visualizer.GenericMask.mask_to_polygons
    """
    # cv2.RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
    # hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
    # Internal contours (holes) are placed in hierarchy-2.
    # cv2.CHAIN_APPROX_NONE flag gets vertices of polygons from contours.
    mask = np.ascontiguousarray(mask)  # some versions of cv2 does not support incontiguous arr
    
    # Ensure mask is 2D (H, W)
    if mask.ndim == 3:
        mask = mask.squeeze()
    
    # If still not 2D, raise error or try to fix
    if mask.ndim != 2:
        # Fallback: if somehow it has extra dims of size 1
        mask = mask.reshape(mask.shape[-2], mask.shape[-1])
        
    res = cv2.findContours(
        mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )
    
    # Handle different OpenCV versions returning (contours, hierarchy) vs (image, contours, hierarchy)
    if len(res) == 3:
        _, contours, hierarchy = res
    else:
        contours, hierarchy = res
        
    if hierarchy is None:  # empty mask
        return []
    
    # hierarchy shape is (1, N, 4)
    # The 4th element is the parent index. -1 means no parent (outer contour).
    # We want to filter for external contours or handle holes?
    # SAM3 logic:
    # has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
    # res = res[-2] (which is contours)
    # res = [x.flatten() for x in res]
    
    polygons = []
    for contour in contours:
        # contour shape is (N, 1, 2)
        if len(contour) >= 3: 
             # Reshape to (N, 2) and convert to list of lists [[x,y], [x,y], ...]
             points = contour.reshape(-1, 2).tolist()
             polygons.append(points)
            
    return polygons
