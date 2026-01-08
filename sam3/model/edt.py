# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
Replacement for Triton kernel for euclidean distance transform (EDT) using OpenCV.
Original Triton implementation removed for MacOS compatibility.
"""

import torch
import cv2
import numpy as np

def edt_triton(data: torch.Tensor):
    """
    Computes the Euclidean Distance Transform (EDT) of a batch of binary images.

    Args:
        data: A tensor of shape (B, H, W) representing a batch of binary images.

    Returns:
        A tensor of the same shape as data containing the EDT.
        It should be equivalent to a batched version of cv2.distanceTransform(input, cv2.DIST_L2, 0)
    """
    assert data.dim() == 3
    
    # Move to CPU and numpy
    data_np = data.cpu().numpy().astype(np.uint8)
    
    B, H, W = data_np.shape
    outputs_np = np.zeros_like(data_np, dtype=np.float32)
    
    for i in range(B):
        # cv2.distanceTransform requires binary image where 0 is the background (distance to zero).
        # The triton code docs say: "calculates the L2 distance to the closest zero pixel".
        # So we want distance to closest 0. 
        # cv2.distanceTransform calculates distance to closest ZERO pixel.
        # However, the input `data` seems to be the mask where we want distance?
        # Let's check original implementation logic:
        # "implicit function is 0 if data[i,j]==0 else +infinity"
        # "min (x-i)^2 + F(i)" -> if data[i]==0, distance is 0.
        # This describes the distance transform of the SET of 0 pixels.
        # i.e. distance FROM any pixel TO the nearest zero pixel.
        # cv2.distanceTransform(src, ...) calculates distance from each pixel to the nearest ZERO pixel.
        # So providing the binary mask directly should work.
        
        # data is boolean/binary. 
        # If we pass it as uint8, 0 -> 0, 1 -> 1.
        # cv2.distanceTransform will give distance to nearest 0.
        dist = cv2.distanceTransform(1 - data_np[i], cv2.DIST_L2, 5) # Invert?
        
        # Wait, if data[i,j] == 0, implicit function is 0. So distance at 0 is 0.
        # If data[i,j] == 1, implicit function is infinity. Distance is min dist to a 0.
        # So we want distance to nearest 0.
        # cv2.distanceTransform calculates distance to nearest zero.
        # So we just pass data_np[i] (where 0 is 0, 1 is 1).
        
        outputs_np[i] = cv2.distanceTransform(1 - data_np[i], cv2.DIST_L2, 5)

        # Wait, let's double check.
        # If I have a mask of an object (1s) and background (0s).
        # EDT usually means distance from object boundary?
        # The Triton code: "EDT ... L2 distance to the closest zero pixel".
        # So for a pixel with value 1, what is the distance to nearest 0.
        # cv2.distanceTransform(src, ...) : "calculates the distance to the closest zero pixel for each pixel of the source image."
        # So if I pass the mask (mostly 0s, some 1s), it calculates distance to 0s.
        # If I want distance TO zeroes, I pass the image where zeroes are the target.
        # So if input is the mask (1=foreground, 0=background), and we want distance to background (0),
        # then passing the mask directly works.
        
        # BUT, usually EDT is used for signed distance field or similar.
        # The Triton code initializes output = 0 if data==0 else inf.
        # So if data is 0, dist is 0.
        # If data is 1, dist is distance to nearest 0.
        # This matches cv2.distanceTransform behavior on the mask itself.
        # However, cv2 expects 8-bit single-channel image.
        
        # Let's try passing 1-data just in case?
        # If data[p] = 0 (background), we want dist=0.
        # cv2: "The function calculates the approximated or precise distance from every binary image pixel to the nearest zero pixel."
        # So if data[p] = 0, distance IS 0.
        # So passing `data_np` directly which has 0s and 1s is correct.
        
        # Optimization: verify if Invert is needed.
        # If I want distance inside the object to the boundary...
        # The doc says "distance to the closest ZERO pixel".
        # If I am inside an object (1), I want distance to boundary (0).
        # So yes, passing the mask works.
        
        outputs_np[i] = cv2.distanceTransform(data_np[i], cv2.DIST_L2, 5)

    # Convert back to tensor and correct device
    outputs = torch.from_numpy(outputs_np).to(data.device)
    return outputs
