import torch
import numpy as np
import rasterio
import cv2
import torch.nn.functional as F

def preprocess_tif_files(tif_files, target_size=(256, 256)):
    """
    Given a list of 5 TIF files, preprocess them into a tensor of shape (5, 23, 256, 256).
    Also returns the binarized 23rd band of each file for display purposes.
    """
    processed_days = []
    band_23_images = []  # To store the binarized 23rd band of each file

    for file in tif_files:
        with rasterio.open(file) as src:
            img = src.read()  # (C, H, W)

        # Convert to tensor for resizing
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # (1, C, H, W)

        # Resize all bands together
        img_resized = F.interpolate(img_tensor, size=target_size, mode='bilinear', align_corners=False)

        # Remove batch dim → (C, H, W)
        img_resized = img_resized.squeeze(0).numpy()

        # Fix NaNs → replace with 0
        img_resized = np.nan_to_num(img_resized, nan=0.0)

        # Store the binarized 23rd band for display
        if img_resized.shape[0] >= 23:
            band_23 = img_resized[22]
        else:
            band_23 = img_resized[-1]  # Use last band if there aren't 23 bands
        
        # Binarize: positive pixels as 1, others as 0
        band_23_binary = np.where(band_23 > 0, 1, 0)
        band_23_images.append(band_23_binary)

        # Only keep first 23 bands
        processed_days.append(img_resized[:23])

    # Stack all 5 days → (5, 23, H, W)
    final_array = np.stack(processed_days, axis=0)

    return_tensor = torch.tensor(final_array, dtype=torch.float32)
    print(f"Returning tensor with shape: {return_tensor.shape}")
    return return_tensor, band_23_images, final_array


def median_filter_band(image, ksize=3):
    """
    Apply median filtering to a single-band grayscale image.
    """
    image = image.astype(np.uint8) if image.max() <= 255 else image
    return cv2.medianBlur(image, ksize)
