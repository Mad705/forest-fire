import torch
import numpy as np
from PIL import Image

def final_run(model, inputs, device):
    model.eval()
    with torch.no_grad():
        inputs = inputs.to(device)
        outputs = model(inputs)

        # Debug outputs
        print(f"Raw outputs range: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")
        
        # --- Probability map (before thresholding, apply sigmoid)
        prob_map = torch.sigmoid(outputs)
        print(f"Probability map range: [{prob_map.min().item():.4f}, {prob_map.max().item():.4f}]")

        # --- Binary mask (thresholded) - try different threshold
        predicted_mask = (torch.sigmoid(outputs) > 0.35).float() 
        num_fire_pixels = int((predicted_mask > 0).sum().item())
        print(f"Number of fire pixels: {num_fire_pixels}")

        # --- Convert tensors to images ---
        prob_map = prob_map.squeeze().cpu().numpy()
        predicted_mask = predicted_mask.squeeze().cpu().numpy()

        print(f"Prob map numpy range: [{prob_map.min():.4f}, {prob_map.max():.4f}]")
        print(f"Mask numpy range: [{predicted_mask.min():.4f}, {predicted_mask.max():.4f}]")

        # Normalize prob_map to [0,255]
        prob_map = (prob_map * 255).astype(np.uint8)

        # Convert mask to [0,255] (binary)
        predicted_mask = (predicted_mask * 255).astype(np.uint8)

        # Convert to PIL images
        prob_img = Image.fromarray(prob_map)
        mask_img = Image.fromarray(predicted_mask)

        return mask_img, prob_img