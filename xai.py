# xai.py
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import io

def get_saliency_map(model, inputs, device):
    model.train()
    inputs = inputs.clone().to(device)
    inputs.requires_grad_(True)

    out = model(inputs)
    score = out.sum()
    model.zero_grad()
    score.backward()

    sal = inputs.grad.abs()
    sal = sal.mean(dim=2).sum(dim=1).squeeze(0).cpu().numpy()
    sal = np.maximum(sal, 0)
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    return np.uint8(255 * sal)

def get_band_importance(model, inputs, device, n_steps=20):
    model.train()
    inputs = inputs.clone().to(device)
    inputs.requires_grad_(True)
    baseline = torch.zeros_like(inputs)

    attr = torch.zeros_like(inputs)
    for alpha in np.linspace(0, 1, n_steps):
        interp = baseline + alpha * (inputs - baseline)
        interp.requires_grad_(True)
        out = model(interp)
        score = torch.sigmoid(out).mean()
        grad = torch.autograd.grad(score, interp, retain_graph=(alpha < 1))[0]
        attr += grad * (inputs - baseline)

    imp = attr.abs().mean(dim=(0,1,3,4)).detach().cpu().numpy()
    return imp / (imp.sum() + 1e-8)

def overlay_heatmap(mask_np, heatmap_uint8):
    mask_np = (mask_np * 255).astype(np.uint8)
    mask_3ch = np.stack([mask_np] * 3, axis=-1)
    heat = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(mask_3ch, 0.6, heat, 0.4, 0)
    return Image.fromarray(overlay)

def plot_band_importance(imp):
    bands = [f"B{i+1}" for i in range(23)]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(bands, imp, color='orange')
    ax.set_xlabel('Importance')
    ax.set_title('Band Importance')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0); plt.close()
    return Image.open(buf)