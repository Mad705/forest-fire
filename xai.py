# # xai.py
# import torch
# import numpy as np
# import cv2
# from PIL import Image
# import matplotlib.pyplot as plt
# import io

# def get_saliency_map(model, inputs, device):
#     model.train()
#     inputs = inputs.clone().to(device)
#     inputs.requires_grad_(True)

#     out = model(inputs)
#     score = out.sum()
#     model.zero_grad()
#     score.backward()

#     sal = inputs.grad.abs()
#     sal = sal.mean(dim=2).sum(dim=1).squeeze(0).cpu().numpy()
#     sal = np.maximum(sal, 0)
#     sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
#     return np.uint8(255 * sal)

# def get_band_importance(model, inputs, device, n_steps=20):
#     model.train()
#     inputs = inputs.clone().to(device)
#     inputs.requires_grad_(True)
#     baseline = torch.zeros_like(inputs)

#     attr = torch.zeros_like(inputs)
#     for alpha in np.linspace(0, 1, n_steps):
#         interp = baseline + alpha * (inputs - baseline)
#         interp.requires_grad_(True)
#         out = model(interp)
#         score = torch.sigmoid(out).mean()
#         grad = torch.autograd.grad(score, interp, retain_graph=(alpha < 1))[0]
#         attr += grad * (inputs - baseline)

#     imp = attr.abs().mean(dim=(0,1,3,4)).detach().cpu().numpy()
#     return imp / (imp.sum() + 1e-8)

# def overlay_heatmap(mask_np, heatmap_uint8):
#     mask_np = (mask_np * 255).astype(np.uint8)
#     mask_3ch = np.stack([mask_np] * 3, axis=-1)
#     heat = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
#     heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
#     overlay = cv2.addWeighted(mask_3ch, 0.6, heat, 0.4, 0)
#     return Image.fromarray(overlay)

# def plot_band_importance(imp):
#     bands = [f"B{i+1}" for i in range(23)]
#     fig, ax = plt.subplots(figsize=(8, 5))
#     ax.barh(bands, imp, color='orange')
#     ax.set_xlabel('Importance')
#     ax.set_title('Band Importance')
#     plt.tight_layout()
#     buf = io.BytesIO()
#     plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
#     buf.seek(0); plt.close()
#     return Image.open(buf)



# xai.py
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import io

# ===============================================
# 1. SALIENCY MAP (UNCHANGED – STILL WORKS GREAT)
# ===============================================
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


# ===============================================
# 2. BAND IMPORTANCE – NEW LOGIC (YOUR REQUEST)
# ===============================================
def get_band_importance(model, inputs, device):
    """
    Computes band importance using a single backward pass on the mean prediction.
    inputs: (1, 5, 23, 256, 256) – one sample
    """
    model.eval()  # Use eval mode – no dropout/BN issues
    inputs = inputs.clone().to(device)
    inputs.requires_grad_(True)

    # Forward: get prediction
    pred = model(inputs)           # (1, 1, 256, 256)
    score = pred.mean()            # Reduce to scalar

    # Backward: compute gradients
    model.zero_grad()
    score.backward()

    # Get gradients w.r.t. input
    grads = inputs.grad            # (1, 5, 23, 256, 256)
    grads = grads.squeeze(0)       # (5, 23, 256, 256)

    # Absolute mean over time + spatial dims → (23,)
    band_importance = grads.abs().mean(dim=(0, 2, 3)).cpu().numpy()

    # Normalize to percentages
    total = band_importance.sum()
    if total > 0:
        band_importance = band_importance / total
    else:
        band_importance = np.ones(23) / 23  # fallback

    return band_importance


# ===============================================
# 3. OVERLAY HEATMAP (UNCHANGED)
# ===============================================
def overlay_heatmap(mask_np, heatmap_uint8):
    mask_np = (mask_np * 255).astype(np.uint8)
    mask_3ch = np.stack([mask_np] * 3, axis=-1)
    heat = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(mask_3ch, 0.6, heat, 0.4, 0)
    return Image.fromarray(overlay)


# ===============================================
# 4. PLOT BAND IMPORTANCE (UPDATED TO MATCH NEW LOGIC)
# ===============================================
def plot_band_importance(imp):
    bands = [f"B{i+1}" for i in range(23)]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(bands, imp, color='orange')
    ax.set_xlabel('Relative Importance')
    ax.set_title('Band Importance (Gradient-based)')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return Image.open(buf)