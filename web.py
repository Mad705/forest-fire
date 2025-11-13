# import streamlit as st
# import base64
# import torch
# import numpy as np
# from PIL import Image
# from preprocess import preprocess_tif_files
# from model import Load_model
# from result import final_run
# from streamlit_image_coordinates import streamlit_image_coordinates
# from coords import get_coords

# IMG_SIZE = 256
# RES = 375  # meters per pixel

# # --- Page Config ---
# st.set_page_config(page_title="Forest Fire TIF Uploader", layout="wide")

# # --- Encode local background image ---
# def get_base64(file_path):
#     with open(file_path, "rb") as f:
#         data = f.read()
#     return base64.b64encode(data).decode()

# img_base64 = get_base64("image2.jpeg")

# # --- Custom CSS ---
# page_bg = f"""
# <style>
# /* Background */
# .stApp {{
#     background: url("data:image/jpeg;base64,{img_base64}") no-repeat center center fixed;
#     background-size: cover;
# }}
# #MainMenu {{visibility: hidden;}}
# header {{visibility: hidden;}}
# .navbar {{
#     position: fixed;
#     top: 0;
#     left: 0;
#     width: 100%;
#     background: #1e211f;
#     padding: 15px 30px;
#     display: flex;
#     justify-content: space-between;
#     align-items: center;
#     color: #ffcc00;
#     font-size: 20px;
#     font-family: 'Inter', 'Poppins', 'Segoe UI', 'Roboto', sans-serif;
#     font-weight: bold;
#     letter-spacing: 0.5px;
#     z-index: 1000;
#     text-shadow: 0 0 5px #ff6600, 0 0 10px #ff3300, 0 0 20px #ff0000;
#     box-shadow: 0 4px 10px rgba(0,0,0,0.5);
# }}
# .day-label {{
#     text-align: center;
#     font-size: 20px;
#     font-weight: bold;
#     color: #ffcc00;
#     text-shadow: 0 0 5px #ff6600, 0 0 10px #ff3300, 0 0 20px #ff0000;
#     margin-bottom: 0px;
# }}
# .file-upload-container {{
#     display: flex;
#     justify-content: center;
#     gap: 20px;
#     margin-top: 20px;
#     flex-wrap: wrap;
# }}
# .band-display-container {{
#     display: flex;
#     justify-content: center;
#     gap: 10px;
#     margin: 20px 0;
#     flex-wrap: wrap;
# }}
# .band-image {{
#     border: 2px solid #ffcc00;
#     border-radius: 5px;
#     padding: 5px;
#     background-color: rgba(0, 0, 0, 0.7);
# }}
# </style>
# """
# st.markdown(page_bg, unsafe_allow_html=True)

# # --- Navbar ---
# st.markdown(
#     """
#     <div class="navbar">
#         <div> Forest Fire Spread Prediction</div>
#     </div>
#     """,
#     unsafe_allow_html=True
# )


# st.markdown('<div class="file-upload-container">', unsafe_allow_html=True)

# uploaded_files = st.file_uploader(
#     "Upload 5 TIF files (Day 1–Day 5)",
#     type=["tif"],
#     accept_multiple_files=True
# )

# st.markdown('</div>', unsafe_allow_html=True)

# # Validate file count
# if uploaded_files:
#     if len(uploaded_files) != 5:
#         st.error("⚠️ Please upload exactly 5 TIF files.")
#     else:
#         st.success(" All 5 TIF files uploaded successfully!")
# st.markdown('</div>', unsafe_allow_html=True)

# # --- Initialize session state for outputs ---
# if "fire_mask" not in st.session_state:
#     st.session_state.fire_mask = None
# if "fire_mask_clean" not in st.session_state:
#     st.session_state.fire_mask_clean = None
# if "prob_map" not in st.session_state:
#     st.session_state.prob_map = None
# if "band_images" not in st.session_state:
#     st.session_state.band_images = None
# if "coordinates" not in st.session_state:
#     st.session_state.coordinates = {"lat": None, "long": None}

# # --- Model processing function ---
# def process_files(files):
#     inputs, band_images = preprocess_tif_files(files)
#     print(f"Preprocessed input shape: {inputs.shape}")  # Should be (5, 23, 256, 256)
    
#     # Add batch dimension and ensure correct shape
#     inputs = inputs.unsqueeze(0)  # Now (1, 5, 23, 256, 256)
#     print(f"Input to model shape: {inputs.shape}")
    
#     model, device = Load_model("50_2020.pth")
    
#     # Debug: Check model output range
#     with torch.no_grad():
#         outputs = model(inputs.to(device))
#         print(f"Model output range: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")
#         print(f"Model output shape: {outputs.shape}")
    
#     # Store band images in session state for display
#     st.session_state.band_images = band_images
    
#     return final_run(model, inputs, device)

# # --- Process button ---
# if (len(uploaded_files) == 5):
#     if st.button("🚀 Process Files"):
#         with st.spinner("Processing files..."):
#             # Get coordinates and store in session state
#             lat, long = get_coords(uploaded_files[0])
#             st.session_state.coordinates = {"lat": lat, "long": long}
            
#             fire_mask, prob_map = process_files(uploaded_files)
#             st.session_state.fire_mask = fire_mask
#             st.session_state.prob_map = prob_map
            
#         st.success("🎉 Processing complete!")
        
#         # Show coordinates if available
#         if st.session_state.coordinates["lat"] and st.session_state.coordinates["long"]:
#             st.info(f"📍 Top-left coordinates: {st.session_state.coordinates['lat']:.6f}, {st.session_state.coordinates['long']:.6f}")
#         else:
#             st.warning("⚠️ Could not extract coordinates from the TIFF file")

# # --- Display binarized 23rd bands horizontally ---
# if st.session_state.band_images:
#     st.markdown("---")
#     st.markdown("### 📊 Binarized 23rd Band of Each TIF File (Positive Pixels = 1)")
    
#     # Create a container for horizontal display
#     st.markdown('<div class="band-display-container">', unsafe_allow_html=True)
    
#     cols = st.columns(5)
#     for i, band_img in enumerate(st.session_state.band_images):
#         with cols[i]:
#             # Convert binary array (0s and 1s) to image
#             # Scale to 0-255 range for display
#             band_display = band_img * 255
#             band_pil = Image.fromarray(band_display.astype(np.uint8))
#             st.image(band_pil, caption=f"Day {i+1} - Band 23 (Binary)", use_container_width=True)
    
#     st.markdown('</div>', unsafe_allow_html=True)

# # --- Display outputs side by side ---
# if st.session_state.fire_mask and st.session_state.prob_map:
#     st.markdown("---")
#     st.markdown("### 🔥 Model Outputs")
#     col1, col2 = st.columns(2)
#     with col1:
#         st.image(st.session_state.fire_mask, caption="Fire Mask", use_container_width=True)
#     with col2:
#         st.image(st.session_state.prob_map, caption="Probability Map", use_container_width=True)
    
#     st.markdown(f"# Coordinates Finder ")
    
#     # Use session state coordinates with fallback values
#     default_lat = st.session_state.coordinates["lat"] if st.session_state.coordinates["lat"] else 0.0
#     default_long = st.session_state.coordinates["long"] if st.session_state.coordinates["long"] else 0.0
    
#     lat0 = st.number_input("Enter top-left latitude:", value=float(default_lat))
#     lon0 = st.number_input("Enter top-left longitude:", value=float(default_long))

#     if isinstance(st.session_state.fire_mask, Image.Image):
#         fire_mask = np.array(st.session_state.fire_mask)
#     else:
#         fire_mask = st.session_state.fire_mask

#     # Convert band image list to NumPy if any are PIL images
#     band_images = [
#         np.array(img) if isinstance(img, Image.Image) else img
#         for img in st.session_state.band_images
#     ]

#     # Get the last day's band (binary)
#     last_band = band_images[-1]

#     # Ensure boolean arrays
#     fire_mask = fire_mask.astype(bool)
#     last_band = last_band.astype(bool)

#     # Remove overlapping 1s (pixels that are 1 in both)
#     fire_mask_cleaned = np.logical_and(fire_mask, np.logical_not(last_band)).astype(np.uint8)
#     # Display image and capture click coordinates
#     st.session_state.fire_mask_clean = Image.fromarray(fire_mask_cleaned * 255)

#     coords = streamlit_image_coordinates(st.session_state.fire_mask_clean, width=600)

#     if coords is not None:
#         i, j = coords["y"], coords["x"]   # row=i, col=j

#         # Only calculate if we have valid coordinates
#         if lat0 is not None and lon0 is not None:
#             # Convert to lat/lon
#             lati = lat0 - (i * RES / 111320)
#             longi = lon0 + (j * RES / (111320 * np.cos(np.radians(lat0))))

#             st.markdown(f" Pixel: ({i}, {j})")
#             st.markdown(f"**Latitude:** {lati:.6f}   |   **Longitude:** {longi:.6f}")
#         else:
#             st.error("Please enter valid latitude and longitude values")
    
#     # Clear button
#     if st.button("🧹 Clear Outputs"):
#         st.session_state.fire_mask = None
#         st.session_state.prob_map = None
#         st.session_state.band_images = None
#         st.session_state.coordinates = {"lat": None, "long": None}
#         st.rerun()  # Reset the page to initial state
# else:
#     st.warning("Upload all 5 TIF files and click 'Process' to see outputs.")

# # --- About Us Section ---
# st.markdown("---")
# st.markdown("### About Us", unsafe_allow_html=True)
# st.write("We are building a Forest Fire Spread Analysis tool.")
















# # web.py
# import streamlit as st
# import base64
# import torch
# import numpy as np
# from PIL import Image
# from preprocess import preprocess_tif_files
# from model import Load_model
# from result import final_run          # returns 5 items now
# from streamlit_image_coordinates import streamlit_image_coordinates
# from coords import get_coords

# IMG_SIZE = 256
# RES = 375  # meters per pixel

# # ----------------------------------------------------------------------
# # 1. Page config + background
# # ----------------------------------------------------------------------
# st.set_page_config(page_title="Forest Fire TIF Uploader", layout="wide")

# def get_base64(file_path):
#     with open(file_path, "rb") as f:
#         data = f.read()
#     return base64.b64encode(data).decode()

# img_base64 = get_base64("image2.jpeg")

# page_bg = f"""
# <style>
# .stApp {{
#     background: url("data:image/jpeg;base64,{img_base64}") no-repeat center center fixed;
#     background-size: cover;
# }}
# #MainMenu {{visibility: hidden;}}
# header {{visibility: hidden;}}
# .navbar {{
#     position: fixed; top:0; left:0; width:100%; background:#1e211f;
#     padding:15px 30px; display:flex; justify-content:space-between;
#     align-items:center; color:#ffcc00; font-size:20px;
#     font-family:'Inter','Poppins','Segoe UI','Roboto',sans-serif;
#     font-weight:bold; letter-spacing:0.5px; z-index:1000;
#     text-shadow:0 0 5px #ff6600,0 0 10px #ff3300,0 0 20px #ff0000;
#     box-shadow:0 4px 10px rgba(0,0,0,0.5);
# }}
# .day-label {{text-align:center; font-size:20px; font-weight:bold;
#     color:#ffcc00; text-shadow:0 0 5px #ff6600,0 0 10px #ff3300,0 0 20px #ff0000;
#     margin-bottom:0px;}}
# .file-upload-container {{display:flex; justify-content:center; gap:20px;
#     margin-top:20px; flex-wrap:wrap;}}
# .band-display-container {{display:flex; justify-content:center; gap:10px;
#     margin:20px 0; flex-wrap:wrap;}}
# .band-image {{border:2px solid #ffcc00; border-radius:5px;
#     padding:5px; background-color:rgba(0,0,0,0.7);}}
# </style>
# """
# st.markdown(page_bg, unsafe_allow_html=True)

# # ----------------------------------------------------------------------
# # 2. Navbar
# # ----------------------------------------------------------------------
# st.markdown(
#     """
#     <div class="navbar">
#         <div> Forest Fire Spread Prediction</div>
#     </div>
#     """,
#     unsafe_allow_html=True
# )

# # ----------------------------------------------------------------------
# # 3. File uploader
# # ----------------------------------------------------------------------
# st.markdown('<div class="file-upload-container">', unsafe_allow_html=True)
# uploaded_files = st.file_uploader(
#     "Upload 5 TIF files (Day 1–Day 5)",
#     type=["tif"],
#     accept_multiple_files=True
# )
# st.markdown('</div>', unsafe_allow_html=True)

# if uploaded_files:
#     if len(uploaded_files) != 5:
#         st.error("Warning: Please upload **exactly 5** TIF files.")
#     else:
#         st.success("All 5 TIF files uploaded successfully!")

# # ----------------------------------------------------------------------
# # 4. Session-state initialisation (add XAI keys)
# # ----------------------------------------------------------------------
# for key in ["fire_mask","fire_mask_clean","prob_map","band_images",
#             "coordinates","saliency_overlay","band_imp_img","band_imp"]:
#     if key not in st.session_state:
#         st.session_state[key] = None

# # ----------------------------------------------------------------------
# # 5. Model processing (now returns 5 items)
# # ----------------------------------------------------------------------
# def process_files(files):
#     inputs, band_images = preprocess_tif_files(files)
#     inputs = inputs.unsqueeze(0)                     # (1,5,23,256,256)

#     model, device = Load_model("50_2020.pth")
#     st.session_state.band_images = band_images

#     # final_run now returns: mask_img, prob_img, saliency_overlay,
#     #                         band_imp_img, band_imp
#     mask_img, prob_img, sal_overlay, band_img, band_imp = \
#         final_run(model, inputs, device)

#     # store everything
#     st.session_state.fire_mask        = mask_img
#     st.session_state.prob_map         = prob_img
#     st.session_state.saliency_overlay = sal_overlay
#     st.session_state.band_imp_img     = band_img
#     st.session_state.band_imp        = band_imp

#     return mask_img, prob_img

# # ----------------------------------------------------------------------
# # 6. Process button
# # ----------------------------------------------------------------------
# if len(uploaded_files) == 5:
#     if st.button("Process Files"):
#         with st.spinner("Processing files..."):
#             lat, long = get_coords(uploaded_files[0])
#             st.session_state.coordinates = {"lat": lat, "long": long}

#             process_files(uploaded_files)

#         st.success("Processing complete!")

#         if st.session_state.coordinates["lat"] and st.session_state.coordinates["long"]:
#             st.info(
#                 f"Top-left coordinates: "
#                 f"{st.session_state.coordinates['lat']:.6f}, "
#                 f"{st.session_state.coordinates['long']:.6f}"
#             )
#         else:
#             st.warning("Could not extract coordinates from the TIFF file")

# # ----------------------------------------------------------------------
# # 7. Binarized 23-rd band (unchanged)
# # ----------------------------------------------------------------------
# if st.session_state.band_images:
#     st.markdown("---")
#     st.markdown("### Binarized 23rd Band of Each TIF File (Positive Pixels = 1)")
#     st.markdown('<div class="band-display-container">', unsafe_allow_html=True)

#     cols = st.columns(5)
#     for i, band_img in enumerate(st.session_state.band_images):
#         with cols[i]:
#             band_display = band_img * 255
#             band_pil = Image.fromarray(band_display.astype(np.uint8))
#             st.image(band_pil, caption=f"Day {i+1} - Band 23 (Binary)",
#                      use_container_width=True)

#     st.markdown('</div>', unsafe_allow_html=True)

# # ----------------------------------------------------------------------
# # 8. Model outputs + XAI
# # ----------------------------------------------------------------------
# if st.session_state.fire_mask and st.session_state.prob_map:
#     st.markdown("---")
#     st.markdown("### Model Outputs + XAI")

#     # ---- original two images ----
#     col1, col2 = st.columns(2)
#     with col1:
#         st.image(st.session_state.fire_mask,
#                  caption="Fire Mask", use_container_width=True)
#     with col2:
#         st.image(st.session_state.prob_map,
#                  caption="Probability Map", use_container_width=True)

#     # ---- XAI: saliency overlay ----
#     st.image(st.session_state.saliency_overlay,
#              caption="Saliency (model attention)", use_container_width=True)

#     # ---- XAI: band importance chart ----
#     st.image(st.session_state.band_imp_img,
#              caption="Band importance (23 bands)", use_container_width=True)

#     # ---- top-5 bands ----
#     top5 = np.argsort(st.session_state.band_imp)[-5:][::-1]
#     st.markdown("**Top-5 most important bands:**")
#     for i, idx in enumerate(top5, 1):
#         st.markdown(f"{i}. **Band {idx+1:02d}** – {st.session_state.band_imp[idx]:.3f}")

#     # ------------------------------------------------------------------
#     # 9. Coordinate finder (unchanged, just uses fire_mask_clean)
#     # ------------------------------------------------------------------
#     st.markdown("# Coordinates Finder")

#     default_lat = st.session_state.coordinates["lat"] or 0.0
#     default_long = st.session_state.coordinates["long"] or 0.0

#     lat0 = st.number_input("Enter top-left latitude:", value=float(default_lat))
#     lon0 = st.number_input("Enter top-left longitude:", value=float(default_long))

#     # mask for coordinate finder
#     fire_mask_np = np.array(st.session_state.fire_mask) if isinstance(
#         st.session_state.fire_mask, Image.Image) else st.session_state.fire_mask

#     last_band = np.array(st.session_state.band_images[-1]) if isinstance(
#         st.session_state.band_images[-1], Image.Image) else st.session_state.band_images[-1]

#     fire_mask_bool = fire_mask_np.astype(bool)
#     last_band_bool = last_band.astype(bool)

#     fire_mask_cleaned = np.logical_and(fire_mask_bool,
#                                       np.logical_not(last_band_bool)).astype(np.uint8)
#     st.session_state.fire_mask_clean = Image.fromarray(fire_mask_cleaned * 255)

#     coords = streamlit_image_coordinates(st.session_state.fire_mask_clean, width=600)

#     if coords is not None:
#         i, j = coords["y"], coords["x"]
#         if lat0 is not None and lon0 is not None:
#             lati = lat0 - (i * RES / 111320)
#             longi = lon0 + (j * RES / (111320 * np.cos(np.radians(lat0))))
#             st.markdown(f" Pixel: ({i}, {j})")
#             st.markdown(f"**Latitude:** {lati:.6f} | **Longitude:** {longi:.6f}")
#         else:
#             st.error("Please enter valid latitude/longitude values")

#     # ------------------------------------------------------------------
#     # 10. Clear button
#     # ------------------------------------------------------------------
#     if st.button("Clear Outputs"):
#         for k in ["fire_mask","prob_map","band_images","coordinates",
#                   "saliency_overlay","band_imp_img","band_imp",
#                   "fire_mask_clean"]:
#             st.session_state[k] = None
#         st.rerun()
# else:
#     st.warning("Upload all 5 TIF files and click **Process** to see outputs.")

# # ----------------------------------------------------------------------
# # 11. About Us
# # ----------------------------------------------------------------------
# st.markdown("---")
# st.markdown("### About Us", unsafe_allow_html=True)
# st.write("We are building a Forest Fire Spread Analysis tool.")







# web.py  (FULL UPDATED CODE – NO WARNINGS)
import streamlit as st
import base64
import torch
import numpy as np
from PIL import Image
from preprocess import preprocess_tif_files
from model import Load_model
from result import final_run
from streamlit_image_coordinates import streamlit_image_coordinates
from coords import get_coords

IMG_SIZE = 256
RES = 375

# ----------------------------------------------------------------------
st.set_page_config(page_title="Forest Fire TIF Uploader", layout="wide")

def get_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img_base64 = get_base64("image2.jpeg")

page_bg = f"""
<style>
.stApp {{
    background: url("data:image/jpeg;base64,{img_base64}") no-repeat center center fixed;
    background-size: cover;
}}
#MainMenu {{visibility: hidden;}}
header {{visibility: hidden;}}
.navbar {{
    position: fixed; top:0; left:0; width:100%; background:#1e211f;
    padding:15px 30px; display:flex; justify-content:space-between;
    align-items:center; color:#ffcc00; font-size:20px;
    font-family:'Inter','Poppins','Segoe UI','Roboto',sans-serif;
    font-weight:bold; letter-spacing:0.5px; z-index:1000;
    text-shadow:0 0 5px #ff6600,0 0 10px #ff3300,0 0 20px #ff0000;
    box-shadow:0 4px 10px rgba(0,0,0,0.5);
}}
.file-upload-container {{display:flex; justify-content:center; gap:20px; margin-top:20px; flex-wrap:wrap;}}
.band-display-container {{display:flex; justify-content:center; gap:10px; margin:20px 0; flex-wrap:wrap;}}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

st.markdown(
    """
    <div class="navbar">
        <div> Forest Fire Spread Prediction</div>
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------------------------------------------------------------
st.markdown('<div class="file-upload-container">', unsafe_allow_html=True)
uploaded_files = st.file_uploader(
    "Upload 5 TIF files (Day 1–Day 5)",
    type=["tif"],
    accept_multiple_files=True
)
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_files:
    if len(uploaded_files) != 5:
        st.error("Warning: Please upload **exactly 5** TIF files.")
    else:
        st.success("All 5 TIF files uploaded successfully!")

# ----------------------------------------------------------------------
for key in ["fire_mask","fire_mask_clean","prob_map","band_images",
            "coordinates","saliency_overlay","band_imp_img","band_imp"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ----------------------------------------------------------------------
def process_files(files):
    inputs, band_images = preprocess_tif_files(files)
    inputs = inputs.unsqueeze(0)

    model, device = Load_model("50_2020.pth")
    st.session_state.band_images = band_images

    mask_img, prob_img, sal_overlay, band_img, band_imp = final_run(model, inputs, device)

    st.session_state.fire_mask        = mask_img
    st.session_state.prob_map         = prob_img
    st.session_state.saliency_overlay = sal_overlay
    st.session_state.band_imp_img     = band_img
    st.session_state.band_imp        = band_imp

    return mask_img, prob_img

# ----------------------------------------------------------------------
if len(uploaded_files) == 5:
    if st.button("Process Files"):
        with st.spinner("Processing files..."):
            lat, long = get_coords(uploaded_files[0])
            st.session_state.coordinates = {"lat": lat, "long": long}
            process_files(uploaded_files)

        st.success("Processing complete!")
        if st.session_state.coordinates["lat"] and st.session_state.coordinates["long"]:
            st.info(f"Top-left coordinates: {st.session_state.coordinates['lat']:.6f}, {st.session_state.coordinates['long']:.6f}")
        else:
            st.warning("Could not extract coordinates from the TIFF file")

# ----------------------------------------------------------------------
if st.session_state.band_images:
    st.markdown("---")
    st.markdown("### Binarized 23rd Band of Each TIF File (Positive Pixels = 1)")
    st.markdown('<div class="band-display-container">', unsafe_allow_html=True)
    cols = st.columns(5)
    for i, band_img in enumerate(st.session_state.band_images):
        with cols[i]:
            band_display = band_img * 255
            band_pil = Image.fromarray(band_display.astype(np.uint8))
            st.image(band_pil, caption=f"Day {i+1} - Band 23 (Binary)", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------
if st.session_state.fire_mask and st.session_state.prob_map:
    st.markdown("---")
    st.markdown("### Model Outputs")
    col1, col2 = st.columns(2)
    with col1:
        st.image(st.session_state.fire_mask, caption="Fire Mask", use_container_width=True)
    with col2:
        st.image(st.session_state.prob_map, caption="Probability Map", use_container_width=True)

    # --------------------- XAI: SIDE-BY-SIDE ---------------------
    st.markdown("### XAI: Model Explanation")
    col_xai1, col_xai2 = st.columns(2)
    with col_xai1:
        st.image(
            st.session_state.saliency_overlay,
            caption="Saliency Map (What the model focused on)",
            use_container_width=True   # FIXED: new parameter name
        )
    with col_xai2:
        st.image(
            st.session_state.band_imp_img,
            caption="Band Importance (Top bands used)",
            use_container_width=True   # FIXED: new parameter name
        )

    # Top-5 bands
    top5 = np.argsort(st.session_state.band_imp)[-5:][::-1]
    st.markdown("**Top-5 Most Important Bands:**")
    for i, idx in enumerate(top5, 1):
        st.markdown(f"{i}. **Band {idx+1:02d}** – {st.session_state.band_imp[idx]:.3f}")

    # ------------------------------------------------------------------
    st.markdown("# Coordinates Finder")
    default_lat = st.session_state.coordinates["lat"] or 0.0
    default_long = st.session_state.coordinates["long"] or 0.0
    lat0 = st.number_input("Enter top-left latitude:", value=float(default_lat))
    lon0 = st.number_input("Enter top-left longitude:", value=float(default_long))

    fire_mask_np = np.array(st.session_state.fire_mask) if isinstance(st.session_state.fire_mask, Image.Image) else st.session_state.fire_mask
    last_band = np.array(st.session_state.band_images[-1]) if isinstance(st.session_state.band_images[-1], Image.Image) else st.session_state.band_images[-1]

    fire_mask_cleaned = np.logical_and(fire_mask_np.astype(bool), np.logical_not(last_band.astype(bool))).astype(np.uint8)
    st.session_state.fire_mask_clean = Image.fromarray(fire_mask_cleaned * 255)

    coords = streamlit_image_coordinates(st.session_state.fire_mask_clean, width=600)
    if coords is not None:
        i, j = coords["y"], coords["x"]
        if lat0 is not None and lon0 is not None:
            lati = lat0 - (i * RES / 111320)
            longi = lon0 + (j * RES / (111320 * np.cos(np.radians(lat0))))
            st.markdown(f" Pixel: ({i}, {j})")
            st.markdown(f"**Latitude:** {lati:.6f} | **Longitude:** {longi:.6f}")
        else:
            st.error("Please enter valid latitude/longitude values")

    if st.button("Clear Outputs"):
        for k in ["fire_mask","prob_map","band_images","coordinates",
                  "saliency_overlay","band_imp_img","band_imp","fire_mask_clean"]:
            st.session_state[k] = None
        st.rerun()
else:
    st.warning("Upload all 5 TIF files and click **Process** to see outputs.")

st.markdown("---")
st.markdown("### About Us", unsafe_allow_html=True)
st.write("We are building a Forest Fire Spread Analysis tool.")