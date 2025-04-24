/*
    Satvika Eda, Divya Sri Bandaru & Dhriti Anjaria
    23rd April 2025
    This code is to used to create a web app for stroke-based image colorization using Streamlit.
*/
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import torch
import cv2
import io
import datetime

st.title("🎨 Stroke-based Image Colorization")

# Loading model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("model_20250423_032933.pth", map_location=device, weights_only=False)
model = model.to(device)
model.eval()

# Upload an image
st.subheader("1. Upload an image")
uploaded_file = st.file_uploader("Upload a grayscale image (PNG/JPG)", type=["png", "jpg", "jpeg"])
if uploaded_file is None:
    st.stop()

# Converting uploaded image to grayscale and RGB (for canvas)
gray_image = Image.open(uploaded_file).convert("L").resize((256, 256))
gray_rgb = gray_image.convert("RGB")

st.image(gray_rgb, caption="Grayscale Input", width=256)

# Canvas to draw strokes
st.subheader("2. Draw your strokes on the image")
stroke_color = st.color_picker("Pick a stroke color", "#FF0000")

canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0)",
    stroke_width=8,
    stroke_color=stroke_color,
    background_image=gray_rgb,
    update_streamlit=True,
    height=256,
    width=256,
    drawing_mode="freedraw",
    key="canvas",
)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

if canvas_result.image_data is not None:
    strokes_only = canvas_result.image_data.astype("uint8")
    
    # Convert grayscale image to RGB numpy array
    gray_np = np.array(gray_rgb)
    
    mask = strokes_only[:, :, 3] > 0 
    composite_img = gray_np.copy()
    
    # Apply the strokes where mask is True
    composite_img[mask] = strokes_only[mask, :3]
    
    # download button for composite image
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pil_composite = Image.fromarray(composite_img)
    buf_composite = io.BytesIO()
    pil_composite.save(buf_composite, format="PNG")
    buf_composite.seek(0)
    
    st.download_button(
        "\U0001F4BE Save Stroked Image",
        data=buf_composite.getvalue(),
        file_name=f"gray_with_strokes_{timestamp}.png",
        mime="image/png",
        key="download_composite"
    )
# Colorize when button is clicked using loaded model
if st.button("🎨 Colorize Image"):
    if canvas_result.image_data is None:
        st.warning("Please draw at least one stroke.")
        st.stop()

    # Converting grayscale image to LAB L channel
    gray_np = np.array(gray_rgb)
    lab_gray = cv2.cvtColor(gray_np, cv2.COLOR_RGB2LAB)
    L_channel = lab_gray[..., 0]
    L_tensor = torch.from_numpy(L_channel / 255.0).unsqueeze(0).unsqueeze(0).to(device).float()

    # Converting canvas strokes to AB hints
    strokes_np = canvas_result.image_data[..., :3].astype(np.uint8)
    lab_hint = cv2.cvtColor(strokes_np, cv2.COLOR_RGB2LAB)
    ab_hint = lab_hint[..., 1:].astype(np.float32)
    ab_hint = (ab_hint - 128) / 128.0
    ab_hint_tensor = torch.from_numpy(ab_hint.transpose(2, 0, 1)).unsqueeze(0).to(device).float()

    # Combining L and AB
    input_tensor = torch.cat([L_tensor, ab_hint_tensor], dim=1)

    # Model prediction
    with torch.no_grad():
        pred_ab = model(input_tensor)[0].cpu().numpy()

    # Reconstructing LAB and convert to RGB
    pred_ab = (pred_ab * 128 + 128).clip(0, 255).astype(np.uint8)
    lab_output = np.stack([L_channel, pred_ab[0], pred_ab[1]], axis=-1).astype(np.uint8)
    rgb_output = cv2.cvtColor(lab_output, cv2.COLOR_LAB2RGB)

    st.subheader("3. Colorized Output")
    st.image(rgb_output, caption="Colorized Image", width=256)

    output_pil = Image.fromarray(rgb_output)
    buf_output = io.BytesIO()
    output_pil.save(buf_output, format="PNG")
    colorized_filename = f"colorized_image_{timestamp}.png"
    st.download_button("💾 Save Colorized Image", data=buf_output.getvalue(), file_name=colorized_filename, mime="image/png")