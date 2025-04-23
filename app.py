# app.py
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from model import ColorizationNet
import cv2

st.title("🎨 Stroke-based Image Colorization")

model = ColorizationNet()
model.load_state_dict(torch.load("colorization_model_ab1.pth", map_location=torch.device('cpu')))
model.eval()  

uploaded_img = st.file_uploader("Upload a grayscale image (PNG/JPG)", type=["png", "jpg", "jpeg"])
if uploaded_img:
    image = Image.open(uploaded_img).convert("L").resize((256, 256))
    gray_rgb = image.convert("RGB")
    st.image(gray_rgb, caption="Input Grayscale", width=256)

    color = st.color_picker("Pick a stroke color", "#FF0000")

    canvas_result = st_canvas(
        fill_color="rgba(255,255,255,0)",
        stroke_width=8,
        stroke_color=color,
        background_image=gray_rgb,
        update_streamlit=True,
        height=256,
        width=256,
        drawing_mode="freedraw",
        key="canvas",
    )

    if st.button("Colorize Image"):
        if canvas_result.image_data is not None:
            transform_gray = transforms.Compose([
                transforms.ToTensor(),
            ])
            transform_rgb = transforms.Compose([
                transforms.ToPILImage(), transforms.Resize((256, 256)), transforms.ToTensor()
            ])

            gray_tensor = transform_gray(image).unsqueeze(0)  
            strokes_np = canvas_result.image_data.astype(np.uint8)
            stroke_tensor = transform_rgb(strokes_np).unsqueeze(0) 

            print(gray_tensor.shape)
            print(stroke_tensor.shape)
            
            stroke_mask = (stroke_tensor.sum(dim=1, keepdim=True) > 0).float() 

            print(stroke_mask.shape)

            input_tensor = torch.cat([gray_tensor, stroke_tensor, stroke_mask], dim=1) 

            with torch.no_grad():
                output = model(input_tensor)[0]
                output_ab = output.cpu().numpy().transpose(1, 2, 0)  
                output_ab = (output_ab * 128).astype(np.uint8)

                L = (gray_tensor[0].numpy().transpose(1, 2, 0) * 255).astype(np.uint8) 
                lab = np.concatenate([L, output_ab], axis=2)  
                rgb_result = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)

            st.image(rgb_result, caption="Colorized Output", width=256)