# Streamlit Web App for Image Colorization
# Satvika Eda, Divya Sri Bandaru & Dhriti Anjaria
# 23rd April 2025
# This code is used to create a web app for stroke-based and free-form image colorization using Streamlit.

import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import torch
import cv2
import io
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torchvision.transforms as T

import numpy as np
import cv2
import torchvision.models as models
import torch.nn.functional as F

# Defined a convolutional block 
def conv(in_c, out_c, k=4, s=2, p=1, bn=True, act=nn.LeakyReLU(0.2)):
    layers=[nn.Conv2d(in_c, out_c, k, s, p, bias=False)]
    if bn: layers.append(nn.BatchNorm2d(out_c))
    layers.append(act)
    return nn.Sequential(*layers)

# U-Net generator to encode L input and decode to ab channels
class UNetG(nn.Module):

    # Initializes the U-Net generator with encoder and decoder layers
    def __init__(self):
        super().__init__()
        # Encoder
        self.e1 = conv(1,64,bn=False)
        self.e2 = conv(64,128)
        self.e3 = conv(128,256)
        self.e4 = conv(256,512)
        self.e5 = conv(512,512)
        self.e6 = conv(512,512)
        self.e7 = conv(512,512)
        # Decoder
        self.d1 = nn.ConvTranspose2d(512,512,4,2,1,bias=False)
        self.d2 = nn.ConvTranspose2d(1024,512,4,2,1,bias=False)
        self.d3 = nn.ConvTranspose2d(1024,512,4,2,1,bias=False)
        self.d4 = nn.ConvTranspose2d(1024,256,4,2,1,bias=False)
        self.d5 = nn.ConvTranspose2d(512,128,4,2,1,bias=False)
        self.d6 = nn.ConvTranspose2d(256,64,4,2,1,bias=False)
        self.out= nn.ConvTranspose2d(128,2,4,2,1)
        self.relu, self.tanh = nn.ReLU(), nn.Tanh()

    # TO run a forward pass through encoder, decoder, and skip connections
    def forward(self,x):
        e1=self.e1(x);  e2=self.e2(e1); e3=self.e3(e2); e4=self.e4(e3)
        e5=self.e5(e4); e6=self.e6(e5); e7=self.e7(e6)
        d1=self.relu(self.d1(e7))
        d2=self.relu(self.d2(torch.cat([d1,e6],1)))
        d3=self.relu(self.d3(torch.cat([d2,e5],1)))
        d4=self.relu(self.d4(torch.cat([d3,e4],1)))
        d5=self.relu(self.d5(torch.cat([d4,e3],1)))
        d6=self.relu(self.d6(torch.cat([d5,e2],1)))
        return self.tanh(self.out(torch.cat([d6,e1],1)))


# ResNet U-Net-like architecture for colorization of free form strokes
class ResNetUNetColor(nn.Module):
    def __init__(self):
        super(ResNetUNetColor, self).__init__()
        resnet = models.resnet18(pretrained=True)

        self.input_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3 
        self.layer4 = resnet.layer4 

        # Decoder
        self.up1 = self.up_block(512, 256)
        self.up2 = self.up_block(512, 128)
        self.up3 = self.up_block(256, 64)
        self.up4 = self.up_block(128, 64) 
        self.up5 = nn.Sequential(       
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, kernel_size=1) 
        )

    def up_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        x1 = self.relu(self.bn1(self.input_conv(x))) 
        x2 = self.layer1(self.maxpool(x1))          
        x3 = self.layer2(x2)                        
        x4 = self.layer3(x3)                         
        x5 = self.layer4(x4)                       

        # Decoder 
        d1 = self.up1(x5)                             
        d2 = self.up2(torch.cat([d1, x4], dim=1))     
        d3 = self.up3(torch.cat([d2, x3], dim=1))  
        d4 = self.up4(torch.cat([d3, x2], dim=1))     
        out = self.up5(d4)                          

        return out


# Generator: U-Net-like architecture for colorization
class ColorizationGenerator(nn.Module):
    def __init__(self):
        super(ColorizationGenerator, self).__init__()
        
        def down(in_c, out_c, norm=True):
            layers = [nn.Conv2d(in_c, out_c, 4, 2, 1)]
            if norm:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        def up(in_c, out_c, dropout=0.0):
            layers = [nn.ConvTranspose2d(in_c, out_c, 4, 2, 1),
                      nn.BatchNorm2d(out_c),
                      nn.ReLU(inplace=True)]
            if dropout:
                layers.append(nn.Dropout(dropout))
            return nn.Sequential(*layers)

        self.down1 = down(3, 64, norm=False)
        self.down2 = down(64, 128)
        self.down3 = down(128, 256)
        self.down4 = down(256, 512)
        self.down5 = down(512, 512)
        self.down6 = down(512, 512)
        
        self.up1 = up(512, 512)
        self.up2 = up(1024, 512)
        self.up3 = up(1024, 256)
        self.up4 = up(512, 128)
        self.up5 = up(256, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 2, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)

        u1 = self.up1(d6)
        u2 = self.up2(torch.cat([u1, d5], 1))
        u3 = self.up3(torch.cat([u2, d4], 1))
        u4 = self.up4(torch.cat([u3, d3], 1))
        u5 = self.up5(torch.cat([u4, d2], 1))
        out = self.final(torch.cat([u5, d1], 1))
        return out


st.title("🎨 Image Colorization App")

# Model and mode selection
mode = st.radio("Choose Colorization Mode", ["Stroke-Based", "Free-Form"])
model_choice = st.selectbox("Choose a Model", ["U-Net", "GAN"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model based on selection
if mode == "Stroke-Based":
    if model_choice == "U-Net":
        model_path = "model_20250423_032933.pth"
        model = torch.load(model_path, map_location=device, weights_only=False)

    elif model_choice == "GAN":
        model_path = "generator_epoch2.pth"
        model = ColorizationGenerator()
        model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device).eval()

elif mode == "Free-Form":
    if model_choice == "U-Net":
        model_path = "resnet_freeform.pth"
        model = ResNetUNetColor().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
    elif model_choice == "GAN":
        model_path = "generator_freeform.pth"
        model = UNetG()
        model.load_state_dict(torch.load(model_path, map_location=device))
    
# Upload image
st.subheader("1. Upload an image")
uploaded_file = st.file_uploader("Upload an image (PNG/JPG)", type=["png", "jpg", "jpeg"])
if uploaded_file is None:
    st.stop()

# Process for Free-Form
if mode == "Free-Form":
    gray_image = Image.open(uploaded_file).convert("RGB").resize((256, 256))
    grayscale_image = gray_image.convert("L")
    gray_rgb = grayscale_image.convert("RGB")
    st.image(gray_rgb, caption="Grayscale Input", width=256)
    gray_np = np.array(gray_image)
    lab = cv2.cvtColor(gray_np, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0] / 255.0
    L_tensor = torch.from_numpy(L).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        ab_pred = model(L_tensor)[0].cpu().numpy().transpose(1, 2, 0) * 128.0
    lab_pred = np.zeros((256, 256, 3), dtype=np.float32)
    lab_pred[:, :, 0] = L * 255.0
    lab_pred[:, :, 1:] = ab_pred
    rgb_pred = cv2.cvtColor(lab_pred.astype(np.uint8), cv2.COLOR_LAB2RGB)
    st.subheader("2. Colorized Output")
    st.image(rgb_pred, caption="Colorized Image", width=256)

else:
    gray_image = Image.open(uploaded_file).convert("L").resize((256, 256))
    gray_rgb = gray_image.convert("RGB")
    st.image(gray_rgb, caption="Grayscale Input", width=256)
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

    # Display the drawn strokes
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if canvas_result.image_data is not None:
        strokes_only = canvas_result.image_data.astype("uint8")
        gray_np = np.array(gray_rgb)
        mask = strokes_only[:, :, 3] > 0
        composite_img = gray_np.copy()
        composite_img[mask] = strokes_only[mask, :3]
        pil_composite = Image.fromarray(composite_img)
        buf_composite = io.BytesIO()
        pil_composite.save(buf_composite, format="PNG")
        buf_composite.seek(0)
        st.download_button("💾 Save Stroked Image", data=buf_composite.getvalue(), file_name=f"gray_with_strokes_{timestamp}.png", mime="image/png")

    if st.button("🎨 Colorize Image"):
        if canvas_result.image_data is None:
            st.warning("Please draw at least one stroke.")
            st.stop()

        gray_np = np.array(gray_rgb)
        lab_gray = cv2.cvtColor(gray_np, cv2.COLOR_RGB2LAB)
        L_channel = lab_gray[..., 0]
        L_tensor = torch.from_numpy(L_channel / 255.0).unsqueeze(0).unsqueeze(0).to(device).float()
        strokes_np = canvas_result.image_data[..., :3].astype(np.uint8)
        lab_hint = cv2.cvtColor(strokes_np, cv2.COLOR_RGB2LAB)
        ab_hint = lab_hint[..., 1:].astype(np.float32)
        ab_hint = (ab_hint - 128) / 128.0
        ab_hint_tensor = torch.from_numpy(ab_hint.transpose(2, 0, 1)).unsqueeze(0).to(device).float()
        input_tensor = torch.cat([L_tensor, ab_hint_tensor], dim=1)

        with torch.no_grad():
            pred_ab = model(input_tensor)[0].cpu().numpy()

        pred_ab = (pred_ab * 128 + 128).clip(0, 255).astype(np.uint8)
        lab_output = np.stack([L_channel, pred_ab[0], pred_ab[1]], axis=-1).astype(np.uint8)
        rgb_output = cv2.cvtColor(lab_output, cv2.COLOR_LAB2RGB)

        st.subheader("3. Colorized Output")
        st.image(rgb_output, caption="Colorized Image", width=256)
        output_pil = Image.fromarray(rgb_output)
        buf_output = io.BytesIO()
        output_pil.save(buf_output, format="PNG")
        st.download_button("💾 Save Colorized Image", data=buf_output.getvalue(), file_name=f"colorized_image_{timestamp}.png", mime="image/png")