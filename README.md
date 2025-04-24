# Stroke-Based and Free-Form Image Colorization

Group Members: Divya Sri Bandaru, Dhriti Anjaria, Satvika Eda

In this project, we develop an image colorization pipeline for greyscale images using 
both free-form and stroke-based techniques. Free-form colorization predicts 
the colors from the greyscale input, using a baseline encoder-decoder, 
vanilla U-Net, ResNet18-based U-Net, Stable Diffusion and Generative Adversarial Networks. 
Stroke-based colorization allows users to apply colored strokes, which are propagated by the model. 
For this, we use a U-Net with a ResNet34 encoder and ControlNet with Stable Diffusion v1.5. 
All models are trained on subsets of ImageNet—100 classes for free-form and 50 for stroke-based colorization.


Link to the presentation: 

| **Filename**                                                   | **Description**                                                                                                          |
|----------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| `App.py`                                                       | Streamlit web app for image colorization.                                                                                |
| `controlnet-and-stable-diffusion.ipynb`                        | Fine-tuning ControlNet and zero-shot inference using Stable Diffusion + fine-tuned ControlNet for stroke-based colorization. |
| `dataset and controlnet + stable diffusion model.ipynb`        | Inference for stroke-based colorization using only pretrained models.                                                   |
| `dataset.py`                                                   | Contains the `MultiCueStrokeDataset` class.                                                                             |
| `freeformGan.ipynb`                                            | Free-form image colorization using GAN.                                                                                 |
| `freeform-autoencoder.ipynb`                                   | Baseline autoencoder model for free-form colorization.                                                                  |
| `freeform-resnet-unet.ipynb`                                   | Free-form image colorization using ResNet-UNet hybrid.                                                                  |
| `freeform-stable-diffusion.ipynb`                              | Free-form image colorization using Stable Diffusion v1.5.                                                               |
| `freeform-unet.ipynb`                                          | Free-form image colorization using U-Net.                                                                               |
| `stroke-controlnet + stable diffusion -model-inference.ipynb`  | Inference for stroke-based colorization using pretrained models; includes dataset experiments and visualization.        |
| `stroke-controlnet-and-stable-diffusion-training.ipynb`        | Fine-tuning ControlNet and inference with Stable Diffusion + ControlNet for stroke-based colorization.                 |
| `stroke-unet-gan.ipynb`                                        | Training a stroke-based U-Net model with ResNet34 encoder and GAN architecture.                                         |
