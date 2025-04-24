# Stroke-Based and Free-Form Image Colorization Using Stable Diffusion and ControlNet

Group Members: Divya Sri Bandaru, Dhriti Anjaria, Satvika Eda

In this project, we develop an image colorization pipeline for greyscale images using 
both free-form and stroke-based techniques. Free-form colorization predicts 
the colors from the greyscale input, using a baseline encoder-decoder, 
vanilla U-Net, ResNet18-based U-Net, Stable Diffusion and Generative Adversarial Networks. 
Stroke-based colorization allows users to apply colored strokes, which are propagated by the model. 
For this, we use a U-Net with a ResNet34 encoder and ControlNet with Stable Diffusion v1.5. 
All models are trained on subsets of ImageNet—100 classes for free-form and 50 for stroke-based colorization.


Link to the presentation: 