"""
Image Deblurring Web App
========================
This file deploys your trained CNN deblurring model as a web app using Gradio.
Run this with: python app.py
Then open: http://localhost:7860
"""

import torch
import torch.nn as nn
import gradio as gr
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F

# ============================================================
# STEP 1: Define your model architecture
# ------------------------------------------------------------
# IMPORTANT: This must EXACTLY match the architecture you used
# during training. If your architecture is different, update
# this class to match your notebook.
# ============================================================

class DeblurCNN(nn.Module):
    """
    A simple CNN for image deblurring.
    Adjust this to match your training architecture.
    Based on: https://debuggercafe.com/image-deblurring-using-convolutional-neural-networks-and-deep-learning/
    """
    def __init__(self):
        super(DeblurCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding =2 )
        self.conv2 = nn.Conv2d(64, 32, kernel_size = 1, padding = 2)
        self.conv3 = nn.Conv2d(32, 3, kernel_size = 5, padding = 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)

        return x
    
# ============================================================
# STEP 2: Load your model
# ------------------------------------------------------------
# Set the path to your .pth file here.
# ============================================================

MODEL_PATH = "C:/Users/nissa/.cache/kagglehub/datasets/kwentar/blur-dataset/versions/2model.pth"

# Detect device: use GPU if available, otherwise CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

def load_model():
    """Load the trained model from disk."""
    model = DeblurCNN().to(DEVICE)
    # Load saved weights
    # If you saved the whole model (torch.save(model, ...)) use:
    #   model = torch.load(MODEL_PATH, map_location=DEVICE)
    # If you saved only state_dict (torch.save(model.state_dict(), ...)) use:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()  # Set to evaluation mode (disables dropout, batchnorm training behavior)
    return model

try:
    model = load_model()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"⚠️  Could not load model: {e}")
    print("App will still run but predictions will fail until model path is corrected.")
    model = None


# ============================================================
# STEP 3: Define the prediction pipeline
# ------------------------------------------------------------
# This function is called every time a user uploads an image.
# It:
#   1. Converts the PIL image to a PyTorch tensor
#   2. Runs the model
#   3. Converts the output tensor back to a PIL image
# ============================================================

# Image preprocessing: convert PIL → Tensor (values 0-1)
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts HxWxC uint8 [0,255] → CxHxW float [0,1]
])

def deblur_image(input_image: Image.Image) -> Image.Image:
    """
    Takes a blurry PIL Image, runs deblurring model, returns sharp PIL Image.
    """
    if model is None:
        raise gr.Error("Model not loaded. Check your MODEL_PATH in app.py")

    # Convert to RGB in case the image has an alpha channel (RGBA/PNG)
    input_image = input_image.convert("RGB")

    # Transform image to tensor: shape [1, C, H, W]
    input_tensor = transform(input_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():  # Disable gradient tracking (faster, less memory)
        output_tensor = model(input_tensor)

    # Clamp values to [0,1] range and remove batch dimension
    output_tensor = output_tensor.squeeze(0).clamp(0, 1)

    # Convert tensor [C, H, W] back to PIL image
    output_image = transforms.ToPILImage()(output_tensor.cpu())

    return output_image


# ============================================================
# STEP 4: Build the Gradio Interface
# ------------------------------------------------------------
# Gradio creates a web UI automatically.
# - `inputs`: what the user provides (image upload)
# - `outputs`: what the app returns (deblurred image)
# ============================================================

with gr.Blocks(
    title="🔍 Image Deblurring AI",
    theme=gr.themes.Soft(primary_hue="violet"),
    css="""
        .gradio-container { max-width: 900px; margin: auto; }
        h1 { text-align: center; font-size: 2rem; }
        .subtitle { text-align: center; color: #666; margin-bottom: 1rem; }
    """
) as demo:

    gr.Markdown("# 🔍 Image Deblurring AI")
    gr.Markdown(
        "<p class='subtitle'>Upload a blurry image and let the CNN model restore sharpness!</p>"
    )

    with gr.Row():
        with gr.Column():
            input_img = gr.Image(
                label="📷 Blurry Input Image",
                type="pil",
            )
            deblur_btn = gr.Button("✨ Deblur Image", variant="primary", size="lg")

        with gr.Column():
            output_img = gr.Image(
                label="🌟 Deblurred Output Image",
                type="pil",
            )

    deblur_btn.click(
        fn=deblur_image,
        inputs=input_img,
        outputs=output_img,
    )

    gr.Markdown("""
    ---
    ### How it works
    1. **Upload** a blurry image (Gaussian blur, motion blur, defocus blur)
    2. Click **Deblur Image**
    3. The CNN processes it in milliseconds and returns a sharpened version

    > **Model**: Convolutional Neural Network trained on the [Blur Dataset](https://www.kaggle.com/datasets/kwentar/blur-dataset)
    """)


# ============================================================
# STEP 5: Launch the app
# ============================================================

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # Makes it accessible on your local network
        server_port=7860,       # Open http://localhost:7860 in your browser
        share=False,            # Set to True to get a public URL (via Gradio tunnel)
        show_error=True,
    )
