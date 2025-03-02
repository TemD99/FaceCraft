import pickle
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys
import tkinter as tk
from tkinter import Scale, HORIZONTAL, Button
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import gc  # Garbage collection

sys.path.append('/Python/data/SG2')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open('/Data/00010-coco_processed@256-lite-gpus1-b224-bgpu1/network-snapshot-003001.pkl', 'rb') as f:
    G = pickle.load(f)['G_ema'].to(device)

# Load CLIP model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Linear layer for dimensionality reduction from 512 to 64
dim_reduction_layer = torch.nn.Linear(512, 64).to(device)

def tensor_to_image(tensor):
    tensor = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    tensor = (tensor * 127.5 + 127.5).astype(np.uint8)
    return Image.fromarray(tensor)

def find_direction(attribute_text, opposite_text, num_samples=1, perturbation_strength=0.0005):
    latent_directions = []
    losses = []
    for _ in range(num_samples):
        z_base = torch.randn(1, G.z_dim, device=device)
        c_base = torch.zeros(1, G.c_dim, device=device)
        img_base = G(z_base, c_base, truncation_psi=0.7)

        z_perturbed = z_base + perturbation_strength * torch.randn(1, G.z_dim, device=device)
        img_perturbed = G(z_perturbed, c_base, truncation_psi=0.7)

        img_base_pil = tensor_to_image(img_base)
        img_perturbed_pil = tensor_to_image(img_perturbed)

        inputs_base = clip_processor(images=img_base_pil, return_tensors="pt").to(device)
        inputs_perturbed = clip_processor(images=img_perturbed_pil, return_tensors="pt").to(device)
        inputs_text_attr = clip_processor(text=[attribute_text], return_tensors="pt", padding=True).to(device)
        inputs_text_opposite = clip_processor(text=[opposite_text], return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            image_features_base = clip_model.get_image_features(**inputs_base).float()
            image_features_perturbed = clip_model.get_image_features(**inputs_perturbed).float()
            text_features_attr = clip_model.get_text_features(**inputs_text_attr).float()
            text_features_opposite = clip_model.get_text_features(**inputs_text_opposite).float()

        # Calculate the direction
        direction = (text_features_attr - text_features_opposite).mean(dim=0)
        latent_directions.append(direction)

        # Calculate the loss (cosine similarity)
        similarity_attr = torch.nn.functional.cosine_similarity(image_features_perturbed, text_features_attr)
        similarity_opposite = torch.nn.functional.cosine_similarity(image_features_perturbed, text_features_opposite)
        loss = similarity_attr - similarity_opposite
        losses.append(loss.item())

    # Output the losses
    for i, loss in enumerate(losses):
        print(f"Sample {i + 1}: Loss = {loss}")

    direction_mean = torch.stack(latent_directions).mean(dim=0)

    # Normalize the direction vector
    direction_mean /= direction_mean.norm(dim=-1, keepdim=True)
    
    return direction_mean

def redefine_age_direction():
    global age_direction
    age_direction = find_direction("a young person", "an old person").to(device)
    if age_direction.shape[0] != 512:
        age_direction = age_direction.view(512)
    directions['age'] = age_direction
    print("Age direction redefined.")

# Initialize directions dictionary
directions = {}

# Find directions for specific attributes
redefine_age_direction()

current_directions = {
    'age': 0
}

# Initialize the figure and axes
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
canvas = None

def generate_image():
    z = torch.randn(1, G.z_dim, device=device)
    c = torch.zeros(1, G.c_dim, device=device)
    img = G(z, c, truncation_psi=0.7)
    return z, c, tensor_to_image(img)

def new_image():
    global z, c, original_img
    z, c, original_img = generate_image()
    update_image(resolution=1024)

# Generate a random latent vector and image
z, c, original_img = generate_image()

def update_image(resolution=256):
    global z, c, original_img
    valid_image = False
    
    while not valid_image:
        # Clear the previous images
        for ax in axes:
            ax.clear()

        manipulated_latent = z.clone()
        for attr, alpha in current_directions.items():
            direction = directions[attr].unsqueeze(0)  # Add batch dimension
            transformed_direction = dim_reduction_layer(direction).squeeze(0)  # Transform direction
            manipulated_latent[:, :64] += alpha * transformed_direction

        with torch.no_grad():
            manipulated_img = G(manipulated_latent, c, truncation_psi=0.7)

        manipulated_img_pil = tensor_to_image(manipulated_img)

        if resolution < 1024:
            manipulated_img_pil = manipulated_img_pil.resize((resolution, resolution), Image.Resampling.LANCZOS)

        # Calculate loss to check if it meets the threshold
        inputs_image = clip_processor(images=manipulated_img_pil, return_tensors="pt").to(device)
        inputs_text = clip_processor(text=["a young person"], return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs_image).float()
            text_features = clip_model.get_text_features(**inputs_text).float()

        similarity_attr = torch.nn.functional.cosine_similarity(image_features, text_features)
        loss = similarity_attr.mean().item()
        print(f"Loss: {loss}")

        if loss >= 0.027:
            valid_image = True
            axes[0].imshow(original_img.resize((resolution, resolution), Image.Resampling.LANCZOS))
            axes[0].set_title("Original Image")
            axes[0].axis('off')

            axes[1].imshow(manipulated_img_pil)
            axes[1].set_title("Manipulated Image")
            axes[1].axis('off')

            canvas.draw()
            gc.collect()
        else:
            z, c, original_img = generate_image()

def on_slider_change(attr):
    def callback(value):
        current_directions[attr] = float(value)
        update_image(resolution=256)
    return callback

def on_slider_release(attr):
    def callback(event):
        update_image(resolution=1024)
    return callback

def save_direction():
    torch.save(directions['age'], 'age_direction.pth')
    print("Age direction saved.")

def on_close():
    root.destroy()
    plt.close(fig)
    sys.exit()

root = tk.Tk()
root.title("StyleGAN2 Image Manipulation")
root.protocol("WM_DELETE_WINDOW", on_close)

# Add the sliders
for attr in directions.keys():
    scale = Scale(root, from_=-10, to=10, resolution=0.1, orient=HORIZONTAL, label=attr, command=on_slider_change(attr))
    scale.set(0)
    scale.pack()
    scale.bind("<ButtonRelease-1>", on_slider_release(attr))

# Add the save button
save_button = Button(root, text="Save Age Direction", command=save_direction)
save_button.pack()

# Add the new image button
new_image_button = Button(root, text="Generate New Image", command=new_image)
new_image_button.pack()

# Add the redefine age direction button
redefine_age_button = Button(root, text="Redefine Age Direction", command=redefine_age_direction)
redefine_age_button.pack()

# Initialize the canvas
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack()

# Initial image update
update_image(resolution=1024)

root.mainloop()
