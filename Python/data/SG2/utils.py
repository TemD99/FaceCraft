
import sys
if '/app' not in sys.path:
    sys.path.append('/app')
import torch
from torchvision.utils import save_image
import os
from torch import nn
from math import sqrt
import torch.nn.functional as F 
import numpy as np
from config import *
import datetime
import pickle
from models.clip import process_text, description_to_attributes, clip_model
import torch
import clip


class ProjectTextFeatures(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ProjectTextFeatures, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)
    
    
# Sample W
#   samples Z randomly and gets W from the mapping network
def get_w(batch_size, mapping_network, text_features, project_text_features):
    z_dim = mapping_network.mapping[0].in_features
    text_features_dim = W_DIM // 2  # Projected text features dimension
    noise_dim = z_dim - text_features_dim

    if noise_dim <= 0:
        raise ValueError(f"Dimension mismatch: z_dim ({z_dim}) is less than or equal to text_features_dim ({text_features_dim}).")

    z = torch.randn(batch_size, noise_dim).to(DEVICE)

    # Debugging
    print(f"z shape: {z.shape}")
    print(f"text_features shape: {text_features.shape}")

    # Project text features to a smaller dimension
    text_features_projected = project_text_features(text_features)
    
    if text_features_projected.size(0) != batch_size:
        raise ValueError(f"Batch size mismatch: text_features_projected has batch size {text_features_projected.size(0)}, but expected {batch_size}")

    z = torch.cat((z, text_features_projected), dim=1)

    # Debugging
    print(f"z shape after concatenation: {z.shape}")

    w = mapping_network(z)
    return w


# Generate noise
#   generates noise for each generator block
def get_noise(batch_size):
    noise = []
    resolution = 4

    for i in range(LOG_RESOLUTION):
        # Generate noise for each block, ensuring no None values are used
        if i == 0:
            n1 = None
        else:
            n1 = torch.randn(batch_size, 1, resolution, resolution, device=DEVICE)
        n2 = torch.randn(batch_size, 1, resolution, resolution, device=DEVICE)

        noise.append((n1, n2))

        resolution *= 2

    return noise
# goal of this function is to generate n fake images and save them as a result for each epoch
def generate_examples(gen, epoch, mapping_network, text_descriptions, n=100):
    gen.eval()
    for i in range(n):
        with torch.no_grad():
            # Convert text descriptions to attribute vectors
            description = text_descriptions[i % len(text_descriptions)]
            attribute_vector = description_to_attributes(description)
            attribute_tensor = process_text(attribute_vector, clip_model, DEVICE)

            # Generate the text features using CLIP
            text_features = process_text(description, clip_model, DEVICE)

            # Generate w and noise
            w = get_w(1, mapping_network, text_features, project_text_features)
            noise = get_noise(1)
            
            # Generate the image
            img = gen(w, text_features, attribute_tensor)
            
            # Save the generated image
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            if not os.path.exists(f'Python/data/SG2/saved_training/saved_examples_main/{timestamp}/epoch{epoch}'):
                os.makedirs(f'Python/data/SG2/saved_training/saved_examples_main/{timestamp}/epoch{epoch}', exist_ok=True)
            save_image(img*0.5+0.5, f"Python/data/SG2/saved_training/saved_examples_main/{timestamp}/epoch{epoch}/img_{i}.png")

    gen.train()

def export_model_to_onnx(model, input_shape, epoch, w_shape, directory="/app/Python/data/SG2/saved_training/saved_models", provided_filename="model_test"):
    model.eval()  # Ensure the model is in eval mode for export
    os.makedirs(directory, exist_ok=True)
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    filename = f"{provided_filename}_{date}_epoch_{epoch}.onnx"
    export_path = os.path.join(directory, filename)
    
    # Generate dummy inputs that match the expected input sizes
    dummy_img_input = torch.randn(input_shape, device=DEVICE)
    dummy_w = torch.randn(w_shape, device=DEVICE)  

    # Export the model
    torch.onnx.export(model, (dummy_img_input, dummy_w), export_path)  # Pass both inputs to the model
    print(f"Model exported to ONNX format at {export_path}")

# Example usage:
# export_model_to_onnx(gen, (1, 3, 128, 128), epoch, (1, W_DIM))

def export_model(model, checkpoint_path, onnx_path):
    # Initialize the model (adjust according to your model's architecture)
    w_shape = (1, W_DIM)
    input_shape = (1, 3, 128, 128) 
    
    # Load the model weights from checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    if 'gen_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['gen_state_dict'])  # Ensure the key matches your checkpoint's dictionary keys
    else:
        raise KeyError("No generator state dict found in checkpoint.")
    model.to(DEVICE)
    # Ensure model is in eval mode
    model.eval()

    
    dummy_input = torch.randn(input_shape, device=DEVICE)  # Adjust the size as per your needs
    dummy_w = torch.randn(w_shape, device=DEVICE)  # Adjust the size as per your needs

    # Export the model
    torch.onnx.export(model, (dummy_input, dummy_w), onnx_path, export_params=True, opset_version=12, verbose=True)

    print(f'Model exported to ONNX format at {onnx_path}')

def load_checkpoint(model, optimizer, path, component):
    """Load model and optimizer states from a checkpoint file."""
    checkpoint = torch.load(path)
    model_state_dict = checkpoint.get(f'{component}_state_dict')

    if 'state' in model_state_dict or 'param_groups' in model_state_dict:
        raise ValueError("It appears optimizer state is being loaded into the model. Check the keys.")

    if model_state_dict:
        missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
        print("Missing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)
    else:
        raise KeyError(f"Missing {component}_state_dict in checkpoint")

    optimizer_state_dict = checkpoint.get(f'opt_{component}_state_dict')
    if optimizer_state_dict:
        optimizer.load_state_dict(optimizer_state_dict)
    start_epoch = checkpoint['epoch']
    return model, optimizer, start_epoch

def export_model_to_pickle(model, optimizer, epoch, directory="/app/Python/data/SG2/saved_training/exported_pickles", provided_filename="Test_model_01"):
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Define the filename and path
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    filename = f"{provided_filename}_{date}_epoch_{epoch}.pkl"
    path = os.path.join(directory, filename)
    
    # Save the model state dictionary, optimizer state, and the epoch number in a dictionary
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    
    # Use pickle to serialize the checkpoint dictionary
    with open(path, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    print(f"Checkpoint saved as a pickle file at {path}")

def save_checkpoint(gen, critic, mapping_network, opt_gen, opt_critic, opt_mapping_network, epoch, directory="/app/Python/data/SG2/saved_training/saved_checkpoints", provided_filename="unnamed_1_"):
    os.makedirs(directory, exist_ok=True)
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    filename = f"{provided_filename}_{date}_epoch_{epoch}.pth"
    path = os.path.join(directory, filename)
    model_directory = "/app/Python/data/SG2/saved_training/saved_models"
    
    # Saving state dictionaries
    torch.save({
        'epoch': epoch,
        'gen_state_dict': gen.state_dict(),
        'critic_state_dict': critic.state_dict(),
        'mapping_network_state_dict': mapping_network.state_dict(),
        'opt_gen_state_dict': opt_gen.state_dict(),
        'opt_critic_state_dict': opt_critic.state_dict(),
        'opt_mapping_network_state_dict': opt_mapping_network.state_dict()
    }, path)
    
    # Saving entire models
    torch.save(gen, os.path.join(model_directory, f"{provided_filename}_gen_full_{date}_epoch_{epoch}.pt"))
    torch.save(critic, os.path.join(model_directory, f"{provided_filename}_critic_full_{date}_epoch_{epoch}.pt"))
    torch.save(mapping_network, os.path.join(model_directory, f"{provided_filename}_mapping_network_full_{date}_epoch_{epoch}.pt"))
    torch.save(gen.state_dict(), os.path.join(model_directory, f"{provided_filename}_model_weights_{date}_epoch_{epoch}.pt"))

    print(f"Checkpoint saved to {directory} and full models saved to {model_directory}")


class CLIPTextEncoder:
    def __init__(self, model_name="ViT-B/32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)

    def encode_text(self, text):
        text_input = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_input)
        return text_features