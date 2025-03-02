import torch
import numpy as np
import PIL.Image
import dill
from clip import clip_model, clip_processor

pickle_file = "/app/HuggingFace/MonthOne/Model/model_v1.pkl"

def load_model():
    # load pickle file
    with open(pickle_file, 'rb') as f:
        data = dill.load(f)
    generator = data['G_ema']
    return generator

def encode_prompts(prompts, device):
    text_inputs = clip_processor(text=prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        text_features = clip_model.get_text_features(**text_inputs)
    return text_features

def generate_images(generator, device, prompts, num_images):
    generator.eval().to(device)

    z_dim=64 # seems to want 64
    z = torch.randn(num_images, z_dim, device=device)

    text_features = encode_prompts(prompts, device)

    with torch.no_grad():
        images = generator(z, text_features)

    # convert to PIL images
    images = (images.clamp(-1, 1) + 1) * 127.5
    images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    return [PIL.Image.fromarray(img) for img in images]

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = load_model()
    prompts = ["An elderly man with a hat"]
    images = generate_images(generator, device, prompts, 1)

    for i, img in enumerate(images):
        img.save(f'image_{i}.png')