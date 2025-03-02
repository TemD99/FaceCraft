import pickle
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import nltk
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import random
from transformers import CLIPProcessor, CLIPModel
from transformers import MarianMTModel, MarianTokenizer

from langdetect import detect
from matplotlib import pyplot as plt
import numpy as np
import os
import sys
import keyboard
import ninja
import time

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Append the path to the StyleGAN and torch_utils
sys.path.append('D:\\Flickr_Images\\ClipGanIntegration\\stylegan2-ada-pytorch-main')

# Ensure you have the WordNet data


# Setup the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

# Load CLIP models
clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(device)
clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

# Define an augmentation pipeline
augmentations = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(),
])

translation_models = {
    'fr': 'Helsinki-NLP/opus-mt-fr-en',  # French to English
    'es': 'Helsinki-NLP/opus-mt-es-en',  # Spanish to English
    'de': 'Helsinki-NLP/opus-mt-de-en',  # German to English
    'ja': 'Helsinki-NLP/opus-mt-ja-en',  # Japanese to English
    'zh': 'Helsinki-NLP/opus-mt-zh-en',  # Chinese to English
    'pt': 'Helsinki-NLP/opus-mt-pt-en',  # Portuguese to English
    # Add other languages as needed
}

# Function to load a pre-trained GAN model
def load_pretrained_gan_model(path_to_pkl):
    with open(path_to_pkl, 'rb') as f:
        gan_model = pickle.load(f)['G_ema']
        return gan_model.to(device)

# Load the GAN model
gan_model = load_pretrained_gan_model(r'D:\Flickr_Images\ClipGanIntegration\ffhq.pkl')

# Define tensor normalization for CLIP
def normalize_tensor(image_tensor):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
    return (image_tensor - mean) / std

# Function to generate initial latent vectors and text features
def generate_initial_latent(text, clip_model, clip_processor, num_samples=1):
    latents = []
    scores = []
    inputs = clip_processor(text=text, return_tensors="pt", padding=True).to(device)
    text_features = clip_model.get_text_features(**inputs)
    for _ in range(num_samples):
        latent = torch.randn(1, 512, device=device)
        image = gan_model(latent, None)
        image = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
        image = normalize_tensor(image)
        image_features = clip_model.get_image_features(image)
        score = torch.cosine_similarity(text_features, image_features).mean()
        latents.append(latent)
        scores.append(score.item())
    best_indices = np.argsort(scores)[-5:]  # Select top 5 latents
    return [latents[i] for i in best_indices]

def filter_text(description):
    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize(description)
    filtered_words = [word for word in words if word.lower() not in stop_words and word.isalnum()]
    return ' '.join(filtered_words)
# Function to optimize latent vectors with AugCLIP-

def optimize_latent(latents, text_features, clip_model, gan_model, iterations=1, initial_lr=5):
    optimizer = torch.optim.Adam(latents, lr=initial_lr)
    for i in range(iterations):
        optimizer.zero_grad()
        loss = 0
        for latent in latents:
            image = gan_model(latent, None)
            image = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
            image = normalize_tensor(image)
            for _ in range(5): 
                aug_image = augmentations(image)
                image_features = clip_model.get_image_features(aug_image)
                loss += -torch.cosine_similarity(text_features, image_features).mean()
        loss = loss / (len(latents) * 5)
        loss.backward(retain_graph=True if i < iterations - 1 else False)
        optimizer.step()
        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss.item()}")
    return image

def generate_image(text_description):
    initial_latents = generate_initial_latent(text_description, clip_model, clip_processor, num_samples=50)
    inputs = clip_processor(text=text_description, return_tensors="pt", padding=True).to(device)
    text_features = clip_model.get_text_features(**inputs)
    final_image = optimize_latent(initial_latents, text_features, clip_model, gan_model)
    loss = -torch.cosine_similarity(text_features, clip_model.get_image_features(final_image)).mean()
   
    final_image_np = final_image.squeeze().permute(1,2, 0).detach().cpu().numpy()
    final_image_np = (final_image_np - final_image_np.min()) / (final_image_np.max() - final_image_np.min())
    print(f"Loss: {loss.item()}")
    return final_image_np
  

def translate_text(text):
    """Detects the text's language and translates it to English if necessary."""
    lang = detect(text)
    model_name = translation_models.get(lang)
    if model_name and lang != 'en':  # Assuming 'en' is English and does not need translation
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
        text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return text



def filter_text(description):
    # Words and phrases to exclude
    exclude_phrases = ['please', 'draw', 'create', 'depict', 'make', 'show', 'portray']
    # Convert to lowercase for case-insensitive matching
    description = description.lower()
    
    # Remove specific phrases
    for phrase in exclude_phrases:
        description = description.replace(phrase, '')

    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize(description)
    filtered_words = [word for word in words if not word.lower() in stop_words and word.isalnum()]
    
    # Reassemble the description
    return ' '.join(filtered_words)


# Example usage in your main function
def main():
    while True:
        try:
            if keyboard.is_pressed('g'):
                text_description = input("Enter a text description: ")
                # Translate description to English if necessary
                english_description = translate_text(text_description)
                print(f"English description: {english_description}")
                
                # Filter out unnecessary words and refine
                filtered_description = filter_text(english_description)
                print(f"Filtered description: {filtered_description}")
                
                # Generate image
                image = generate_image(filtered_description)
                plt.imshow(image)
                plt.axis('off')
                plt.show()
        except Exception as e:
            print(f"An error occurred: {e}")
            break  # Exit the loop if an error occurs

if __name__ == "__main__":
    main()
