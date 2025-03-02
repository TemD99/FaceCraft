import os
import sys
import pickle
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import nltk
from nltk.corpus import stopwords
from transformers import CLIPProcessor, CLIPModel, MarianMTModel, MarianTokenizer
from matplotlib import pyplot as plt
import numpy as np
from config import DEVICE

# Ensure you have the WordNet data
def setup_nltk():
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt')
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Append the path to the StyleGAN and torch_utils
sys.path.append('/app/Python/data/SG2')

# Setup the device
device = DEVICE

CLIP_MODEL = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(device)
CLIP_PROCESSOR = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
clip_model = CLIP_MODEL.to(device)
clip_processor = CLIP_PROCESSOR

# Define an augmentation pipeline
augmentations = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(),
])

# Function to load a pre-trained GAN model
def load_pretrained_gan_model(gen, PKL_PATH=None):
    with open(PKL_PATH, 'rb') as f:
        checkpoint = pickle.load(f)
        model = gen 
        model.load_state_dict(checkpoint['gen_state_dict'])
        return model.to(device)  

# Function to map descriptions to attributes
def description_to_attributes(description):
    mappings = {
        "middle aged": {"Young": -1},
        "young": {"Young": 1},
        "old": {"Young": -1, "Gray_Hair": 1},
        "Asian": {"Blond_Hair": -1, "Black_Hair": 1},
        "Caucasian": {"Blond_Hair": 1, "Black_Hair": -1},
        "man": {"Male": 1},
        "woman": {"Male": -1},
        "attractive": {"Attractive": 1},
        "unattractive": {"Attractive": -1},
        "smiling": {"Smiling": 1},
        "serious": {"Smiling": -1},
        "black hair": {"Black_Hair": 1},
        "blonde hair": {"Blond_Hair": 1},
        "brown hair": {"Brown_Hair": 1},
        "gray hair": {"Gray_Hair": 1},
        "bald": {"Bald": 1, "Hair": -1},
        "wearing glasses": {"Eyeglasses": 1},
        "no glasses": {"Eyeglasses": -1},
        "mustache": {"Mustache": 1},
        "no mustache": {"Mustache": -1},
        "beard": {"No_Beard": -1, "Goatee": 1},
        "no beard": {"No_Beard": 1, "Goatee": -1},
        "sideburns": {"Sideburns": 1},
        "no sideburns": {"Sideburns": -1},
        "wearing makeup": {"Heavy_Makeup": 1},
        "no makeup": {"Heavy_Makeup": -1},
        "pale skin": {"Pale_Skin": 1},
        "rosy cheeks": {"Rosy_Cheeks": 1},
        "big lips": {"Big_Lips": 1},
        "big nose": {"Big_Nose": 1},
        "pointy nose": {"Pointy_Nose": 1},
        "chubby": {"Chubby": 1},
        "thin": {"Chubby": -1},
        "bushy eyebrows": {"Bushy_Eyebrows": 1},
        "arched eyebrows": {"Arched_Eyebrows": 1},
        "narrow eyes": {"Narrow_Eyes": 1},
        "bags under eyes": {"Bags_Under_Eyes": 1},
        "double chin": {"Double_Chin": 1},
        "high cheekbones": {"High_Cheekbones": 1},
        "receding hairline": {"Receding_Hairline": 1},
        "wearing hat": {"Wearing_Hat": 1},
        "wearing earrings": {"Wearing_Earrings": 1},
        "wearing lipstick": {"Wearing_Lipstick": 1},
        "wearing necklace": {"Wearing_Necklace": 1},
        "wearing necktie": {"Wearing_Necktie": 1},
        "mouth slightly open": {"Mouth_Slightly_Open": 1},
        "straight hair": {"Straight_Hair": 1, "Wavy_Hair": -1},
        "wavy hair": {"Wavy_Hair": 1, "Straight_Hair": -1},
        "blurry": {"Blurry": 1},
        "not blurry": {"Blurry": -1},
    }

    attribute_vector = {attr: 0 for attr in [
        "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald", 
        "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", 
        "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair", 
        "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache", 
        "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose", 
        "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", 
        "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", 
        "Wearing_Necklace", "Wearing_Necktie", "Young"
    ]}

     # Apply the mappings
    for key, value in mappings.items():
        if key in description:
            for attr, val in value.items():
                attribute_vector[attr] = val

    return attribute_vector

# Process text into CLIP text features
def process_text(clip_model, user_input, project_text_features, device=DEVICE):
    if isinstance(user_input, str):
        inputs = clip_processor(text=user_input, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            text_features = clip_model.get_text_features(**inputs)
        return text_features
    elif isinstance(user_input, torch.Tensor):
        return user_input.to(device)
    else:
        raise ValueError("Input must be either a text description (str) or a tensor of attribute values (torch.Tensor).")


# Define tensor normalization for CLIP
def normalize_tensor(image_tensor):
    mean = torch.tensor([0.5211, 0.4260, 0.3812], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.2469, 0.2210, 0.2182], device=device).view(1, 3, 1, 1)
    return (image_tensor - mean) / std

# Function to generate initial latent vectors and text features
def generate_initial_latent(text, clip_model, clip_processor, num_samples=1, GAN_MODEL=None):
    if GAN_MODEL is None:
        print("Please provide a pre-trained GAN model")
        return
    latents = []
    scores = []
    inputs = clip_processor(text=text, return_tensors="pt", padding=True).to(device)
    text_features = clip_model.get_text_features(**inputs)
    for _ in range(num_samples):
        latent = torch.randn(1, 256, device=device)
        image = GAN_MODEL(latent, None)
        image = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
        image = normalize_tensor(image)
        image_features = clip_model.get_image_features(image)
        score = torch.cosine_similarity(text_features, image_features).mean()
        latents.append(latent)
        scores.append(score.item())
    best_indices = np.argsort(scores)[-5:]  # Select top 5 latents
    return [latents[i] for i in best_indices]

# Optimize the latent vectors based on text features
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

# Generate image based on text description
def generate_image(text_description, generator, gan_model=None):
    gan_models = load_pretrained_gan_model(generator, gan_model)
    initial_latents = generate_initial_latent(text_description, clip_model, clip_processor, num_samples=50)
    inputs = clip_processor(text=text_description, return_tensors="pt", padding=True).to(device)
    text_features = clip_model.get_text_features(**inputs)
    final_image = optimize_latent(initial_latents, text_features, clip_model, gan_models)
    loss = -torch.cosine_similarity(text_features, clip_model.get_image_features(final_image)).mean()
   
    final_image_np = final_image.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    final_image_np = (final_image_np - final_image_np.min()) / (final_image_np.max() - final_image_np.min())
    print(f"Loss: {loss.item()}")
    return final_image_np

# Translate text to English if necessary
def translate_to_english(text, model_name="Helsinki-NLP/opus-mt-mul-en"):
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    if not text.isascii():
        translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
        text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return text

# Filter text descriptions
def filter_text(descriptions):
    stop_words = set(stopwords.words('english'))
    filtered_descriptions = []

    for description in descriptions:
        description = description.lower()
        words = nltk.word_tokenize(description)
        filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
        filtered_description = ' '.join(filtered_words)
        filtered_descriptions.append(filtered_description)

    return filtered_descriptions

# Process user input and generate image
def user_input_and_generate(user_input):
    if user_input.isascii():
        english_description = translate_to_english(user_input)
        print(f"English description: {english_description}")
        
        filtered_description = filter_text([english_description])[0]
        print(f"Final description: {filtered_description}")
        
        image = generate_image(filtered_description)
        plt.imshow(image)
        plt.axis('off')
        plt.show()
    else:
        print("Image processing not supported yet")
