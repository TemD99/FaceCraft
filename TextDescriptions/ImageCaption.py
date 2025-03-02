import os
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import time

# Initialize the model and tokenizer
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Set up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the generation parameters
max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

def predict_caption(image_path):
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert(mode="RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")
    pixel_values = inputs.pixel_values.to(device)
    attention_mask = inputs.get('attention_mask', None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    output_kwargs = gen_kwargs.copy()
    if attention_mask is not None:
        output_kwargs.update({"attention_mask": attention_mask})

    output_ids = model.generate(pixel_values, **output_kwargs)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption

def process_folder(folder_path, output_file, processed_log):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    processed_images = set()
    if os.path.exists(processed_log):
        with open(processed_log, 'r') as file:
            processed_images = set(file.read().splitlines())
    total_images = len(image_files)
    start_time = time.time()
    print(f"Processing {total_images} images...")

    with open(output_file, 'a') as file, open(processed_log, 'a') as log:
        for i, image_file in enumerate(image_files, 1):
            image_path = os.path.join(folder_path, image_file)
            if image_path in processed_images:
                continue
            try:
                print(f"Processing image {i}/{total_images}: {os.path.basename(image_path)}")
                caption = predict_caption(image_path)
                file.write(f"{os.path.basename(image_path)}: {caption}\n")
                log.write(image_path + '\n')
                file.flush()
                log.flush()

                elapsed_time = time.time() - start_time
                images_left = total_images - i
                estimated_time = (elapsed_time / i) * images_left
                print(f"Estimated time remaining: {estimated_time/60:.2f} minutes")
            except Exception as e:
                print(f"An error occurred while processing {image_path}: {e}")

# Replace 'your_image_folder_path' with the path to your image folder and 'output_captions.txt' with your desired output file name
process_folder('128x128', 'output_captions.txt', 'processed_images.log')