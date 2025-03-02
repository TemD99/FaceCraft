import gradio as gr
import numpy as np
from PIL import Image
from css import css
from generate import load_model, generate_images
import torch

gr.set_static_paths(['.'])
title = 'FaceCraft Month One'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = load_model()

def create_blank():
    image = Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8))
    return [image]

def generate_from_prompt(prompts):
    num_images = 1
    images = generate_images(generator, device, prompts, num_images)
    return images

def input_prompt(input):
    prompt = input
    # if using prompt, ensure input isnt empty and is only alphabetic chars
    if not prompt.strip():
        return ["Prompt cannot be blank.", []]
    elif not prompt.replace(' ', '').isalpha():
        return ["Prompt can only contain alphabetic characters and spaces.", []]

    images = generate_from_prompt(prompt)

    # testing wise show the prompt and blank images
    return prompt, images

with gr.Blocks(title=title,css=css) as demo:
    gr.HTML("""
                <header>
                    <img class="logo" src="file/logo.png" />
                    <h1>FaceCraft</h1>
                </header>
            """)

    with gr.Row():
        label = gr.Label(value="Waiting for Prompt...")
        gallery = gr.Gallery(interactive=False)
    
    prompt_input = gr.Textbox(lines=1, placeholder="Enter your prompt here...", visible=True, show_label=False)

    with gr.Row():
        btn_generate = gr.Button("Generate", scale=0, min_width=200)
        btn_generate.click(fn=input_prompt, inputs=[prompt_input], outputs=[label,gallery])


    gr.HTML("""
                <footer class="custom-footer">
                    <p>Developed By</p>
                    <p>Will Hoover | Temitayo Shorunke | Ethan Stanks</p>
                </footer>
            """)

demo.launch()