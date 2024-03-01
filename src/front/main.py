import os
import json
from datetime import datetime

import numpy as np
from PIL import Image

import torch

import streamlit as st

from diffusers import StableDiffusionPipeline



import requests
import urllib.parse as parse

# a function to determine whether a string is a URL or not
def is_url(string):
    try:
        result = parse.urlparse(string)
        return all([result.scheme, result.netloc, result.path])
    except:
        return False

# a function to load an image
def load_image(image_path):
    if is_url(image_path):
        return Image.open(requests.get(image_path, stream=True).raw)
    elif os.path.exists(image_path):
        return Image.open(image_path)
    
# def proportional_resize_img(image, width_target):
#     width_original, height_original = image.size
#     proportional = width_original / height_original
#     height_target = int(width_original * height_original)
#     return image.resize((width_target, height_target))
    
def resize_img(image, width_target, height_target):
    width_original, height_original = image.size
    proportional = width_original / height_original
    new_height = int(height_target / proportional)
    return image.resize((width_target, new_height))

def build_front_env():

    # Models
    options = [
        'CompVis/stable-diffusion-v1-4',
        'runwayml/stable-diffusion-v1-5',
        'stabilityai/stable-diffusion-2-1',
        'dreamlike-art/dreamlike-photoreal-2.0',
        # 'stabilityai/stable-diffusion-2-depth',
    ]

    selected_option_model = st.selectbox('Select model', options)

    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Device
    options = ['cpu', 'cuda']

    selected_option_device = st.selectbox('Select device', options, index=options.index(default_device))

    # # input image
    # input_img_url = st.text_input('Input image', '')

    # prompts
    prompt = st.text_input('Prompt', 'Cyber lizard in space')

    prompt_neg = st.text_input('Prompt (neg)', 'photorealism')

    # params

    seed = int(st.text_input('Initial seed', '42'))
    strength = float(st.text_input('Strength', '0.7'))
    width = int(st.text_input('Width', '256'))
    height = int(st.text_input('Height', '256'))
    num_inference_steps = int(st.text_input('Num. inference steps', '10'))

    # button execute
    if st.button('Generate'):

        # if input_img_url != '':
        #     img = load_image(input_img_url)
        #     img = resize_img(img, width, height)
        # else:
        #     img = None

        torch.manual_seed(seed)

        pipe = StableDiffusionPipeline.from_pretrained(
            selected_option_model,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            safety_checker = None,
            requires_safety_checker = False,            
        )
        
        pipe.to(selected_option_device)

        images = pipe(
            prompt,
            negative_prompt=prompt_neg,
            num_inference_steps=num_inference_steps,
            strength=strength,
            width = width,
            height = height,
            # image = img,
        ).images

        image = images[0]

        st.image(image, caption='Generated image', use_column_width=True)

        filename = int(datetime.now().timestamp() * 1000)

        os.makedirs(f'src/assets/output/{ filename }')

        image.save(f'src/assets/output/{ filename }/image.png')

        metadata = {
            'model': selected_option_model,
            'device': selected_option_device,
            'prompt': prompt,
            'prompt_neg': prompt_neg,
            'seed': seed,
            'width': width,
            'height': height,
            'num_inference_steps': num_inference_steps,
            'strength': strength,
            # 'input_img_url': input_img_url if input_img_url != '' else 'no used'
        }

        with open(f'src/assets/output/{ filename }/metadata.json', 'w') as file:
            json.dump(metadata, file)

        del pipe
