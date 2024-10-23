import gradio as gr
import gradio as gr
from groq import Groq
import os
from deep_translator import GoogleTranslator
from deep_translator import GoogleTranslator  # Import the GoogleTranslator class
import whisper
import gradio as gr
from groq import Groq
import os
from deep_translator import GoogleTranslator # Import the GoogleTranslator class
import pickle
import whisper
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
import matplotlib.pyplot as plt
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


# Replace with your actual API key
api_key = "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"
client = Groq(api_key=api_key)

# Load the custom model for image generation
base = "stabilityai/stable-diffusion-xl-base-1.0"
repo = "ByteDance/SDXL-Lightning"
ckpt = "sdxl_lightning_4step_unet.safetensors"  # Ensure the correct checkpoint

# Load the custom UNet and set up the pipeline
unet = UNet2DConditionModel.from_config(base, subfolder="unet").to("cpu", torch.float16)
unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device="cpu"))
pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16").to("cpu")
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")


# Function to transcribe, translate, and generate an image
def process_audio(audio_path, generate_image):
    if audio_path is None:
        return "Please upload an audio file.", None, None

    # Step 1: Transcribe audio
    try:
        with open(audio_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(audio_path), file.read()),
                model="whisper-large-v3",
                language="ta",
                response_format="verbose_json",
            )
        tamil_text = transcription.text
    except Exception as e:
        return f"An error occurred during transcription: {str(e)}", None, None

    # Step 2: Translate Tamil to English
    try:
        translator = GoogleTranslator(source='ta', target='en')
        translation = translator.translate(tamil_text)
    except Exception as e:
        return tamil_text, f"An error occurred during translation: {str(e)}", None

    # Step 3: Generate image (if selected)
    if generate_image:
        try:
            # Use the custom model and pipeline to generate an image
            img = pipe(translation, num_inference_steps=4, guidance_scale=0).images[0]
            return tamil_text, translation, img
        except Exception as e:
            return tamil_text, translation, f"An error occurred during image generation: {str(e)}"

    return tamil_text, translation, None


# Function for direct prompt to image generation
def generate_image_from_prompt(prompt):
    try:
        img = pipe(prompt, num_inference_steps=4, guidance_scale=0).images[0]
        return img
    except Exception as e:
        return f"An error occurred during image generation: {str(e)}"


# Assuming your 'process_audio' and 'generate_image_from_prompt' functions are defined elsewhere

# Gradio interface with the requested customizations
with gr.Blocks(css="""
    .gradio-container {background-color: #D8D2C2;} 
    .btn-red {background-color: red; color: white;} 
    .gr-button:hover {color: white !important;} 
    .gr-button {color: black !important;} 
    .gr-textbox {color: black !important;}
    .gr-Tab {color: black !important;}  /* Tab text color set to black */
""") as iface:
    
    # Title
    gr.Markdown("<h1 style='text-align: center; color:black;'>TransArt - Multimodal Application</h1>")

    # First Tab: Audio to Text -> Image
    with gr.Tab("Audio to Text"):
        gr.Markdown("<h3 style='text-align: center; color:black;'>Upload audio file, translate and generate an image</h3>")
        
        # Audio input and processing button
        with gr.Row():
            audio_input = gr.Audio(type="filepath", label="Upload Audio File")
            generate_image_checkbox = gr.Checkbox(label="Generate Image", value=False)
        
        # Outputs for transcription, translation, and image
        outputs = [
            gr.Textbox(label="Tamil Transcription"),
            gr.Textbox(label="English Translation"),
            gr.Image(label="Generated Image")  # Expecting an image output
        ]
        
        # Button for processing audio
        btn = gr.Button("Proceed Audio", elem_classes="btn-red")
        # Bind the correct function that returns transcription, translation, and an image
        btn.click(fn=process_audio, inputs=[audio_input, generate_image_checkbox], outputs=outputs)
    
    # Second Tab: Direct Prompt to Image Generation
    with gr.Tab("Prompt to Image"):
        gr.Markdown("<h3 style='text-align: center; color:black;'>Input a prompt and generate an image</h3>")
        
        # Text input for the prompt
        prompt_input = gr.Textbox(label="Enter Prompt", placeholder="Enter the scene description here...", lines=5)

        # Image output
        image_output = gr.Image(label="Generated Image")  # Expecting an image output

        # Button for generating the image from the prompt
        btn_image = gr.Button("Proceed Image Generation", elem_classes="btn-red")
        # Bind the correct function that returns an image
        btn_image.click(fn=generate_image_from_prompt, inputs=prompt_input, outputs=image_output)

# Launch the interface
iface.launch(server_name="0.0.0.0")
