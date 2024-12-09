import os
import json
from datetime import datetime
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import streamlit as st
from pathlib import Path

class ImageGenerator:
    def __init__(self):
        self.model_id = "runwayml/stable-diffusion-v1-5"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        self.pipe = self.pipe.to(self.device)
        
    def generate_image(self, prompt, num_inference_steps=50, guidance_scale=7.5, seed=None):
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
            
        image = self.pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]
        
        return image
    
    def save_image_with_metadata(self, image, prompt, output_dir="images", metadata_dir="metadata"):
        # Create directories if they don't exist
        Path(output_dir).mkdir(exist_ok=True)
        Path(metadata_dir).mkdir(exist_ok=True)
        
        # Generate timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = f"generated_{timestamp}.png"
        metadata_filename = f"metadata_{timestamp}.json"
        
        # Save image
        image_path = os.path.join(output_dir, image_filename)
        image.save(image_path)
        
        # Save metadata
        metadata = {
            "timestamp": timestamp,
            "prompt": prompt,
            "model_id": self.model_id,
            "device": self.device,
            "image_path": image_path
        }
        
        metadata_path = os.path.join(metadata_dir, metadata_filename)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
            
        return image_path, metadata_path

def main():
    st.title("LLM Image Generator")
    
    # Initialize the image generator
    generator = ImageGenerator()
    
    # User input
    prompt = st.text_area("Enter your prompt:", "A beautiful sunset over a mountain lake, digital art")
    num_steps = st.slider("Number of inference steps:", min_value=20, max_value=100, value=50)
    guidance = st.slider("Guidance scale:", min_value=1.0, max_value=20.0, value=7.5)
    
    # Handle seed input with a checkbox to enable/disable
    use_random_seed = st.checkbox("Use custom seed", value=False)
    seed = None
    if use_random_seed:
        seed = st.number_input("Enter seed value:", min_value=0, value=42)
    
    if st.button("Generate Image"):
        with st.spinner("Generating image..."):
            # Generate the image
            image = generator.generate_image(
                prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance,
                seed=seed
            )
            
            # Save image and metadata
            image_path, metadata_path = generator.save_image_with_metadata(image, prompt)
            
            # Display the generated image
            st.image(image, caption=f"Generated image: {prompt}")
            st.success(f"Image saved to: {image_path}")
            st.success(f"Metadata saved to: {metadata_path}")

if __name__ == "__main__":
    main() 