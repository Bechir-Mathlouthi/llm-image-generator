# LLM Image Generator

This project uses Stable Diffusion to generate high-quality images from text prompts. It provides a simple Streamlit interface for image generation and management.

## Features

- Text-to-image generation using Stable Diffusion
- User-friendly Streamlit interface
- Image metadata tracking
- Image gallery visualization
- Customizable generation parameters

## Installation

1. Clone this repository
2. Create a virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit interface:
```bash
streamlit run scripts/generate_images.py
```

2. Enter your text prompt in the interface
3. Adjust generation parameters if desired:
   - Number of inference steps (higher = better quality, slower generation)
   - Guidance scale (higher = stronger adherence to prompt)
   - Random seed (optional, for reproducible results)
4. Click "Generate Image" to create your image
5. Generated images are saved in the `images` folder with metadata in the `metadata` folder

## Project Structure

```
project_root/
├── scripts/
│   ├── generate_images.py    # Main script for image generation
│   ├── utils.py             # Utility functions
├── images/                  # Generated images storage
├── metadata/               # Image metadata storage
├── README.md              # This file
└── requirements.txt       # Python dependencies
```

## Requirements

- Python 3.7+
- CUDA-capable GPU recommended (but not required)
- See requirements.txt for Python package dependencies

## License

This project is open source and available under the MIT License. 